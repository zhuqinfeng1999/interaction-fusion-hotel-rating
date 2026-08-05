from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import psutil
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error


CLIP_MIN = 1.0
CLIP_MAX = 5.0
BOOTSTRAP_SEED = 42


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def prediction_metrics(
    y_true: np.ndarray,
    y_pred_raw: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred_raw = np.asarray(y_pred_raw, dtype=float)
    y_pred_clipped = np.clip(y_pred_raw, CLIP_MIN, CLIP_MAX)

    def calculate(predictions: np.ndarray) -> Dict[str, float]:
        return {
            "mae": float(mean_absolute_error(y_true, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        }

    return {
        "raw": calculate(y_pred_raw),
        "clipped": calculate(y_pred_clipped),
    }


def summarize_folds(
    fold_records: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for variant in ("raw", "clipped"):
        mae = np.asarray(
            [record[variant]["mae"] for record in fold_records],
            dtype=float,
        )
        rmse = np.asarray(
            [record[variant]["rmse"] for record in fold_records],
            dtype=float,
        )
        summary[variant] = {
            "mean": {
                "mae": float(mae.mean()),
                "rmse": float(rmse.mean()),
            },
            "std": {
                "mae": float(mae.std(ddof=1)),
                "rmse": float(rmse.std(ddof=1)),
            },
        }
    return summary


class PeakRSSMonitor:
    def __init__(self, interval_sec: float = 0.05) -> None:
        self.interval_sec = interval_sec
        self.process = psutil.Process()
        self.peak_bytes = self.process.memory_info().rss
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            try:
                self.peak_bytes = max(
                    self.peak_bytes,
                    self.process.memory_info().rss,
                )
            except psutil.Error:
                return

    def __enter__(self) -> "PeakRSSMonitor":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        try:
            self.peak_bytes = max(
                self.peak_bytes,
                self.process.memory_info().rss,
            )
        except psutil.Error:
            pass

    @property
    def peak_mb(self) -> float:
        return float(self.peak_bytes / (1024 * 1024))


@dataclass
class FoldArtifact:
    model_id: str
    fold: int
    prediction_path: Path
    metadata_path: Path
    record: Dict[str, Any]


class EvaluationHarness:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.predictions_dir = output_dir / "predictions"
        self.metrics_dir = output_dir / "metrics"
        self.per_hotel_dir = output_dir / "per_hotel"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.per_hotel_dir.mkdir(parents=True, exist_ok=True)

    def prediction_path(self, model_id: str, fold: int) -> Path:
        return self.predictions_dir / model_id / f"fold{fold}.parquet"

    def metadata_path(self, model_id: str, fold: int) -> Path:
        return self.predictions_dir / model_id / f"fold{fold}.meta.json"

    def has_fold(self, model_id: str, fold: int) -> bool:
        return self.prediction_path(model_id, fold).exists() and self.metadata_path(
            model_id, fold
        ).exists()

    def load_fold(self, model_id: str, fold: int) -> FoldArtifact:
        prediction_path = self.prediction_path(model_id, fold)
        metadata_path = self.metadata_path(model_id, fold)
        if not prediction_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"incomplete fold artifact: {model_id} fold {fold}")
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        return FoldArtifact(
            model_id=model_id,
            fold=fold,
            prediction_path=prediction_path,
            metadata_path=metadata_path,
            record=record,
        )

    def write_fold(
        self,
        model_id: str,
        fold: int,
        row_id: np.ndarray,
        offering_id: np.ndarray,
        y_true: np.ndarray,
        y_pred_raw: np.ndarray,
        fit_seconds: float,
        predict_seconds: float,
        peak_memory_mb: float,
        selected_hyperparameters: Dict[str, Any],
        preprocessing_seconds: float = 0.0,
        tuning_seconds: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> FoldArtifact:
        prediction_path = self.prediction_path(model_id, fold)
        metadata_path = self.metadata_path(model_id, fold)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)

        y_true = np.asarray(y_true, dtype=float)
        y_pred_raw = np.asarray(y_pred_raw, dtype=float)
        y_pred_clipped = np.clip(y_pred_raw, CLIP_MIN, CLIP_MAX)
        frame = pd.DataFrame(
            {
                "row_id": np.asarray(row_id, dtype=np.int64),
                "offering_id": np.asarray(offering_id),
                "y_true": y_true,
                "y_pred_raw": y_pred_raw,
                "y_pred_clipped": y_pred_clipped,
                "fold": np.full(len(y_true), fold, dtype=np.int8),
            }
        )
        if frame["row_id"].duplicated().any():
            duplicates = int(frame["row_id"].duplicated().sum())
            raise ValueError(
                f"{model_id} fold {fold} contains {duplicates} duplicate row_id values"
            )
        frame.to_parquet(prediction_path, index=False)

        metrics = prediction_metrics(y_true, y_pred_raw)
        record: Dict[str, Any] = {
            "fold": fold,
            **metrics,
            "n_test": int(len(frame)),
            "n_test_hotels": int(frame["offering_id"].nunique()),
            "preprocessing_seconds": float(preprocessing_seconds),
            "tuning_seconds": float(tuning_seconds),
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "peak_memory_mb": float(peak_memory_mb),
            "selected_hyperparameters": selected_hyperparameters,
            "prediction_path": str(prediction_path),
        }
        if extra:
            record["extra"] = extra
        write_json(metadata_path, record)
        return FoldArtifact(
            model_id=model_id,
            fold=fold,
            prediction_path=prediction_path,
            metadata_path=metadata_path,
            record=record,
        )

    def finalize_model(
        self,
        model_id: str,
        metadata: Dict[str, Any],
        expected_folds: Iterable[int] = range(1, 6),
    ) -> Dict[str, Any]:
        artifacts = [self.load_fold(model_id, fold) for fold in expected_folds]
        fold_records = [artifact.record for artifact in artifacts]
        summary = summarize_folds(fold_records)

        all_predictions = pd.concat(
            [
                pd.read_parquet(artifact.prediction_path)
                for artifact in artifacts
            ],
            ignore_index=True,
        )
        if all_predictions["row_id"].duplicated().any():
            duplicates = int(all_predictions["row_id"].duplicated().sum())
            raise ValueError(
                f"{model_id} has {duplicates} duplicated held-out predictions"
            )

        all_predictions["absolute_error_raw"] = np.abs(
            all_predictions["y_true"] - all_predictions["y_pred_raw"]
        )
        all_predictions["squared_error_raw"] = np.square(
            all_predictions["y_true"] - all_predictions["y_pred_raw"]
        )
        all_predictions["absolute_error"] = np.abs(
            all_predictions["y_true"] - all_predictions["y_pred_clipped"]
        )
        all_predictions["squared_error"] = np.square(
            all_predictions["y_true"] - all_predictions["y_pred_clipped"]
        )
        grouped = all_predictions.groupby(["offering_id", "fold"], sort=True)
        per_hotel = grouped.agg(
            n_reviews=("row_id", "size"),
            mae=("absolute_error", "mean"),
            mse=("squared_error", "mean"),
            mae_raw=("absolute_error_raw", "mean"),
            mse_raw=("squared_error_raw", "mean"),
        ).reset_index()
        per_hotel["rmse"] = np.sqrt(per_hotel.pop("mse"))
        per_hotel["rmse_raw"] = np.sqrt(per_hotel.pop("mse_raw"))
        per_hotel_path = self.per_hotel_dir / f"{model_id}.parquet"
        per_hotel.to_parquet(per_hotel_path, index=False)

        timing_fields = [
            "preprocessing_seconds",
            "tuning_seconds",
            "fit_seconds",
            "predict_seconds",
            "peak_memory_mb",
        ]
        timing_summary = {
            field: {
                "mean": float(
                    np.mean([record.get(field, 0.0) for record in fold_records])
                ),
                "total": float(
                    np.sum([record.get(field, 0.0) for record in fold_records])
                ),
            }
            for field in timing_fields
        }
        payload = {
            "model_id": model_id,
            "meta": metadata,
            "folds": fold_records,
            "aggregate": summary,
            "timing": timing_summary,
            "per_hotel_path": str(per_hotel_path),
            "n_predictions": int(len(all_predictions)),
            "n_hotels": int(per_hotel["offering_id"].nunique()),
        }
        write_json(self.metrics_dir / f"{model_id}.json", payload)
        return payload


def _paired_hotel_frame(
    first_path: Path,
    second_path: Path,
    metric: str = "mae",
) -> pd.DataFrame:
    first = pd.read_parquet(first_path, columns=["offering_id", metric]).rename(
        columns={metric: "first"}
    )
    second = pd.read_parquet(second_path, columns=["offering_id", metric]).rename(
        columns={metric: "second"}
    )
    paired = first.merge(
        second,
        on="offering_id",
        how="inner",
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError("no shared hotels for paired comparison")
    return paired


def paired_wilcoxon(
    first_path: Path,
    second_path: Path,
    metric: str = "mae",
) -> Dict[str, Any]:
    paired = _paired_hotel_frame(first_path, second_path, metric)
    differences = paired["first"].to_numpy() - paired["second"].to_numpy()
    if np.allclose(differences, 0.0):
        statistic = 0.0
        p_value = 1.0
    else:
        result = stats.wilcoxon(
            differences,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    return {
        "metric": metric,
        "difference": "first_minus_second",
        "n_hotels": int(len(paired)),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "statistic": statistic,
        "p_value": p_value,
    }


def bootstrap_mean_difference(
    first_path: Path,
    second_path: Path,
    metric: str = "mae",
    n_resamples: int = 10_000,
    seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 250,
) -> Dict[str, Any]:
    paired = _paired_hotel_frame(first_path, second_path, metric)
    differences = paired["first"].to_numpy() - paired["second"].to_numpy()
    rng = np.random.default_rng(seed)
    bootstrap_means: List[np.ndarray] = []
    for start in range(0, n_resamples, chunk_size):
        current_size = min(chunk_size, n_resamples - start)
        indices = rng.integers(
            0,
            len(differences),
            size=(current_size, len(differences)),
        )
        bootstrap_means.append(differences[indices].mean(axis=1))
    samples = np.concatenate(bootstrap_means)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "metric": metric,
        "difference": "first_minus_second",
        "n_hotels": int(len(paired)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "mean_difference": float(differences.mean()),
        "ci_95": [float(lower), float(upper)],
    }


def cluster_robust_ridge_se(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    alpha: float,
    feature_names: Sequence[str],
    include_intercept: bool = True,
) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups)
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional dense matrix")
    if x.shape[1] != len(feature_names):
        raise ValueError("feature_names length does not match x")
    if x.shape[1] > 2_000:
        raise ValueError("cluster robust covariance is limited to 2,000 features")

    if include_intercept:
        design = np.column_stack([np.ones(len(x), dtype=float), x])
        names = ["intercept", *feature_names]
        penalty = np.eye(design.shape[1], dtype=float) * float(alpha)
        penalty[0, 0] = 0.0
    else:
        design = x
        names = list(feature_names)
        penalty = np.eye(design.shape[1], dtype=float) * float(alpha)

    residuals = y_true - y_pred
    bread = np.linalg.pinv(design.T @ design + penalty)
    unique_groups, group_codes = np.unique(groups, return_inverse=True)
    score_rows = design * residuals[:, None]
    cluster_scores = np.zeros(
        (len(unique_groups), design.shape[1]),
        dtype=float,
    )
    np.add.at(cluster_scores, group_codes, score_rows)
    meat = cluster_scores.T @ cluster_scores
    covariance = bread @ meat @ bread

    n_samples = len(y_true)
    n_clusters = len(unique_groups)
    n_parameters = design.shape[1]
    if n_clusters > 1 and n_samples > n_parameters:
        correction = (n_clusters / (n_clusters - 1)) * (
            (n_samples - 1) / (n_samples - n_parameters)
        )
        covariance *= correction
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return pd.DataFrame(
        {
            "feature": names,
            "cluster_robust_se": standard_errors,
            "n_clusters": n_clusters,
            "alpha": float(alpha),
        }
    )
