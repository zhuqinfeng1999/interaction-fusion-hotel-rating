"""Generalized ridge on sparse text plus small dense blocks.

The implementation uses a centered ``LinearOperator`` and SciPy LSQR. Dense
block multipliers implement block-specific penalties without repeatedly
materializing hundreds of sparse matrix copies during grid search.
"""

from __future__ import annotations

import itertools
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, lsqr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold


def default_alpha_grid() -> np.ndarray:
    return np.logspace(-5, 9, 57, dtype=np.float64)


@dataclass
class SparseRidgeFit:
    theta: np.ndarray
    intercept: float
    coefficients: Dict[str, np.ndarray]
    alpha: float
    lambdas: Dict[str, float]
    iterations: int
    stop_code: int
    fit_seconds: float


class CenteredBlockOperator(LinearOperator):
    def __init__(
        self,
        text: sp.csr_matrix,
        blocks: Mapping[str, np.ndarray],
        lambdas: Mapping[str, float],
    ) -> None:
        self.text = text.tocsr()
        self.block_names = list(blocks)
        self.blocks = {
            name: np.asarray(blocks[name], dtype=np.float32)
            for name in self.block_names
        }
        self.lambdas = {
            name: float(lambdas.get(name, 1.0)) for name in self.block_names
        }
        n_rows = self.text.shape[0]
        if any(len(block) != n_rows for block in self.blocks.values()):
            raise ValueError("all blocks must have the same row count as text")
        self.slices: Dict[str, slice] = {}
        cursor = self.text.shape[1]
        for name in self.block_names:
            width = self.blocks[name].shape[1]
            self.slices[name] = slice(cursor, cursor + width)
            cursor += width
        self.text_slice = slice(0, self.text.shape[1])
        text_mean = np.asarray(self.text.mean(axis=0)).ravel().astype(np.float64)
        means = [text_mean]
        for name in self.block_names:
            means.append(
                self.blocks[name].mean(axis=0, dtype=np.float64)
                * self.lambdas[name]
            )
        self.feature_mean = np.concatenate(means)
        super().__init__(
            dtype=np.dtype(np.float64),
            shape=(n_rows, cursor),
        )

    def _matvec(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        result = np.asarray(
            self.text @ vector[self.text_slice], dtype=np.float64
        )
        for name in self.block_names:
            result += self.lambdas[name] * (
                self.blocks[name] @ vector[self.slices[name]]
            )
        result -= float(self.feature_mean @ vector)
        return result

    def _rmatvec(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        total = float(vector.sum(dtype=np.float64))
        parts = [
            np.asarray(self.text.T @ vector, dtype=np.float64)
            - self.feature_mean[self.text_slice] * total
        ]
        for name in self.block_names:
            raw = self.blocks[name].T @ vector
            selected = self.slices[name]
            parts.append(
                self.lambdas[name] * np.asarray(raw, dtype=np.float64)
                - self.feature_mean[selected] * total
            )
        return np.concatenate(parts)

    def uncentered_prediction(
        self,
        theta: np.ndarray,
        y_mean: float,
    ) -> np.ndarray:
        intercept = float(y_mean - self.feature_mean @ theta)
        result = np.asarray(
            self.text @ theta[self.text_slice], dtype=np.float64
        )
        for name in self.block_names:
            result += self.lambdas[name] * (
                self.blocks[name] @ theta[self.slices[name]]
            )
        return result + intercept

    def original_coefficients(self, theta: np.ndarray) -> Dict[str, np.ndarray]:
        output = {"text": np.asarray(theta[self.text_slice], dtype=np.float64)}
        for name in self.block_names:
            output[name] = (
                self.lambdas[name]
                * np.asarray(theta[self.slices[name]], dtype=np.float64)
            )
        return output

    def transform_warm_start(
        self,
        previous: Optional["CenteredBlockOperator"],
        previous_theta: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if previous is None or previous_theta is None:
            return None
        if self.shape[1] != previous.shape[1]:
            return None
        transformed = np.asarray(previous_theta, dtype=np.float64).copy()
        for name in self.block_names:
            if name not in previous.slices:
                return None
            original = (
                previous.lambdas[name] * previous_theta[previous.slices[name]]
            )
            transformed[self.slices[name]] = original / self.lambdas[name]
        return transformed


def fit_operator(
    operator: CenteredBlockOperator,
    y: np.ndarray,
    alpha: float,
    tolerance: float = 1e-4,
    max_iterations: int = 2_000,
    x0: Optional[np.ndarray] = None,
) -> SparseRidgeFit:
    y = np.asarray(y, dtype=np.float64)
    y_mean = float(y.mean())
    started = time.perf_counter()
    solution = lsqr(
        operator,
        y - y_mean,
        damp=math.sqrt(float(alpha)),
        atol=tolerance,
        btol=tolerance,
        iter_lim=max_iterations,
        show=False,
        x0=x0,
    )
    theta = np.asarray(solution[0], dtype=np.float64)
    intercept = float(y_mean - operator.feature_mean @ theta)
    return SparseRidgeFit(
        theta=theta,
        intercept=intercept,
        coefficients=operator.original_coefficients(theta),
        alpha=float(alpha),
        lambdas=dict(operator.lambdas),
        iterations=int(solution[2]),
        stop_code=int(solution[1]),
        fit_seconds=time.perf_counter() - started,
    )


def predict_blocks(
    text: sp.csr_matrix,
    blocks: Mapping[str, np.ndarray],
    fit: SparseRidgeFit,
) -> np.ndarray:
    prediction = np.asarray(
        text @ fit.coefficients["text"], dtype=np.float64
    )
    for name, block in blocks.items():
        prediction += np.asarray(block, dtype=np.float32) @ fit.coefficients[name]
    return prediction + fit.intercept


def _configuration_key(
    alpha: float,
    lambdas: Mapping[str, float],
) -> str:
    suffix = ",".join(f"{name}={lambdas[name]:g}" for name in sorted(lambdas))
    return f"alpha={alpha:g}" + (f"|{suffix}" if suffix else "")


def grouped_cv_configurations(
    text: sp.csr_matrix,
    blocks: Mapping[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    configurations: Sequence[Tuple[float, Dict[str, float]]],
    tolerance: float,
    max_iterations: int,
    log_prefix: str = "",
    n_jobs: int = 1,
) -> pd.DataFrame:
    splitter = GroupKFold(n_splits=3)
    # Repeated configurations would make the per-fold bookkeeping ambiguous.
    # Keep the first occurrence while preserving the warm-start path order.
    unique_configurations: List[Tuple[float, Dict[str, float]]] = []
    seen = set()
    for alpha, lambdas in configurations:
        key = _configuration_key(alpha, lambdas)
        if key not in seen:
            unique_configurations.append((float(alpha), dict(lambdas)))
            seen.add(key)
    configurations = unique_configurations
    records: Dict[str, Dict[str, Any]] = {}
    for alpha, lambdas in configurations:
        key = _configuration_key(alpha, lambdas)
        records[key] = {
            "configuration": key,
            "alpha": float(alpha),
            **{f"lambda_{name}": float(value) for name, value in lambdas.items()},
            "inner_mae": [],
            "inner_iterations": [],
            "inner_seconds": [],
            "inner_stop_codes": [],
        }

    inner_splits = list(splitter.split(text, y, groups=groups))

    def evaluate_inner(
        inner_fold: int,
        fit_idx: np.ndarray,
        valid_idx: np.ndarray,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        text_fit = text[fit_idx].tocsr()
        text_valid = text[valid_idx].tocsr()
        blocks_fit = {name: block[fit_idx] for name, block in blocks.items()}
        blocks_valid = {name: block[valid_idx] for name, block in blocks.items()}
        previous_operator: Optional[CenteredBlockOperator] = None
        previous_theta: Optional[np.ndarray] = None
        previous_alpha: Optional[float] = None
        inner_records: List[Dict[str, Any]] = []
        for config_index, (alpha, lambdas) in enumerate(configurations, start=1):
            operator = CenteredBlockOperator(text_fit, blocks_fit, lambdas)
            # Warm starts are most useful along a descending alpha path. For
            # lambda grids, convert the prior scaled coefficients to the new scale.
            warm = operator.transform_warm_start(
                previous_operator,
                previous_theta,
            )
            if previous_alpha is not None and alpha > previous_alpha * 100:
                warm = None
            fit = fit_operator(
                operator,
                y[fit_idx],
                alpha,
                tolerance=tolerance,
                max_iterations=max_iterations,
                x0=warm,
            )
            prediction = predict_blocks(text_valid, blocks_valid, fit)
            mae = float(mean_absolute_error(y[valid_idx], prediction))
            inner_records.append(
                {
                    "configuration": _configuration_key(alpha, lambdas),
                    "mae": mae,
                    "iterations": fit.iterations,
                    "seconds": fit.fit_seconds,
                    "stop_code": fit.stop_code,
                }
            )
            previous_operator = operator
            previous_theta = fit.theta
            previous_alpha = alpha
            if config_index % 25 == 0 or config_index == len(configurations):
                print(
                    f"{log_prefix} inner {inner_fold}: "
                    f"{config_index}/{len(configurations)} configs",
                    flush=True,
                )
        del text_fit, text_valid, blocks_fit, blocks_valid
        return inner_fold, inner_records

    completed: List[Tuple[int, List[Dict[str, Any]]]] = []
    worker_count = max(1, min(int(n_jobs), len(inner_splits)))
    if worker_count == 1:
        for inner_fold, (fit_idx, valid_idx) in enumerate(inner_splits, start=1):
            completed.append(evaluate_inner(inner_fold, fit_idx, valid_idx))
    else:
        print(
            f"{log_prefix}: running {len(inner_splits)} inner folds with "
            f"{worker_count} concurrent workers",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(evaluate_inner, inner_fold, fit_idx, valid_idx)
                for inner_fold, (fit_idx, valid_idx) in enumerate(
                    inner_splits, start=1
                )
            ]
            for future in as_completed(futures):
                completed.append(future.result())

    # Sort before appending so inner_fold1/2/3 columns remain deterministic even
    # when the workers finish in a different order.
    for _, inner_records in sorted(completed, key=lambda item: item[0]):
        for inner_record in inner_records:
            record = records[inner_record["configuration"]]
            record["inner_mae"].append(inner_record["mae"])
            record["inner_iterations"].append(inner_record["iterations"])
            record["inner_seconds"].append(inner_record["seconds"])
            record["inner_stop_codes"].append(inner_record["stop_code"])

    rows: List[Dict[str, Any]] = []
    for record in records.values():
        row = {
            key: value
            for key, value in record.items()
            if not key.startswith("inner_")
        }
        for fold in range(3):
            row[f"inner_fold{fold + 1}_mae"] = record["inner_mae"][fold]
            row[f"inner_fold{fold + 1}_iterations"] = record[
                "inner_iterations"
            ][fold]
            row[f"inner_fold{fold + 1}_seconds"] = record["inner_seconds"][fold]
            row[f"inner_fold{fold + 1}_stop_code"] = record[
                "inner_stop_codes"
            ][fold]
        row["mean_mae"] = float(np.mean(record["inner_mae"]))
        row["mean_iterations"] = float(np.mean(record["inner_iterations"]))
        row["fit_seconds_total"] = float(np.sum(record["inner_seconds"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mean_mae", "alpha"])


def _extend_alpha_grid(grid: np.ndarray, direction: str) -> np.ndarray:
    low, high = float(np.log10(grid.min())), float(np.log10(grid.max()))
    if direction == "lower":
        new = np.logspace(low - 2, low - 0.25, 8)
    else:
        new = np.logspace(high + 0.25, high + 2, 8)
    return np.unique(np.concatenate([grid, new]))


def tune_alpha_grouped(
    text: sp.csr_matrix,
    blocks: Mapping[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    lambdas: Mapping[str, float],
    curve_path: Path,
    tolerance: float = 1e-4,
    max_iterations: int = 2_000,
    max_extensions: int = 4,
    log_prefix: str = "",
    n_jobs: int = 1,
) -> Tuple[float, pd.DataFrame, str, List[Dict[str, Any]]]:
    grid = default_alpha_grid()
    extensions: List[Dict[str, Any]] = []
    boundary = "interior"
    for extension_index in range(max_extensions + 1):
        # Descending order gives stable warm starts from strongly regularized fits.
        configurations = [
            (float(alpha), dict(lambdas)) for alpha in grid[::-1]
        ]
        curve = grouped_cv_configurations(
            text,
            blocks,
            y,
            groups,
            configurations,
            tolerance,
            max_iterations,
            log_prefix,
            n_jobs=n_jobs,
        )
        best = curve.iloc[0]
        selected = float(best["alpha"])
        if not np.isclose(selected, grid.min()) and not np.isclose(
            selected, grid.max()
        ):
            boundary = "interior"
            break
        direction = "lower" if np.isclose(selected, grid.min()) else "upper"
        boundary = f"{direction}_boundary"
        if extension_index == max_extensions:
            break
        old = [float(grid.min()), float(grid.max())]
        grid = _extend_alpha_grid(grid, direction)
        extensions.append(
            {
                "direction": direction,
                "old_range": old,
                "new_range": [float(grid.min()), float(grid.max())],
            }
        )
    curve["selected"] = np.isclose(curve["alpha"], selected)
    curve["boundary_status"] = boundary
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve.sort_values("alpha").to_csv(curve_path, index=False)
    return selected, curve, boundary, extensions


def tune_alpha_two_stage_grouped(
    screen_text: sp.csr_matrix,
    screen_blocks: Mapping[str, np.ndarray],
    screen_y: np.ndarray,
    screen_groups: np.ndarray,
    full_text: sp.csr_matrix,
    full_blocks: Mapping[str, np.ndarray],
    full_y: np.ndarray,
    full_groups: np.ndarray,
    lambdas: Mapping[str, float],
    curve_path: Path,
    tolerance: float = 1e-4,
    screen_max_iterations: int = 250,
    max_iterations: int = 2_000,
    refine_half_decades: float = 0.5,
    max_extensions: int = 4,
    log_prefix: str = "",
    n_jobs: int = 1,
) -> Tuple[float, pd.DataFrame, str, List[Dict[str, Any]]]:
    """Auditable reduced-cost alpha tuning.

    Every point of the mandated 57-point grid is screened on a deterministic,
    hotel-preserved coarse sample.  Full outer-training data are then used for
    the neighbourhood around the screening optimum.  A full-data optimum at a
    neighbourhood edge expands the refinement grid in two-decade increments.
    This is deliberately distinct from exhaustive full-data evaluation and the
    returned curve labels both stages explicitly.
    """

    base_grid = default_alpha_grid()
    screen_configurations = [
        (float(alpha), dict(lambdas)) for alpha in base_grid[::-1]
    ]
    screen_curve = grouped_cv_configurations(
        screen_text,
        screen_blocks,
        screen_y,
        screen_groups,
        screen_configurations,
        tolerance,
        screen_max_iterations,
        f"{log_prefix} alpha-screen",
        n_jobs=n_jobs,
    ).copy()
    screen_selected = float(screen_curve.iloc[0]["alpha"])
    screen_curve["stage"] = "coarse_screen"
    screen_curve["selection_rows"] = int(len(screen_y))
    screen_curve["screen_selected"] = np.isclose(
        screen_curve["alpha"], screen_selected
    )
    screen_curve["selected"] = False

    log_selected = math.log10(screen_selected)
    initial_refine = base_grid[
        np.abs(np.log10(base_grid) - log_selected)
        <= float(refine_half_decades) + 1e-12
    ]
    if len(initial_refine) < 3:
        nearest = np.argsort(np.abs(np.log10(base_grid) - log_selected))[:5]
        initial_refine = np.sort(base_grid[nearest])

    evaluated: Dict[float, pd.Series] = {}
    extensions: List[Dict[str, Any]] = []
    pending = np.asarray(initial_refine, dtype=np.float64)
    boundary = "interior"
    candidate_domain = np.asarray(base_grid, dtype=np.float64)

    for extension_index in range(max_extensions + 1):
        new_values = np.asarray(
            [value for value in pending if float(value) not in evaluated],
            dtype=np.float64,
        )
        if len(new_values):
            new_curve = grouped_cv_configurations(
                full_text,
                full_blocks,
                full_y,
                full_groups,
                [
                    (float(alpha), dict(lambdas))
                    for alpha in np.sort(new_values)[::-1]
                ],
                tolerance,
                max_iterations,
                f"{log_prefix} alpha-refine-{extension_index}",
                n_jobs=n_jobs,
            )
            for _, row in new_curve.iterrows():
                evaluated[float(row["alpha"])] = row.copy()

        refine_curve = pd.DataFrame(list(evaluated.values())).sort_values(
            ["mean_mae", "alpha"]
        )
        selected = float(refine_curve.iloc[0]["alpha"])
        evaluated_values = np.sort(
            np.asarray(list(evaluated), dtype=np.float64)
        )
        at_lower = np.isclose(selected, evaluated_values.min())
        at_upper = np.isclose(selected, evaluated_values.max())
        if not at_lower and not at_upper:
            boundary = "interior"
            break

        direction = "lower" if at_lower else "upper"
        boundary = f"{direction}_boundary"
        if extension_index == max_extensions:
            break

        if direction == "lower":
            available = candidate_domain[
                candidate_domain < evaluated_values.min()
            ]
            pending = np.sort(available)[-8:]
            if not len(pending):
                old_domain = candidate_domain.copy()
                candidate_domain = _extend_alpha_grid(
                    candidate_domain, "lower"
                )
                pending = candidate_domain[
                    candidate_domain < old_domain.min()
                ]
        else:
            available = candidate_domain[
                candidate_domain > evaluated_values.max()
            ]
            pending = np.sort(available)[:8]
            if not len(pending):
                old_domain = candidate_domain.copy()
                candidate_domain = _extend_alpha_grid(
                    candidate_domain, "upper"
                )
                pending = candidate_domain[
                    candidate_domain > old_domain.max()
                ]
        extensions.append(
            {
                "direction": direction,
                "trigger_alpha": selected,
                "added_alphas": [float(value) for value in pending],
                "stage": "full_refine",
            }
        )

    refine_curve = pd.DataFrame(list(evaluated.values())).copy()
    refine_curve["stage"] = "full_refine"
    refine_curve["selection_rows"] = int(len(full_y))
    refine_curve["screen_selected"] = False
    refine_curve["selected"] = np.isclose(refine_curve["alpha"], selected)
    combined = pd.concat([screen_curve, refine_curve], ignore_index=True)
    combined["boundary_status"] = boundary
    combined["approved_reduced_protocol"] = True
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    combined.sort_values(["stage", "alpha"]).to_csv(curve_path, index=False)
    return selected, combined, boundary, extensions


def lambda_configurations(
    block_names: Sequence[str],
    values: Sequence[float],
    alpha: float,
) -> List[Tuple[float, Dict[str, float]]]:
    """Full Cartesian block-multiplier grid in a locality-friendly order."""
    combinations = list(itertools.product(values, repeat=len(block_names)))
    # A log-space nearest-neighbour order improves LSQR warm starts.
    if combinations:
        remaining = set(combinations)
        current = min(remaining)
        ordered = [current]
        remaining.remove(current)
        while remaining:
            log_current = np.log10(np.asarray(current, dtype=float))
            current = min(
                remaining,
                key=lambda item: float(
                    np.square(
                        np.log10(np.asarray(item, dtype=float)) - log_current
                    ).sum()
                ),
            )
            ordered.append(current)
            remaining.remove(current)
        combinations = ordered
    return [
        (
            float(alpha),
            {name: float(value) for name, value in zip(block_names, combination)},
        )
        for combination in combinations
    ]
