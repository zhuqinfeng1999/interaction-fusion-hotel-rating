"""Efficient dense ridge paths used by Phase 2 representation experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def default_alpha_grid() -> np.ndarray:
    """1e-5..1e9 with four intervals per decade (57 points)."""
    return np.logspace(-5, 9, 57, dtype=np.float64)


@dataclass
class RidgeSolution:
    alpha: float
    coefficients: np.ndarray
    intercept: float
    curve: pd.DataFrame
    boundary_status: str
    extensions: List[Dict[str, Any]]
    tuning_seconds: float
    fit_seconds: float


@dataclass
class SpectralState:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    projected_xty: np.ndarray
    x_mean: np.ndarray
    y_mean: float
    valid_idx: np.ndarray


def scale_selected_columns(
    train: np.ndarray,
    test: np.ndarray,
    columns: Sequence[int],
    chunk_size: int = 20_000,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Copy float32 matrices and standardize selected columns using train stats."""
    x_train = np.asarray(train, dtype=np.float32).copy()
    x_test = np.asarray(test, dtype=np.float32).copy()
    selected = np.asarray(columns, dtype=np.int64)
    if not len(selected):
        return x_train, x_test, {
            "columns": selected,
            "mean": np.empty(0, dtype=np.float64),
            "std": np.empty(0, dtype=np.float64),
        }
    mean = x_train[:, selected].mean(axis=0, dtype=np.float64)
    std = x_train[:, selected].std(axis=0, dtype=np.float64)
    std[std == 0] = 1.0
    mean32 = mean.astype(np.float32)
    std32 = std.astype(np.float32)
    for matrix in (x_train, x_test):
        for start in range(0, len(matrix), chunk_size):
            stop = min(start + chunk_size, len(matrix))
            matrix[start:stop, selected] = (
                matrix[start:stop, selected] - mean32
            ) / std32
    return x_train, x_test, {
        "columns": selected,
        "mean": mean,
        "std": std,
    }


def _sufficient_statistics(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    chunk_size: int = 20_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n_features = x.shape[1]
    sum_x = np.zeros(n_features, dtype=np.float64)
    sum_y = 0.0
    xtx = np.zeros((n_features, n_features), dtype=np.float64)
    xty = np.zeros(n_features, dtype=np.float64)
    n = 0
    for start in range(0, len(indices), chunk_size):
        selected = indices[start : start + chunk_size]
        xb = np.asarray(x[selected], dtype=np.float32)
        yb = np.asarray(y[selected], dtype=np.float64)
        sum_x += xb.sum(axis=0, dtype=np.float64)
        sum_y += float(yb.sum(dtype=np.float64))
        # Float32 BLAS is substantially faster; accumulation and eigensolve are float64.
        xtx += np.asarray(xb.T @ xb, dtype=np.float64)
        xty += np.asarray(xb.T @ yb, dtype=np.float64)
        n += len(selected)
    if n == 0:
        raise ValueError("empty ridge fit indices")
    x_mean = sum_x / n
    y_mean = sum_y / n
    centered_xtx = xtx - n * np.outer(x_mean, x_mean)
    centered_xty = xty - n * x_mean * y_mean
    centered_xtx = (centered_xtx + centered_xtx.T) * 0.5
    return centered_xtx, centered_xty, x_mean, y_mean


def _spectral_state(
    x: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    valid_idx: np.ndarray,
) -> SpectralState:
    gram, xty, x_mean, y_mean = _sufficient_statistics(
        x,
        y,
        np.asarray(fit_idx, dtype=np.int64),
    )
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_xty = eigenvectors.T @ xty
    return SpectralState(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        projected_xty=projected_xty,
        x_mean=x_mean,
        y_mean=float(y_mean),
        valid_idx=np.asarray(valid_idx, dtype=np.int64),
    )


def _coefficients_for_alphas(
    state: SpectralState,
    alphas: np.ndarray,
) -> np.ndarray:
    shrinkage = state.projected_xty[:, None] / (
        state.eigenvalues[:, None] + alphas[None, :]
    )
    return state.eigenvectors @ shrinkage


def _score_alphas(
    x: np.ndarray,
    y: np.ndarray,
    states: Sequence[SpectralState],
    alphas: np.ndarray,
    prediction_chunk_size: int = 20_000,
) -> Tuple[np.ndarray, np.ndarray]:
    fold_scores = np.zeros((len(states), len(alphas)), dtype=np.float64)
    for fold_index, state in enumerate(states):
        coefficients = _coefficients_for_alphas(state, alphas)
        absolute_error = np.zeros(len(alphas), dtype=np.float64)
        n = 0
        for start in range(0, len(state.valid_idx), prediction_chunk_size):
            selected = state.valid_idx[start : start + prediction_chunk_size]
            xb = np.asarray(x[selected], dtype=np.float32)
            prediction = (
                (xb - state.x_mean.astype(np.float32)) @ coefficients
                + state.y_mean
            )
            absolute_error += np.abs(
                y[selected, None] - prediction
            ).sum(axis=0)
            n += len(selected)
        fold_scores[fold_index] = absolute_error / n
    return fold_scores, fold_scores.mean(axis=0)


def _extend_grid(alphas: np.ndarray, direction: str) -> np.ndarray:
    log_min = float(np.log10(alphas.min()))
    log_max = float(np.log10(alphas.max()))
    if direction == "lower":
        added = np.logspace(log_min - 2, log_min - 0.25, 8)
    elif direction == "upper":
        added = np.logspace(log_max + 0.25, log_max + 2, 8)
    else:
        raise ValueError(direction)
    return np.unique(np.concatenate([alphas, added])).astype(np.float64)


def tune_dense_ridge(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    curve_path: Path,
    alphas: np.ndarray | None = None,
    max_extensions: int = 6,
) -> Tuple[float, pd.DataFrame, str, List[Dict[str, Any]], float]:
    """Grouped three-fold MAE path with automatic two-decade boundary extension."""
    started = time.perf_counter()
    grid = np.asarray(
        default_alpha_grid() if alphas is None else alphas,
        dtype=np.float64,
    )
    splitter = GroupKFold(n_splits=3)
    states = [
        _spectral_state(x, y, fit_idx, valid_idx)
        for fit_idx, valid_idx in splitter.split(x, y, groups=groups)
    ]
    extensions: List[Dict[str, Any]] = []
    boundary_status = "interior"
    for extension_index in range(max_extensions + 1):
        fold_scores, mean_scores = _score_alphas(x, y, states, grid)
        # Prefer the stronger penalty only when scores are tied at floating-point
        # resolution; this prevents meaningless boundary extension on flat paths.
        numerical_tolerance = max(1e-10, abs(float(mean_scores.min())) * 1e-10)
        tied = np.flatnonzero(
            mean_scores <= float(mean_scores.min()) + numerical_tolerance
        )
        best_index = int(tied[-1])
        if best_index not in {0, len(grid) - 1}:
            boundary_status = "interior"
            break
        direction = "lower" if best_index == 0 else "upper"
        boundary_status = f"{direction}_boundary"
        if extension_index == max_extensions:
            break
        old_min, old_max = float(grid.min()), float(grid.max())
        grid = _extend_grid(grid, direction)
        extensions.append(
            {
                "direction": direction,
                "old_min": old_min,
                "old_max": old_max,
                "new_min": float(grid.min()),
                "new_max": float(grid.max()),
            }
        )
    curve = pd.DataFrame(
        {
            "alpha": grid,
            **{
                f"inner_fold{fold + 1}_mae": fold_scores[fold]
                for fold in range(fold_scores.shape[0])
            },
            "mean_mae": mean_scores,
        }
    ).sort_values("alpha")
    best_alpha = float(grid[best_index])
    curve["selected"] = np.isclose(curve["alpha"], best_alpha)
    curve["boundary_status"] = boundary_status
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(curve_path, index=False)
    return (
        best_alpha,
        curve,
        boundary_status,
        extensions,
        time.perf_counter() - started,
    )


def fit_dense_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, float, float]:
    started = time.perf_counter()
    indices = np.arange(len(x), dtype=np.int64)
    gram, xty, x_mean, y_mean = _sufficient_statistics(x, y, indices)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    coefficients = eigenvectors @ (
        (eigenvectors.T @ xty) / (eigenvalues + alpha)
    )
    intercept = float(y_mean - x_mean @ coefficients)
    return coefficients, intercept, time.perf_counter() - started


def predict_dense(
    x: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    chunk_size: int = 20_000,
) -> Tuple[np.ndarray, float]:
    started = time.perf_counter()
    prediction = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), chunk_size):
        stop = min(start + chunk_size, len(x))
        prediction[start:stop] = x[start:stop] @ coefficients + intercept
    return prediction, time.perf_counter() - started


def fit_tuned_dense_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    curve_path: Path,
) -> RidgeSolution:
    alpha, curve, boundary, extensions, tuning_seconds = tune_dense_ridge(
        x_train,
        y_train,
        groups_train,
        curve_path,
    )
    coefficients, intercept, fit_seconds = fit_dense_ridge(
        x_train,
        y_train,
        alpha,
    )
    return RidgeSolution(
        alpha=alpha,
        coefficients=coefficients,
        intercept=intercept,
        curve=curve,
        boundary_status=boundary,
        extensions=extensions,
        tuning_seconds=tuning_seconds,
        fit_seconds=fit_seconds,
    )
