from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ReviewDataset(Dataset):
    def __init__(
        self,
        texts: np.ndarray,
        labels: np.ndarray,
        row_ids: np.ndarray,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.row_ids = row_ids

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "text": str(self.texts[index]),
            "label": float(self.labels[index]),
            "row_id": int(self.row_ids[index]),
        }


class DynamicPaddingCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        encoded = self.tokenizer(
            [row["text"] for row in rows],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(
            [row["label"] for row in rows],
            dtype=torch.float32,
        )
        encoded["row_id"] = torch.tensor(
            [row["row_id"] for row in rows],
            dtype=torch.int64,
        )
        return encoded


def build_loader(
    texts: np.ndarray,
    labels: np.ndarray,
    row_ids: np.ndarray,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = ReviewDataset(texts, labels, row_ids)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=DynamicPaddingCollator(tokenizer, max_length),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    y_clipped = np.clip(y_pred, 1.0, 5.0)

    def calculate(predictions: np.ndarray) -> Dict[str, float]:
        return {
            "mae": float(mean_absolute_error(y_true, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        }

    return {
        "raw": calculate(y_pred),
        "clipped": calculate(y_clipped),
    }


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    row_ids: List[np.ndarray] = []
    for batch in tqdm(
        loader,
        desc=description,
        leave=False,
        mininterval=1.0,
    ):
        batch_row_ids = batch.pop("row_id")
        batch_labels = batch["labels"]
        model_inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs = model(**model_inputs)
        predictions.append(
            outputs.logits.squeeze(-1).float().cpu().numpy()
        )
        labels.append(batch_labels.numpy())
        row_ids.append(batch_row_ids.numpy())
    return (
        np.concatenate(predictions),
        np.concatenate(labels),
        np.concatenate(row_ids),
    )


def train_fold(
    model_name: str,
    checkpoint_dir: Path,
    train_texts: np.ndarray,
    train_labels: np.ndarray,
    train_row_ids: np.ndarray,
    valid_texts: np.ndarray,
    valid_labels: np.ndarray,
    valid_row_ids: np.ndarray,
    device: torch.device,
    max_length: int,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    epochs: int,
    seed: int,
    num_workers: int,
    gradient_checkpointing: bool,
) -> Dict[str, Any]:
    seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="regression",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)

    train_loader = build_loader(
        train_texts,
        train_labels,
        train_row_ids,
        tokenizer,
        max_length,
        train_batch_size,
        True,
        seed,
        num_workers,
    )
    valid_loader = build_loader(
        valid_texts,
        valid_labels,
        valid_row_ids,
        tokenizer,
        max_length,
        eval_batch_size,
        False,
        seed,
        num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * epochs
    warmup_steps = int(math.ceil(total_updates * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_valid_mae = float("inf")
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    fit_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        update_count = 0
        progress = tqdm(
            train_loader,
            desc=f"train epoch {epoch}/{epochs}",
            leave=False,
            mininterval=1.0,
        )
        for step, batch in enumerate(progress, start=1):
            batch.pop("row_id")
            model_inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs = model(**model_inputs)
                loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            total_loss += float(loss.detach().cpu()) * gradient_accumulation_steps
            if (
                step % gradient_accumulation_steps == 0
                or step == len(train_loader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_count += 1
            progress.set_postfix(
                loss=f"{total_loss / step:.4f}",
                updates=update_count,
            )

        valid_pred, valid_true, _ = predict(
            model,
            valid_loader,
            device,
            f"valid epoch {epoch}",
        )
        valid_metrics = regression_metrics(valid_true, valid_pred)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(1, len(train_loader)),
                "valid": valid_metrics,
                "updates": update_count,
            }
        )
        valid_mae = valid_metrics["raw"]["mae"]
        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            best_epoch = epoch
            model.save_pretrained(checkpoint_dir, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_dir)

    fit_seconds = time.perf_counter() - fit_start
    gpu_peak_memory_mb = float(
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    )
    del model, optimizer, scheduler, train_loader, valid_loader
    torch.cuda.empty_cache()
    return {
        "best_epoch": best_epoch,
        "best_valid_mae_raw": best_valid_mae,
        "history": history,
        "fit_seconds": fit_seconds,
        "gpu_peak_memory_mb": gpu_peak_memory_mb,
        "total_updates": total_updates,
        "warmup_steps": warmup_steps,
    }


def load_checkpoint(
    checkpoint_dir: Path,
    device: torch.device,
) -> Tuple[Any, torch.nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.to(device)
    return tokenizer, model


def compute_token_length_stats(
    texts: np.ndarray,
    model_name: str,
    max_length: int,
    output_json: Path,
    output_md: Path,
    batch_size: int = 512,
) -> Dict[str, Any]:
    if output_json.exists():
        return json.loads(output_json.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    original_max_length = tokenizer.model_max_length
    tokenizer.model_max_length = 10**9
    lengths = np.empty(len(texts), dtype=np.int32)
    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="token length audit",
        mininterval=1.0,
    ):
        end = min(len(texts), start + batch_size)
        encoded = tokenizer(
            texts[start:end].tolist(),
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
        )
        lengths[start:end] = np.asarray(encoded["length"], dtype=np.int32)
    tokenizer.model_max_length = original_max_length
    stats = {
        "model_name": model_name,
        "n_reviews": int(len(lengths)),
        "max_length": int(max_length),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p90": float(np.quantile(lengths, 0.90)),
        "p99": float(np.quantile(lengths, 0.99)),
        "max": int(lengths.max()),
        "truncated_n": int((lengths > max_length).sum()),
        "truncated_fraction": float((lengths > max_length).mean()),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    lines = [
        "# Token Length Audit",
        "",
        f"- model: `{model_name}`",
        f"- reviews: {stats['n_reviews']}",
        f"- max_length: {max_length}",
        f"- mean: {stats['mean']:.2f}",
        f"- median: {stats['median']:.2f}",
        f"- p90: {stats['p90']:.2f}",
        f"- p99: {stats['p99']:.2f}",
        f"- max: {stats['max']}",
        f"- truncated: {stats['truncated_n']} ({stats['truncated_fraction']:.4%})",
        "",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")
    return stats
