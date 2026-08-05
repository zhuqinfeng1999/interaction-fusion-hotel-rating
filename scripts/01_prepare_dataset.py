#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils_json import flatten_dict_one_level, parse_json_like  # noqa: E402


RATING_CANONICAL = {
    "overall": "rating_overall",
    "service": "rating_service",
    "cleanliness": "rating_cleanliness",
    "location": "rating_location",
    "rooms": "rating_rooms",
    "value": "rating_value",
    "sleepquality": "rating_sleep_quality",
}


def log(message: str) -> None:
    print(message)


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def extract_ratings(raw: object) -> Dict[str, object]:
    data = parse_json_like(raw)
    output = {col: None for col in RATING_CANONICAL.values()}
    for key, value in data.items():
        norm = normalize_key(key)
        if norm in RATING_CANONICAL:
            output[RATING_CANONICAL[norm]] = value
    return output


def build_review_text(titles: Iterable[str], texts: Iterable[str]) -> List[str]:
    combined = []
    for title, text in zip(titles, texts):
        title = str(title).strip() if title is not None else ""
        text = str(text).strip() if text is not None else ""
        if title and text:
            combined.append(f"{title}. {text}")
        else:
            combined.append(title or text)
    return combined


def expand_series(series: pd.Series, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        [flatten_dict_one_level(parse_json_like(item), prefix) for item in series]
    )


def render_table(headers: List[str], rows: List[List[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def build_stats_markdown(df: pd.DataFrame) -> str:
    lines = ["# Dataset Stats", ""]

    num_samples = len(df)
    num_hotels = df["offering_id"].nunique() if "offering_id" in df.columns else 0
    lines.append(f"- samples: {num_samples}")
    lines.append(f"- hotels: {num_hotels}")

    date_col = "date" if "date" in df.columns else "date_stayed"
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        if dates.notna().any():
            lines.append(f"- date_span: {dates.min().date()} to {dates.max().date()} ({date_col})")
        else:
            lines.append(f"- date_span: N/A ({date_col})")
    lines.append("")

    if "rating_overall" in df.columns:
        overall_counts = (
            df["rating_overall"]
            .value_counts(dropna=False)
            .sort_index()
        )
        rows = [[idx, int(count)] for idx, count in overall_counts.items()]
        lines.append("## Overall Rating Distribution")
        lines.append(render_table(["rating_overall", "count"], rows))
        lines.append("")

    aspect_cols = [
        "rating_service",
        "rating_cleanliness",
        "rating_location",
        "rating_rooms",
        "rating_value",
        "rating_sleep_quality",
    ]
    aspect_rows = []
    for col in aspect_cols:
        if col in df.columns:
            aspect_rows.append(
                [
                    col,
                    f"{df[col].mean():.3f}" if df[col].notna().any() else "N/A",
                    f"{df[col].std():.3f}" if df[col].notna().any() else "N/A",
                ]
            )
    if aspect_rows:
        lines.append("## Aspect Rating Stats")
        lines.append(render_table(["aspect", "mean", "std"], aspect_rows))
        lines.append("")

    missing = df.isna().mean().sort_values(ascending=False)
    missing_rows = [
        [col, f"{rate:.2%}"] for col, rate in missing.head(15).items()
    ]
    lines.append("## Missing Rate Summary (Top 15)")
    lines.append(render_table(["column", "missing_rate"], missing_rows))
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    reviews_path = ROOT / "data_ori" / "reviews.csv"
    offerings_path = ROOT / "data_ori" / "offerings.csv"
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    log("Loading reviews.csv ...")
    reviews = pd.read_csv(reviews_path, low_memory=False)
    log(f"reviews rows: {len(reviews)}")

    ratings_expanded = pd.DataFrame([extract_ratings(item) for item in reviews["ratings"]])
    author_expanded = expand_series(reviews["author"], "author")

    titles = reviews["title"].fillna("").astype(str).tolist()
    texts = reviews["text"].fillna("").astype(str).tolist()
    reviews["review_text"] = build_review_text(titles, texts)

    keep_cols = [
        "offering_id",
        "id",
        "title",
        "text",
        "review_text",
        "date",
        "date_stayed",
        "via_mobile",
        "num_helpful_votes",
    ]
    keep_cols = [col for col in keep_cols if col in reviews.columns]
    reviews = pd.concat([reviews[keep_cols].copy(), ratings_expanded, author_expanded], axis=1)
    log(f"reviews after expansion: {len(reviews)}")

    log("Loading offerings.csv ...")
    offerings = pd.read_csv(offerings_path, low_memory=False)
    log(f"offerings rows: {len(offerings)}")

    address_expanded = expand_series(offerings["address"], "hoteladdr")
    hotel_cols = ["id", "name", "hotel_class", "region_id", "type", "url"]
    hotel_cols = [col for col in hotel_cols if col in offerings.columns]
    offerings = pd.concat([offerings[hotel_cols].copy(), address_expanded], axis=1)

    rename_map = {}
    for col in offerings.columns:
        if col == "id":
            rename_map[col] = "hotel_id"
        elif col.startswith("hoteladdr_"):
            rename_map[col] = col
        else:
            rename_map[col] = f"hotel_{col}"
    offerings = offerings.rename(columns=rename_map)

    log("Joining reviews with offerings ...")
    merged = reviews.merge(
        offerings,
        how="left",
        left_on="offering_id",
        right_on="hotel_id",
    )
    log(f"after join rows: {len(merged)}")

    rating_cols = list(RATING_CANONICAL.values())
    for col in rating_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    before = len(merged)
    merged = merged[merged["review_text"].fillna("").str.strip().ne("")]
    log(f"drop empty review_text: {before} -> {len(merged)}")

    before = len(merged)
    merged = merged[merged["rating_overall"].notna()]
    log(f"drop missing rating_overall: {before} -> {len(merged)}")

    before = len(merged)
    merged = merged[merged["rating_overall"].between(1, 5)]
    log(f"keep rating_overall in [1,5]: {before} -> {len(merged)}")

    dedup_cols = ["offering_id", "author_username", "date", "title", "text"]
    for col in dedup_cols:
        if col not in merged.columns:
            merged[col] = pd.NA
    before = len(merged)
    merged = merged.drop_duplicates(subset=dedup_cols, keep="first")
    log(f"drop duplicates: {before} -> {len(merged)}")

    parquet_path = output_dir / "dataset_clean.parquet"
    stats_path = output_dir / "dataset_stats.md"
    saved_parquet = False
    try:
        merged.to_parquet(parquet_path, index=False)
        saved_parquet = True
        log(f"saved: {parquet_path}")
    except ImportError:
        csv_path = output_dir / "dataset_clean.csv"
        merged.to_csv(csv_path, index=False)
        log("parquet engine missing; saved CSV fallback instead.")
        log(f"saved: {csv_path}")
        log("install pyarrow or fastparquet to enable parquet output.")

    stats_path.write_text(build_stats_markdown(merged), encoding="utf-8")
    log(f"saved: {stats_path}")


if __name__ == "__main__":
    main()
