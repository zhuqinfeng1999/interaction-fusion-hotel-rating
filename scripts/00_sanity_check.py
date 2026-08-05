#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize_df(name: str, df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"## {name}")
    lines.append(f"rows: {len(df)}")
    lines.append(f"columns ({len(df.columns)}): {', '.join(map(str, df.columns))}")
    lines.append("head(3):")
    lines.append(df.head(3).to_string(index=False))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    reviews_path = root / "data_ori" / "reviews.csv"
    hotels_path = root / "data_ori" / "offerings.csv"
    output_path = root / "outputs" / "sanity_check.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    reviews = pd.read_csv(reviews_path, low_memory=False)
    hotels = pd.read_csv(hotels_path, low_memory=False)

    sections = [
        summarize_df("reviews.csv", reviews),
        summarize_df("offerings.csv", hotels),
    ]
    report = "\n".join(sections).strip() + "\n"

    print(report)
    output_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
