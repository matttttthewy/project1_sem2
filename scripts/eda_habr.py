#!/usr/bin/env python3
"""
Подготовка CSV Хабра к анализу:
* парсинг числовых полей (views, comments, rating)
* попытка распарсить дату в любой из встречающихся форм
* сохранение Parquet + Feather
"""

import argparse
import pathlib
import re
import pandas as pd

INT_RE = re.compile(r"\d+")


def to_int(value) -> int:
    """'12K' → 12000; '1,2K' → 1200; '-', '' → 0."""
    if pd.isna(value):
        return 0
    text = str(value).replace(" ", "").replace(",", ".").strip().lower()
    if not text or text in {"nan", "-"}:
        return 0
    if text.endswith("k"):
        try:
            return int(float(text[:-1]) * 1000)
        except ValueError:
            return 0
    m = INT_RE.search(text)
    return int(m.group()) if m else 0


def main() -> None:
    p = argparse.ArgumentParser(description="Очистка и конвертация статей Хабра")
    p.add_argument("input_csv", help="data/raw/habr_*.csv")
    p.add_argument("output_parquet", help="data/processed/habr.parquet")
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)

    # === Дата ===
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True, utc=True)
    df["date"] = df["date"].dt.tz_convert(None)

    # === Числовые ===
    for col in ["views", "comments", "rating"]:
        df[col] = df[col].map(to_int).astype("int32")

    out = pathlib.Path(args.output_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    df.to_feather(out.with_suffix(".feather"))
    print(f"[DONE] {len(df)} строк → {out}  | даты ≠ NaT: {(df['date'].notna()).sum()}")

if __name__ == "__main__":
    main()
