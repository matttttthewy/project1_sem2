#!/usr/bin/env python3
"""
Подготовка CSV Хабра к анализу:
* парсинг числовых полей (views, comments, rating)
* нормализация даты
* сохранение Parquet + Feather
"""

import argparse
import pathlib
import re
import pandas as pd

# регулярка для чисел
INT_RE = re.compile(r"\d+")

def to_int(value) -> int:
    """Преобразует строку вида '12K', '1,2K', '' или уже число → int"""
    if pd.isna(value):
        return 0
    # приводим к строке и чистим визуальные разделители
    text = str(value).replace(" ", "").replace(",", ".").strip().lower()
    if not text:
        return 0
    # 1.5K → 1500
    if text.endswith("k"):
        try:
            return int(float(text[:-1]) * 1000)
        except ValueError:
            return 0
    # просто число
    m = INT_RE.search(text)
    return int(m.group()) if m else 0

def main() -> None:
    parser = argparse.ArgumentParser(description="Очистка и конвертация статей Хабра")
    parser.add_argument("input_csv", help="data/raw/habr_*.csv")
    parser.add_argument("output_parquet", help="data/processed/habr.parquet")
    args = parser.parse_args()

    # читаем исходник
    df = pd.read_csv(args.input_csv)

    # нормализуем даты и числовые признаки
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    for col in ["views", "comments", "rating"]:
        df[col] = df[col].map(to_int).astype("int32")

    # сохраняем
    out = pathlib.Path(args.output_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    df.to_feather(out.with_suffix(".feather"))
    print(f"[DONE] {len(df)} строк → {out}")

if __name__ == "__main__":
    main()