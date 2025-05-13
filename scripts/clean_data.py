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


def extract_main_hub(hubs_text):
    """Извлекает первый/основной хаб из списка хабов, разделенных точкой с запятой"""
    if pd.isna(hubs_text) or not hubs_text:
        return None
    return hubs_text.split(';')[0] if ';' in hubs_text else hubs_text


def process_dataframe(df):
    """Обрабатывает DataFrame с данными статей Хабра"""
    # Копируем для избежания предупреждений о модификации
    df = df.copy()

    # Нормализуем даты и числовые признаки
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    for col in ["views", "comments", "rating"]:
        df[col] = df[col].map(to_int).astype("int32")

    # Добавляем новые признаки для анализа
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday
    df["main_hub"] = df["hubs"].apply(extract_main_hub)

    # Вычисляем время жизни статьи (в днях)
    df["days_live"] = (pd.Timestamp.now() - df["date"]).dt.days

    # Рассчитываем среднее количество просмотров, комментариев в день
    df["views_per_day"] = df["views"] / df["days_live"].clip(lower=1)
    df["comments_per_day"] = df["comments"] / df["days_live"].clip(lower=1)

    # Вычисляем отношение комментариев к просмотрам (вовлеченность)
    df["engagement"] = (df["comments"] / df["views"].clip(lower=1)) * 100

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Очистка и конвертация статей Хабра")
    parser.add_argument("input_csv", help="data/raw/habr_*.csv")
    parser.add_argument("output_parquet", help="data/processed/habr.parquet")
    args = parser.parse_args()

    # Создаем директорию для выходных файлов
    out = pathlib.Path(args.output_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Читаем исходник
    df = pd.read_csv(args.input_csv)
    print(f"Загружено {len(df)} строк из {args.input_csv}")

    # Обрабатываем данные
    df_processed = process_dataframe(df)

    # Сохраняем результаты
    df_processed.to_parquet(out, index=False)
    df_processed.to_feather(out.with_suffix(".feather"))
    print(f"[DONE] {len(df_processed)} строк → {out}")
    print(f"[DONE] {len(df_processed)} строк → {out.with_suffix('.feather')}")


if __name__ == "__main__":
    main()
