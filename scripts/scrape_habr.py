#!/usr/bin/env python3
"""
Скрипт для сбора данных с Хабра:
* собирает статьи с хабра за указанный период
* сохраняет в CSV файл
"""

import os
import time
import random
import argparse
import pathlib
from datetime import datetime, timedelta
import pandas as pd
import requests
from bs4 import BeautifulSoup


def create_dir_if_not_exists(path):
    """Создает директорию, если она не существует"""
    if not os.path.exists(path):
        os.makedirs(path)


def get_page(url, retries=3, sleep_time=1):
    """Загружает страницу с указанного URL с повторными попытками"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                sleep_time = sleep_time * 2  # Экспоненциальная задержка
                time.sleep(sleep_time + random.uniform(0, 1))

    return None


def parse_article(article_html):
    """Извлекает данные из HTML-элемента статьи"""
    try:
        # Заголовок статьи
        title_element = article_html.find('a', class_='tm-title__link')
        title = title_element.text.strip() if title_element else None

        # URL статьи
        url = 'https://habr.com' + title_element['href'] if title_element else None

        # Автор
        author_element = article_html.find('a', class_='tm-user-info__username')
        author = author_element.text.strip() if author_element else None

        # Время публикации
        time_element = article_html.find('time')
        date = time_element['datetime'] if time_element else None

        # Хабы/теги
        hubs_elements = article_html.find_all('a', class_='tm-article-snippet__hubs-item-link')
        hubs = [hub.text.strip() for hub in hubs_elements] if hubs_elements else []

        # Статистика
        stats = article_html.find('div', class_='tm-data-icons')

        # Просмотры
        views_element = stats.find('span', class_='tm-icon-counter__value',
                                   title='Количество просмотров') if stats else None
        views = views_element.text.strip() if views_element else None

        # Комментарии
        comments_element = stats.find('span', class_='tm-icon-counter__value',
                                      title='Комментарии') if stats else None
        comments = comments_element.text.strip() if comments_element else None

        # Рейтинг
        rating_element = stats.find('span', class_='tm-votes-meter__value') if stats else None
        rating = rating_element.text.strip() if rating_element else None

        # Текст аннотации
        text_element = article_html.find('div', class_='article-formatted-body')
        text = text_element.text.strip() if text_element else None

        return {
            'title': title,
            'url': url,
            'author': author,
            'date': date,
            'hubs': ';'.join(hubs),
            'views': views,
            'comments': comments,
            'rating': rating,
            'text': text
        }
    except Exception as e:
        print(f"Error parsing article: {e}")
        return {}


def scrape_habr(pages=20, output_path='data/raw/habr_articles.csv'):
    """Собирает статьи с главной страницы Хабра"""
    base_url = 'https://habr.com/ru/articles/page'
    all_articles = []

    for page in range(1, pages + 1):
        print(f"Scraping page {page}...")
        url = f"{base_url}{page}/"
        html = get_page(url)

        if not html:
            print(f"Failed to fetch page {page}")
            continue

        soup = BeautifulSoup(html, 'html.parser')
        article_elements = soup.find_all('article', class_='tm-articles-list__item')

        for article_html in article_elements:
            article_data = parse_article(article_html)
            if article_data:
                all_articles.append(article_data)

        # Случайная задержка для избежания блокировки
        time.sleep(random.uniform(2, 4))

    # Создаем DataFrame и сохраняем в CSV
    df = pd.DataFrame(all_articles)

    # Создаем директорию, если не существует
    output_dir = os.path.dirname(output_path)
    create_dir_if_not_exists(output_dir)

    # Сохраняем данные
    df.to_csv(output_path, index=False)
    print(f"[DONE] Collected {len(df)} articles → {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Скрапинг статей с Хабра")
    parser.add_argument("--pages", type=int, default=20, help="Количество страниц для скрапинга")
    parser.add_argument("--output", default="data/raw/habr_articles.csv", help="Путь для сохранения CSV")
    args = parser.parse_args()

    create_dir_if_not_exists("data/raw")
    scrape_habr(pages=args.pages, output_path=args.output)


if __name__ == "__main__":
    main()