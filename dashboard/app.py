# dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone
import os
import sys
import re
from io import StringIO

# Корректировка для правильного импорта при запуске через streamlit run
if __name__ == "__main__":
    # Определение абсолютных путей для корректной работы с файлами
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

    # Определение путей к данным
    raw_data_path = os.path.join(parent_dir, "data", "raw", "habr_articles.csv")
    processed_data_path = os.path.join(parent_dir, "data", "processed", "habr.parquet")

# Конфигурация страницы
st.set_page_config(
    page_title="Анализ контента Хабра",
    page_icon="📊",
    layout="wide"
)

# Настройка стилей
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 1rem;
        color: #4e5d78;
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 5px solid #4e9dd6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .footnote {
        font-size: 0.8rem;
        color: #666;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# Функция для преобразования строк вида "1.5K" в числа
def to_int(value):
    if pd.isna(value):
        return 0
    text = str(value).replace(" ", "").replace(",", ".").strip().lower()
    if not text:
        return 0
    if text.endswith("k"):
        try:
            return int(float(text[:-1]) * 1000)
        except ValueError:
            return 0
    match = re.search(r"\d+", text)
    return int(match.group()) if match else 0


# Модифицированные функции загрузки данных с улучшенной обработкой ошибок
@st.cache_data
def load_data(uploaded_file=None):
    """Загрузка данных с улучшенной обработкой путей и исключений"""
    if uploaded_file is not None:
        try:
            # Определяем формат файла по его расширению
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            elif uploaded_file.name.endswith('.feather'):
                df = pd.read_feather(uploaded_file)
            else:
                st.error(f"Неподдерживаемый формат файла: {uploaded_file.name}")
                return None

            # Преобразование даты
            if 'date' in df.columns:
                df["date"] = pd.to_datetime(df["date"])

            # Преобразование числовых колонок
            for col in ["views", "comments", "rating"]:
                if col in df.columns and df[col].dtype == 'object':
                    df[col] = df[col].apply(to_int)

            return df
        except Exception as e:
            st.error(f"Ошибка при чтении загруженного файла: {e}")
            return None

    try:
        # Попытка использования абсолютных путей при запуске через streamlit run
        if 'processed_data_path' in globals():
            if os.path.exists(processed_data_path):
                return pd.read_parquet(processed_data_path)
            elif os.path.exists(raw_data_path):
                df = pd.read_csv(raw_data_path)
                if 'date' in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                return df

        # Альтернативный поиск файлов в стандартных локациях
        paths_to_check = [
            "../data/processed/habr.parquet",
            "../data/raw/habr_articles.csv",
            "data/processed/habr.parquet",
            "data/raw/habr_articles.csv",
            "habr.parquet",
            "habr_articles.csv",
            "habr_processed.csv"
        ]

        for path in paths_to_check:
            if os.path.exists(path):
                if path.endswith('.parquet'):
                    return pd.read_parquet(path)
                elif path.endswith('.csv'):
                    df = pd.read_csv(path)
                    if 'date' in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                    return df

        st.warning("Не удалось найти файлы данных в стандартных местах. Пожалуйста, загрузите файл.")
        return None
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
        return None


@st.cache_data
def preprocess_data(df):
    """Предобработка данных с защитой от None"""
    if df is None:
        return None

    df = df.copy()
    # Обработка пропусков в ключевых полях
    df = df.dropna(subset=['title', 'author', 'date'])

    # Добавление временных полей
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime('%B')
    df["weekday"] = df["date"].dt.weekday
    df["hour"] = df["date"].dt.hour
    df["day"] = df["date"].dt.day

    # Безопасная обработка timezone
    try:
        # Нормализация часового пояса
        if df["date"].dt.tz is not None:
            # Для данных с timezone
            current_date = pd.Timestamp.now(tz=timezone.utc)
            df["days_live"] = (current_date - df["date"]).dt.total_seconds() / (24 * 60 * 60)
        else:
            # Для данных без timezone
            current_date = pd.Timestamp.now()
            df["days_live"] = (current_date - df["date"]).dt.total_seconds() / (24 * 60 * 60)
    except Exception as e:
        st.warning(f"Предупреждение при обработке дат: {e}")
        # Альтернативный расчет
        df["days_live"] = 1

    df["days_live"] = df["days_live"].clip(lower=1).astype(int)

    # Безопасное вычисление длины текста
    if 'text' in df.columns:
        df['text_length'] = df['text'].astype(str).str.len()
        df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))

    # Дополнительные метрики
    if 'views' in df.columns and 'days_live' in df.columns:
        df['views_per_day'] = df['views'] / df['days_live'].clip(lower=1)

    if 'comments' in df.columns and 'views' in df.columns:
        df['engagement'] = (df['comments'] / df['views'].clip(lower=1)) * 100

    # Словарь для преобразования номера дня недели в название
    weekday_names = {
        0: 'Понедельник',
        1: 'Вторник',
        2: 'Среда',
        3: 'Четверг',
        4: 'Пятница',
        5: 'Суббота',
        6: 'Воскресенье'
    }
    df['weekday_name'] = df['weekday'].map(weekday_names)

    return df


def create_sample_data():
    """Создание тестового датасета для демонстрации"""
    import numpy as np

    # Генерация дат с 2023 по 2025 год
    date_range = pd.date_range(start='2023-01-01', end='2025-05-01', freq='D')
    sample_size = 400

    # Случайный выбор дат из диапазона
    random_dates = np.random.choice(date_range, size=sample_size)

    # Список популярных IT тем
    topics = ['Python', 'JavaScript', 'Machine Learning', 'DevOps', 'Cloud',
              'Microservices', 'API', 'Testing', 'UI/UX', 'Git', 'Docker',
              'Kubernetes', 'SQL', 'NoSQL', 'React', 'Vue.js', 'Angular',
              'TensorFlow', 'PyTorch', 'Big Data', 'Data Science', 'Web Development']

    # Генерация заголовков
    titles = [f"Как использовать {np.random.choice(topics)} в проекте" for _ in range(sample_size // 4)]
    titles += [f"Почему {np.random.choice(topics)} лучше для {np.random.choice(topics)}" for _ in
               range(sample_size // 4)]
    titles += [f"10 советов по {np.random.choice(topics)}" for _ in range(sample_size // 4)]
    titles += [f"Обзор {np.random.choice(topics)} в 2024 году" for _ in range(sample_size - len(titles))]

    # Перемешиваем заголовки
    np.random.shuffle(titles)

    # Генерация имен авторов
    authors = [f"Author_{i}" for i in range(1, 51)]

    # Создание датафрейма
    df = pd.DataFrame({
        'title': titles[:sample_size],
        'author': np.random.choice(authors, size=sample_size),
        'date': random_dates,
        'rating': np.random.exponential(scale=20, size=sample_size).astype(int),
        'views': np.random.exponential(scale=5000, size=sample_size).astype(int),
        'comments': np.random.exponential(scale=50, size=sample_size).astype(int),
        'text': ['Sample text content ' * np.random.randint(10, 100) for _ in range(sample_size)]
    })

    return df


# Основное приложение с обработкой исключений
try:
    # Заголовок дашборда
    st.markdown('<div class="main-header">Анализ контента платформы Хабр</div>', unsafe_allow_html=True)

    # Добавление возможности загрузки пользовательских данных
    with st.sidebar:
        st.header("Загрузка данных")
        uploaded_file = st.file_uploader("Загрузите CSV или Parquet файл с данными Хабра",
                                         type=["csv", "parquet", "feather"])

        st.markdown("### О проекте")
        st.markdown("""
        Этот дашборд представляет результаты анализа публикаций на платформе Хабр. 
        Вы можете загрузить свои данные или использовать демонстрационные данные.

        Анализ включает:
        - Распределение рейтингов статей
        - Временные паттерны публикаций
        - Анализ авторского контента
        - Взаимосвязь между метриками
        """)

        use_demo = st.checkbox("Использовать демо-данные", value=False)

    # Загрузка данных
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.success(f"Данные успешно загружены! Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
    elif use_demo:
        st.info("Используются демонстрационные данные")
        df = create_sample_data()
    else:
        df = load_data()

    if df is None:
        st.warning(
            "Не удалось загрузить данные. Переключитесь в режим использования демо-данных или загрузите свой файл.")
        st.stop()

    # Предобработка данных
    df = preprocess_data(df)

    # Основная структура дашборда - вкладки
    tabs = st.tabs(["Обзор", "Данные", "Рейтинги", "Временной анализ", "Авторы", "Корреляции"])

    # Вкладка 1: Обзор
    with tabs[0]:
        # Ключевые метрики
        st.markdown('<div class="sub-header">Ключевые метрики</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">Статей</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{df['author'].nunique()}</div>
                <div class="metric-label">Авторов</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{round(df['rating'].mean(), 1)}</div>
                <div class="metric-label">Средний рейтинг</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{df['rating'].max()}</div>
                <div class="metric-label">Максимальный рейтинг</div>
            </div>
            """, unsafe_allow_html=True)

        # Распределение рейтингов
        st.markdown('<div class="sub-header">Распределение рейтингов статей</div>', unsafe_allow_html=True)

        fig = px.histogram(df, x='rating', nbins=30, marginal='box',
                           title='Распределение рейтингов статей',
                           opacity=0.7, color_discrete_sequence=['steelblue'])
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

        # Ключевые выводы
        st.markdown("""
        <div class="insight-box">
        <strong>Ключевой вывод:</strong> Распределение рейтингов имеет выраженную правостороннюю асимметрию с преобладанием 
        статей с низким и средним рейтингом (0-50) и небольшим количеством материалов с экстремально высокими показателями.
        </div>
        """, unsafe_allow_html=True)

        # Облако слов
        st.markdown('<div class="sub-header">Ключевые темы в заголовках</div>', unsafe_allow_html=True)

        if 'title' in df.columns:
            # Создание облака слов из заголовков
            all_titles = ' '.join(df['title'].astype(str).dropna())
            stopwords = set(
                ['в', 'и', 'на', 'с', 'для', 'по', 'не', 'к', 'как', 'о', 'из', 'что', 'а', 'или', 'вы', 'мы',
                 'от', 'он', 'она', 'оно', 'они', 'мой', 'твой', 'ваш', 'наш', 'этот', 'тот', 'такой', 'так', 'при',
                 'the', 'of', 'and', 'to', 'in', 'a', 'is', 'that', 'for', 'on', 'with', 'by', 'as', 'an', 'be', 'it'])

            wordcloud = WordCloud(
                width=1000,
                height=500,
                background_color='white',
                colormap='viridis',
                stopwords=stopwords,
                max_words=150,
                contour_width=3,
                collocations=True,
                min_font_size=8
            ).generate(all_titles)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

            # Вывод ключевого вывода
            st.markdown("""
            <div class="insight-box">
            <strong>Ключевой вывод:</strong> Лексический анализ заголовков выявил превалирование технологической тематики с 
            акцентом на объяснительный формат ("почему", "как", "это"). Современные технологические тренды занимают заметное 
            место в тематическом ландшафте.
            </div>
            """, unsafe_allow_html=True)

    # Вкладка 2: Данные
    with tabs[1]:
        st.markdown('<div class="sub-header">Исследуемые данные</div>', unsafe_allow_html=True)

        # Фильтры данных
        col1, col2, col3 = st.columns(3)

        with col1:
            min_rating = st.slider("Минимальный рейтинг",
                                   int(df['rating'].min()),
                                   int(df['rating'].max()),
                                   int(df['rating'].min()))

        with col2:
            authors_list = ['Все'] + sorted(df['author'].value_counts().head(20).index.tolist())
            selected_author = st.selectbox("Автор", authors_list)

        with col3:
            # Фильтр по временному диапазону
            if 'date' in df.columns:
                min_date = df['date'].min().date()
                max_date = df['date'].max().date()
                date_range = st.date_input(
                    "Диапазон дат",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date, end_date = min_date, max_date

        # Применение фильтров
        filtered_df = df.copy()

        # Фильтр по рейтингу
        filtered_df = filtered_df[filtered_df['rating'] >= min_rating]

        # Фильтр по автору
        if selected_author != 'Все':
            filtered_df = filtered_df[filtered_df['author'] == selected_author]

        # Фильтр по датам
        if 'date' in df.columns and len(date_range) == 2:
            filtered_df = filtered_df[(filtered_df['date'].dt.date >= start_date) &
                                      (filtered_df['date'].dt.date <= end_date)]

        # Вывод отфильтрованных данных
        display_cols = ['title', 'author', 'date', 'rating']
        if 'views' in filtered_df.columns:
            display_cols.append('views')
        if 'comments' in filtered_df.columns:
            display_cols.append('comments')
        if 'weekday_name' in filtered_df.columns:
            display_cols.append('weekday_name')

        st.dataframe(filtered_df[display_cols].reset_index(drop=True),
                     use_container_width=True,
                     height=400)

        st.markdown(f"**Найдено записей:** {len(filtered_df)} из {len(df)}")

        # Статистика по данным
        st.markdown('<div class="sub-header">Статистические характеристики</div>', unsafe_allow_html=True)

        if 'rating' in df.columns:
            stats_metrics = ['rating']
            if 'views' in df.columns:
                stats_metrics.append('views')
            if 'comments' in df.columns:
                stats_metrics.append('comments')
            if 'text_length' in df.columns:
                stats_metrics.append('text_length')

            stats_df = pd.DataFrame()
            stats_df['Статистика'] = ['Среднее', 'Медиана', 'Минимум', 'Максимум', 'Стандартное отклонение']

            for metric in stats_metrics:
                stats_df[metric.capitalize()] = [
                    round(filtered_df[metric].mean(), 2),
                    round(filtered_df[metric].median(), 2),
                    filtered_df[metric].min(),
                    filtered_df[metric].max(),
                    round(filtered_df[metric].std(), 2)
                ]

            st.dataframe(stats_df, use_container_width=True)

    # Вкладка 3: Рейтинги
    with tabs[2]:
        st.markdown('<div class="sub-header">Анализ рейтингов</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # График зависимости рейтинга от длины текста
            if 'text_length' in df.columns:
                fig = px.scatter(df,
                                 x='text_length',
                                 y='rating',
                                 opacity=0.6,
                                 title='Взаимосвязь между длиной текста и рейтингом',
                                 labels={'text_length': 'Длина текста (символы)',
                                         'rating': 'Рейтинг'},
                                 color_discrete_sequence=['steelblue'],
                                 trendline="ols",
                                 trendline_color_override="red")
                st.plotly_chart(fig, use_container_width=True)

                # Корреляция
                corr = df['text_length'].corr(df['rating'])
                st.markdown(f"**Корреляция между длиной текста и рейтингом:** {corr:.3f}")

                if abs(corr) < 0.1:
                    st.markdown("""
                    <div class="insight-box">
                    <strong>Ключевой вывод:</strong> Объем текста не демонстрирует значимой корреляции с рейтинговыми показателями, 
                    что опровергает гипотезу о предпочтении аудиторией определенного формата статей по критерию длины.
                    </div>
                    """, unsafe_allow_html=True)

        with col2:
            # Топ-10 статей по рейтингу
            st.markdown('<div class="sub-header">Топ-10 статей по рейтингу</div>',
                        unsafe_allow_html=True)

            top_rated = df.sort_values('rating', ascending=False).head(10)
            fig = px.bar(top_rated,
                         y='title',
                         x='rating',
                         orientation='h',
                         title='Топ-10 статей по рейтингу',
                         labels={'title': 'Заголовок', 'rating': 'Рейтинг'},
                         height=500,
                         color_discrete_sequence=['steelblue'])
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        # Добавляем рассчет процентилей рейтингов
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(df['rating'], percentiles)

        percentile_df = pd.DataFrame({
            'Процентиль': [f"{p}%" for p in percentiles],
            'Значение рейтинга': percentile_values.round(1)
        })

        st.markdown('<div class="sub-header">Распределение рейтингов по процентилям</div>',
                    unsafe_allow_html=True)
        st.dataframe(percentile_df, use_container_width=True)

        # Интерпретация
        st.markdown(f"""
        <div class="insight-box">
        <strong>Интерпретация:</strong> Статья с рейтингом {percentile_values[4]:.1f} попадает в топ-10% всех статей по рейтингу. 
        Рейтинг {percentile_values[6]:.1f} и выше имеет только 1% статей, что делает такие материалы исключительно успешными.
        </div>
        """, unsafe_allow_html=True)

    # Вкладка 4: Временной анализ
    with tabs[3]:
        st.markdown('<div class="sub-header">Временные паттерны публикаций</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Средний рейтинг по дням недели
            weekday_avg = df.groupby('weekday_name').agg({
                'rating': ['mean', 'median', 'count']
            }).reset_index()

            # Объединение уровней мультииндекса
            weekday_avg.columns = ['weekday_name' if col == 'weekday_name' else f'rating_{col[1]}' for col in
                                   weekday_avg.columns]

            weekday_order = ['Понедельник', 'Вторник', 'Среда', 'Четверг',
                             'Пятница', 'Суббота', 'Воскресенье']

            # Безопасное применение категориального типа
            try:
                weekday_avg['weekday_name'] = pd.Categorical(
                    weekday_avg['weekday_name'],
                    categories=weekday_order,
                    ordered=True
                )
                weekday_avg = weekday_avg.sort_values('weekday_name')
            except:
                # В случае проблем с категориями
                pass

            fig = px.bar(weekday_avg,
                         x='weekday_name',
                         y='rating_mean',
                         title='Средний рейтинг по дням недели',
                         labels={'weekday_name': 'День недели',
                                 'rating_mean': 'Средний рейтинг'},
                         color_discrete_sequence=['steelblue'],
                         text='rating_mean')
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

            # Добавим метрики количества статей по дням недели
            st.markdown("**Количество статей по дням недели:**")

            weekday_count_fig = px.bar(weekday_avg,
                                       x='weekday_name',
                                       y='rating_count',
                                       labels={'weekday_name': 'День недели',
                                               'rating_count': 'Количество статей'},
                                       color_discrete_sequence=['lightblue'],
                                       text='rating_count')
            weekday_count_fig.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(weekday_count_fig, use_container_width=True)

        with col2:
            # Анализ по часам публикации
            hour_avg = df.groupby('hour').agg({
                'rating': ['mean', 'median', 'count']
            }).reset_index()

            # Объединение уровней мультииндекса
            hour_avg.columns = ['hour' if col == 'hour' else f'rating_{col[1]}' for col in hour_avg.columns]

            fig = px.bar(hour_avg,
                         x='hour',
                         y='rating_mean',
                         title='Средний рейтинг по часам публикации',
                         labels={'hour': 'Час публикации',
                                 'rating_mean': 'Средний рейтинг'},
                         color_discrete_sequence=['indianred'],
                         text='rating_mean')
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
            st.plotly_chart(fig, use_container_width=True)

            # Добавим тепловую карту часы-дни недели
            if len(df) > 50:  # Только если есть достаточно данных
                st.markdown('<div class="sub-header">Тепловая карта: час публикации vs день недели</div>',
                            unsafe_allow_html=True)

                # Создаем сводную таблицу
                heatmap_data = df.pivot_table(
                    values='rating',
                    index='weekday_name',
                    columns='hour',
                    aggfunc='mean'
                )

                # Сортируем индекс по дням недели
                if all(day in heatmap_data.index for day in weekday_order):
                    heatmap_data = heatmap_data.reindex(weekday_order)

                # Создаем тепловую карту
                fig = px.imshow(
                    heatmap_data,
                    labels=dict(x="Час публикации", y="День недели", color="Средний рейтинг"),
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    color_continuous_scale='Viridis',
                    aspect="auto"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        # Ключевые выводы по временному анализу
        st.markdown("""
        <div class="insight-box">
        <strong>Ключевые выводы по временным паттернам:</strong><br>
        1. Временной анализ публикационной активности демонстрирует чёткую корреляцию между днем недели и средним рейтингом публикаций.<br>
        2. Наблюдается заметная вариация рейтингов в зависимости от времени публикации — определённые часы демонстрируют аномально высокие показатели одобрения аудитории.<br>
        3. Оптимальные периоды для публикации можно использовать для планирования размещения материалов с целью максимизации их восприятия аудиторией.
        </div>
        """, unsafe_allow_html=True)

        # Распределение публикаций по месяцам и годам
        if 'year' in df.columns and 'month' in df.columns and df['year'].nunique() > 1:
            st.markdown('<div class="sub-header">Динамика по месяцам и годам</div>',
                        unsafe_allow_html=True)

            df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
            year_month_stats = df.groupby('year_month').agg({
                'title': 'count',
                'rating': 'mean'
            }).reset_index()
            year_month_stats.columns = ['year_month', 'article_count', 'avg_rating']

            # Создаем две вкладки для графиков
            ym_tabs = st.tabs(["Количество статей", "Средний рейтинг"])

            with ym_tabs[0]:
                fig = px.bar(year_month_stats,
                             x='year_month',
                             y='article_count',
                             title='Количество статей по месяцам',
                             labels={'year_month': 'Год-Месяц',
                                     'article_count': 'Количество статей'},
                             color_discrete_sequence=['steelblue'])
                fig.update_layout(xaxis={'categoryorder': 'category ascending'})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

            with ym_tabs[1]:
                fig = px.line(year_month_stats,
                              x='year_month',
                              y='avg_rating',
                              title='Средний рейтинг статей по месяцам',
                              labels={'year_month': 'Год-Месяц',
                                      'avg_rating': 'Средний рейтинг'},
                              markers=True)
                fig.update_layout(xaxis={'categoryorder': 'category ascending'})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

    # Вкладка 5: Авторы
    with tabs[4]:
        st.markdown('<div class="sub-header">Анализ авторского контента</div>',
                    unsafe_allow_html=True)

        # Статистика по авторам
        author_stats = df.groupby('author').agg({
            'title': 'count',
            'rating': ['mean', 'median', 'max', 'sum']
        }).reset_index()

        # Объединение уровней мультииндекса
        author_stats.columns = ['author' if col == 'author' else f"{col[0]}_{col[1]}" for col in author_stats.columns]

        # Переименование столбцов для удобства
        author_stats = author_stats.rename(columns={
            'title_count': 'articles_count',
            'rating_mean': 'avg_rating',
            'rating_median': 'median_rating',
            'rating_max': 'max_rating',
            'rating_sum': 'total_rating'
        })

        col1, col2 = st.columns(2)

        with col1:
            # Топ-10 авторов по количеству статей
            top_authors = author_stats.sort_values('articles_count', ascending=False).head(10)

            fig = px.bar(top_authors,
                         x='author',
                         y='articles_count',
                         title='Топ-10 авторов по количеству статей',
                         labels={'author': 'Автор',
                                 'articles_count': 'Количество статей'},
                         color_discrete_sequence=['steelblue'],
                         text='articles_count')
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(xaxis={'categoryorder': 'total descending'})
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Топ-10 авторов по среднему рейтингу (мин. 2 статьи)
            min_articles = st.slider("Минимальное количество статей у автора", 1, 10, 2)

            top_rated_authors = author_stats[author_stats['articles_count'] >= min_articles].sort_values(
                'avg_rating', ascending=False).head(10)

            fig = px.bar(top_rated_authors,
                         x='author',
                         y='avg_rating',
                         title=f'Топ-10 авторов по среднему рейтингу (мин. {min_articles} статьи)',
                         labels={'author': 'Автор',
                                 'avg_rating': 'Средний рейтинг'},
                         color_discrete_sequence=['indianred'],
                         text='avg_rating')
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig.update_layout(xaxis={'categoryorder': 'total descending'})
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        # Диаграмма рассеяния: количество статей vs средний рейтинг
        st.markdown('<div class="sub-header">Взаимосвязь между количеством статей и средним рейтингом авторов</div>',
                    unsafe_allow_html=True)

        fig = px.scatter(
            author_stats[author_stats['articles_count'] >= 2],
            x='articles_count',
            y='avg_rating',
            size='total_rating',
            hover_name='author',
            labels={
                'articles_count': 'Количество статей',
                'avg_rating': 'Средний рейтинг',
                'total_rating': 'Суммарный рейтинг'
            },
            title='Авторы: количество статей vs средний рейтинг',
            color_discrete_sequence=['steelblue']
        )
        st.plotly_chart(fig, use_container_width=True)

        # Таблица с полной статистикой по авторам (с фильтрацией)
        st.markdown('<div class="sub-header">Полная статистика по авторам</div>',
                    unsafe_allow_html=True)

        min_articles_filter = st.slider("Фильтр: минимальное количество статей", 1, 10, 1, key="author_table_filter")
        author_detailed = author_stats[author_stats['articles_count'] >= min_articles_filter].sort_values(
            'articles_count', ascending=False)

        # Округляем значения для отображения
        for col in ['avg_rating', 'median_rating']:
            if col in author_detailed.columns:
                author_detailed[col] = author_detailed[col].round(2)

        st.dataframe(author_detailed, use_container_width=True)

        # Ключевые выводы
        st.markdown("""
        <div class="insight-box">
        <strong>Ключевой вывод:</strong> Авторский фактор значимо дифференцирует восприятие материалов. 
        Выявлена группа авторов, стабильно получающих высокие оценки, независимо от временных параметров публикации. 
        При этом количество публикаций и качественная оценка контента представляют собой во многом независимые параметры 
        авторской деятельности на платформе.
        </div>
        """, unsafe_allow_html=True)

    # Вкладка 6: Корреляции
    with tabs[5]:
        st.markdown('<div class="sub-header">Корреляционный анализ</div>', unsafe_allow_html=True)

        # Выбираем числовые признаки для корреляционной матрицы
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Исключаем неинформативные столбцы
        exclude_cols = ['year', 'month', 'weekday', 'hour', 'days_live', 'day']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        if len(numeric_cols) > 1:
            # Позволяем пользователю выбрать переменные для корреляционного анализа
            selected_cols = st.multiselect(
                "Выберите переменные для корреляционного анализа",
                options=numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )

            if selected_cols:
                corr_matrix = df[selected_cols].corr()

                # Тепловая карта корреляций
                fig = px.imshow(
                    corr_matrix,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title="Корреляционная матрица числовых признаков",
                    text_auto='.2f'
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

                # Инструкция по интерпретации
                st.markdown("""
                <div class="insight-box">
                <strong>Как интерпретировать:</strong> Корреляционная матрица показывает взаимосвязь между переменными. 
                Значения близкие к 1 указывают на сильную положительную корреляцию, близкие к -1 на сильную отрицательную корреляцию, 
                а близкие к 0 на отсутствие линейной связи.
                </div>
                """, unsafe_allow_html=True)

                # Сильные корреляции (для удобства пользователя)
                strong_correlations = []

                for i in range(len(selected_cols)):
                    for j in range(i + 1, len(selected_cols)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) >= 0.5:  # Порог для "сильной" корреляции
                            strong_correlations.append({
                                'Переменная 1': selected_cols[i],
                                'Переменная 2': selected_cols[j],
                                'Корреляция': corr_value
                            })

                if strong_correlations:
                    st.markdown('<div class="sub-header">Сильные корреляции (|r| ≥ 0.5)</div>', unsafe_allow_html=True)
                    strong_corr_df = pd.DataFrame(strong_correlations)
                    strong_corr_df['Корреляция'] = strong_corr_df['Корреляция'].round(3)
                    st.dataframe(strong_corr_df.sort_values(by='Корреляция', ascending=False), use_container_width=True)
                else:
                    st.info("Не найдено сильных корреляций между выбранными переменными (|r| ≥ 0.5)")
            else:
                st.warning("Пожалуйста, выберите хотя бы одну переменную для анализа.")
        else:
            st.warning("Недостаточно числовых переменных для корреляционного анализа.")

    # Вывод информации об анализе в футер
    st.markdown("""
    ---
    <div class="footnote">
    <strong>Аналитический дашборд контента Хабра</strong><br>
    Данный дашборд представляет результаты анализа рейтингов и тематик публикаций на платформе Хабр. 
    Исследование выявило закономерности временного распределения контента, влияния времени публикации 
    на популярность материалов и определения ключевых паттернов авторской активности.
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Произошла ошибка при выполнении приложения: {e}")
    st.info("Попробуйте запустить приложение командой: streamlit run dashboard/app.py")