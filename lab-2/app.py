import os
import pickle

import nltk
import numpy as np
import pandas as pd
import streamlit as st
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Классификатор Новостей", layout="centered")


@st.cache_resource
def load_best_model():
    """Загрузка лучшей модели и векторизатора"""
    try:
        with open("best_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("Файл 'best_model.pkl' не найден")
        return None


@st.cache_resource
def load_stopwords():
    """Загрузка стоп-слов для предобработки"""
    try:
        return set(stopwords.words("english"))
    except:
        return set()


def preprocess_text(text, stop_words):
    tokens = simple_preprocess(text)
    return [t for t in tokens if t not in stop_words]


def document_vector_w2v(doc, model, stop_words):
    """Создание вектора для Word2Vec (Mean Pooling)"""
    doc_vec = np.array([model.wv[word] for word in doc if word in model.wv])
    return doc_vec.mean(axis=0) if doc_vec.any() else np.zeros(model.vector_size)


st.title("Система автоматической категоризации контента")
st.markdown("""
Введите текст статьи, чтобы определить её тематику.
Если модель ошиблась - исправьте предсказание для дообучения.
""")

model_data = load_best_model()
stop_words = load_stopwords()

CATEGORIES = ["sci.space", "sci.med", "comp.graphics", "rec.sport.baseball"]
CATEGORY_NAMES = {
    "sci.space": "Космос и Наука",
    "sci.med": "Медицина и Здоровье",
    "comp.graphics": "Компьютерная Графика",
    "rec.sport.baseball": "Спорт (Бейсбол)",
}

with st.sidebar:
    st.header("Информация о модели")
    if model_data:
        st.info(f"Метод: {model_data.get('vectorization', 'Unknown')}")
        st.info(f"Алгоритм: {model_data.get('model_name', 'Unknown')}")
    else:
        st.warning("Модель не загружена")

    st.divider()
    st.header("Статистика")
    if os.path.exists("feedback.csv"):
        df_feedback = pd.read_csv("feedback.csv")
        st.metric("Всего отзывов", len(df_feedback))
        if "is_error" in df_feedback.columns:
            errors = df_feedback["is_error"].sum()
            st.metric("Ошибок модели", errors)
    else:
        st.metric("Всего отзывов", 0)

st.subheader("Ввод текста статьи")
user_input = st.text_area(
    "Введите текст новости:",
    height=200,
    placeholder="Example: The new telescope discovered a planet in the nearby galaxy...",
)

col1, col2 = st.columns([1, 4])
with col1:
    predict_button = st.button(
        "Классифицировать", type="primary", use_container_width=True
    )

if predict_button and user_input and model_data:
    model = model_data['model']
    vectorization_type = model_data['vectorization']
    categories = model_data.get('categories', CATEGORIES)
    
    with st.spinner("Обработка..."):
        try:
            if vectorization_type == 'TF-IDF':
                with open('vectorizer.pkl', 'rb') as f:
                    vectorizer = pickle.load(f)
                input_vec = vectorizer.transform([user_input])
                
            elif vectorization_type == 'Word2Vec':
                with open('w2v_model.pkl', 'rb') as f:
                    w2v_model = pickle.load(f)
                tokens = preprocess_text(user_input, stop_words)
                doc_vec = document_vector_w2v(tokens, w2v_model, stop_words)
                input_vec = doc_vec.reshape(1, -1)
            
            prediction_idx = model.predict(input_vec)[0]
            predicted_class = categories[prediction_idx]
            
            st.success(f"Предсказание: **{CATEGORY_NAMES.get(predicted_class, predicted_class)}**")
            st.session_state['last_prediction'] = predicted_class
            st.session_state['last_text'] = user_input[:200]
            
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")
            st.code("Проверьте: 1) файлы .pkl существуют, 2) векторизация совпадает с обучением")

if "last_prediction" in st.session_state:
    st.divider()
    st.subheader("Коррекция предсказания")
    st.write(
        f"Текущее предсказание: **{CATEGORY_NAMES.get(st.session_state['last_prediction'], st.session_state['last_prediction'])}**"
    )

    with st.form("feedback_form"):
        true_class = st.selectbox(
            "Если предсказание неверно, выберите правильный класс:",
            options=CATEGORIES,
            format_func=lambda x: CATEGORY_NAMES.get(x, x),
        )

        submitted = st.form_submit_button(
            "Отправить исправление", use_container_width=True
        )

        if submitted:
            predicted = st.session_state["last_prediction"]
            corrected = true_class
            is_error = predicted != corrected

            feedback_data = {
                "timestamp": pd.Timestamp.now(),
                "text": st.session_state["last_text"],
                "predicted": predicted,
                "corrected": corrected,
                "is_error": is_error,
                "confidence": confidence if "confidence" in locals() else 0.0,
            }

            df_new = pd.DataFrame([feedback_data])
            if not os.path.exists("feedback.csv"):
                df_new.to_csv("feedback.csv", index=False)
            else:
                df_new.to_csv("feedback.csv", mode="a", header=False, index=False)

            if is_error:
                st.info("Ошибка сохранена в feedback.csv для дообучения.")
            else:
                st.success("Подтверждение корректности сохранено.")

            del st.session_state["last_prediction"]
            st.rerun()

with st.expander("Посмотреть логи исправлений (feedback.csv)"):
    if os.path.exists("feedback.csv"):
        df_logs = pd.read_csv("feedback.csv")
        st.dataframe(df_logs, use_container_width=True)

        csv = df_logs.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Скачать логи (CSV)",
            csv,
            "feedback_logs.csv",
            "text/csv",
            key="download-csv",
        )
    else:
        st.write("Пока нет сохраненных отзывов.")
