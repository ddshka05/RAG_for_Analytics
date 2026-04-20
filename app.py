import streamlit as st
import pandas as pd
import re
from rank_bm25 import BM25Okapi
from pathlib import Path

# ====================== НАСТРОЙКИ ======================
st.set_page_config(
    page_title="RAG • Финансовый Аналитик",
    page_icon="📊",
    layout="wide"
)

TOP_K = 5
CHUNKS_PATH = "chunks_final.csv"

# ====================== КЭШИРОВАНИЕ ======================
@st.cache_data(show_spinner="Загружаем корпус чанков...")
def load_chunks():
    df = pd.read_csv(CHUNKS_PATH)
    df["chunk_id"] = df["chunk_id"].astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    # убираем пустые чанки
    df = df[df["text"] != ""].reset_index(drop=True)
    return df

@st.cache_resource(show_spinner="Строим BM25-индекс...")
def build_bm25_index(chunks_df):
    def simple_tokenize(text: str):
        text = str(text).lower()
        text = re.sub(r"[^a-zа-яё0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split() if text else []
    
    tokenized_corpus = [simple_tokenize(text) for text in chunks_df["text"]]
    return BM25Okapi(tokenized_corpus), simple_tokenize

# ====================== ОСНОВНОЙ КОД ======================
chunks_df = load_chunks()
bm25_index, tokenize_func = build_bm25_index(chunks_df)

st.title("📊 RAG для финансового аналитика публичных компаний")
st.markdown("""
**Демонстрация работы Retrieval-Augmented Generation**  
Сейчас используется **BM25 baseline**.
""")

# боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")
    top_k = st.slider("Количество чанков (Top-K)", 3, 15, TOP_K)
    st.info(f"Всего чанков в базе: **{len(chunks_df)}**")

# ====================== ЧАТ ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

# отображаем историю
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# поле ввода
if prompt := st.chat_input("Задайте вопрос про финансовые показатели компании..."):
    # добавляем сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ====================== RETRIEVAL ======================
    with st.chat_message("assistant"):
        with st.spinner("Ищем релевантные чанки по BM25..."):
            tokenized_query = tokenize_func(prompt)
            scores = bm25_index.get_scores(tokenized_query)
            
            top_indices = scores.argsort()[::-1][:top_k]
            
            retrieved = []
            for rank, idx in enumerate(top_indices, 1):
                row = chunks_df.iloc[idx]
                retrieved.append({
                    "rank": rank,
                    "chunk_id": row["chunk_id"],
                    "score": round(float(scores[idx]), 4),
                    "text": row["text"]
                })

        # показываем найденные чанки
        st.markdown("### 🔍 Найденные чанки (BM25)")
        for item in retrieved:
            with st.expander(f"Rank {item['rank']} • chunk_id: {item['chunk_id']} • score: {item['score']}"):
                st.write(item["text"][:800] + "..." if len(item["text"]) > 800 else item["text"])
                st.caption(f"Полный текст чанка — {len(item['text'])} символов")

        # ====================== ЗАГЛУШКА ГЕНЕРАЦИИ ======================
        st.markdown("### 🤖 Ответ RAG-модели")
        
        # TODO: сюда позже подключишь настоящий LLM (Ollama / Groq / Grok API / Llama-3 и т.д.)

        # сохраняем в историю
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer_placeholder
        })

# ====================== НИЖНЯЯ ИНФО ======================
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Метод поиска", "BM25")
with col2:
    st.metric("Размер корпуса", f"{len(chunks_df)} чанков")
with col3:
    st.metric("Версия демо", "v1")
