import streamlit as st
import pandas as pd
import re
from rank_bm25 import BM25Okapi
from pathlib import Path
from openai import OpenAI

# Инициализация OpenRouter API
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets.get("OPENROUTER_API_KEY")
)

# Настройки
st.set_page_config(
    page_title="RAG • Финансовый Аналитик",
    page_icon="📊",
    layout="wide"
)

TOP_K = 5
CHUNKS_PATH = "chunks_final.csv"

# Закешируем чанки
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

# ===== Основной блок кода с реализацией интерфейса ====
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

# Чат
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Задайте вопрос по финансовым показателям компании..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Ищем чанки и генерируем ответ..."):
            # 1. BM25 retrieval
            tokenized_query = tokenize_func(prompt)
            scores = bm25_index.get_scores(tokenized_query)
            top_indices = scores.argsort()[::-1][:top_k]
            
            retrieved_chunks = [chunks_df.iloc[idx]["text"] for idx in top_indices]
            context = "\n\n---\n\n".join(retrieved_chunks)

            # 2. RAG-промпт (очень важная часть!)
            system_prompt = """Ты — профессиональный финансовый аналитик публичных компаний (российский и международный рынок).
Используй ТОЛЬКО информацию из предоставленного контекста.
Отвечай точно, структурировано, с цифрами и фактами.
Если в контексте нет информации — честно скажи об этом."""

            user_prompt = f"""Контекст из документов:
{context}

Вопрос пользователя: {prompt}

Дай полный и точный ответ:"""

            # 3. Вызов OpenRouter
            try:
                selected_model = "openai/gpt-oss-20b"
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2048,
                )
                answer = response.choices[0].message.content
                
                # Показываем использованные чанки (для прозрачности)
                with st.expander("🔍 Показать найденные чанки (BM25)"):
                    for i, text in enumerate(retrieved_chunks, 1):
                        st.write(f"**Чанк {i}** (score: {scores[top_indices[i-1]]:.4f})")
                        st.caption(text[:600] + "..." if len(text) > 600 else text)
                
                st.markdown(answer)
                
            except Exception as e:
                st.error(f"Ошибка OpenRouter: {e}")
                answer = "Извините, произошла ошибка при обращении к LLM."

    st.session_state.messages.append({"role": "assistant", "content": answer})
