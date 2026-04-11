# Импортируем необходимые библиотеки
import json
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# Настройки
DENSE_RANKINGS_PATH = "dense_rankings.csv"
MODEL_NAME = "intfloat/multilingual-e5-large"

TOP_K = 10
BATCH_SIZE = 32

# Загрузим данные
print("Загружаем данные...")

chunks_df = pd.read_csv("chunks_final.csv")
gold_df = pd.read_csv("gold_final.csv")

print(f"chunks_df: {chunks_df.shape}")
print(f"gold_df:   {gold_df.shape}")

# Предобработка данных
required_chunk_cols = ["chunk_id", "text"]
required_gold_cols = ["question_id", "question", "chunk_id"]

assert all(col in chunks_df.columns for col in required_chunk_cols), "Не хватает колонок в chunks_final.csv"
assert all(col in gold_df.columns for col in required_gold_cols), "Не хватает колонок в gold_final.csv"

# Приводим к строкам и очищаем
chunks_df["chunk_id"] = chunks_df["chunk_id"].astype(str).str.strip()
chunks_df["text"]    = chunks_df["text"].fillna("").astype(str).str.strip()

gold_df["question_id"] = gold_df["question_id"].astype(str).str.strip()
gold_df["question"]    = gold_df["question"].fillna("").astype(str).str.strip()
gold_df["chunk_id"]    = gold_df["chunk_id"].astype(str).str.strip()

# Удаляем пустые чанки
chunks_df = chunks_df[chunks_df["text"] != ""].reset_index(drop=True)

print("Предобработка данных завершена\n")

# Загрузим данные
print(f"Загружаем модель: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print("Модель успешно загружена\n")

# Подготовка текстов с префиксами
passage_texts = [f"passage: {text}" for text in chunks_df["text"].tolist()]
query_texts   = [f"query: {text}" for text in gold_df["question"].tolist()]

print(f"Подготовлено {len(passage_texts):,} чанков и {len(query_texts)} вопросов\n")

# Кодируем в эмбеддинги
def encode_texts(model, texts, batch_size=32, show_progress=True):
    """Кодирует список текстов в эмбеддинги"""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress
    )
    return embeddings


# Делаем эмбеддинги чанков
print("Создаём эмбеддинги для всех чанков...")
chunk_embeddings = encode_texts(model, passage_texts, batch_size=BATCH_SIZE)
print(f"Эмбеддинги чанков готовы: {chunk_embeddings.shape}\n")

# Эмбеддинги вопросов
print("Создаём эмбеддинги для вопросов...")
query_embeddings = encode_texts(model, query_texts, batch_size=BATCH_SIZE)
print(f"Эмбеддинги вопросов готовы: {query_embeddings.shape}\n")

# Retrieval функция для вывода топ-k релевантных чанков
def retrieve_top_k_dense(question_id, query_embedding, chunk_embeddings, chunks_df, top_k=10):
    """Возвращает топ-K самых релевантных чанков для одного вопроса"""
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(sorted_indices, start=1):
        results.append({
            'question_id': question_id,
            'method': 'dense',
            'rank': rank,
            'chunk_id': chunks_df.iloc[idx]['chunk_id'],
            'score': float(similarities[idx])
        })
    return pd.DataFrame(results)


def build_dense_rankings(gold_df, query_embeddings, chunk_embeddings, chunks_df, top_k=10):
    """Строит полную таблицу ранжирования для всех вопросов"""
    results_list = []

    print(f"Выполняем dense retrieval для {len(gold_df)} вопросов...")

    for i in range(len(gold_df)):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Обработано {i+1}/{len(gold_df)} вопросов...")

        row = gold_df.iloc[i]
        result = retrieve_top_k_dense(
            question_id=row['question_id'],
            query_embedding=query_embeddings[i],
            chunk_embeddings=chunk_embeddings,
            chunks_df=chunks_df,
            top_k=top_k
        )
        results_list.append(result)

    dense_rankings_df = pd.concat(results_list, ignore_index=True)
    print("Dense retrieval завершён!\n")
    return dense_rankings_df


# Запустим модель
dense_rankings_df = build_dense_rankings(
    gold_df=gold_df,
    query_embeddings=query_embeddings,
    chunk_embeddings=chunk_embeddings,
    chunks_df=chunks_df,
    top_k=TOP_K
)

print(f"Итоговая таблица: {dense_rankings_df.shape}")

# Сохраним результаты
dense_rankings_df.to_csv(DENSE_RANKINGS_PATH, index=False, encoding="utf-8-sig")
print(f"Результат сохранён в файл: {DENSE_RANKINGS_PATH}")

# Проверка
print("\nПервые 10 строк результата:")
print(dense_rankings_df.head(10))
