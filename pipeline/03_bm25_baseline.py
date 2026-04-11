"""
03_bm25_baseline.py — Baseline решение с использованием BM25
"""

import re
import pandas as pd
from rank_bm25 import BM25Okapi
from pathlib import Path

print("=== BM25 Baseline — запуск ===\n")

# настройки
CHUNKS_PATH = "chunks_final.csv"
GOLD_PATH   = "gold_final.csv"
OUTPUT_PATH = "bm25_rankings.csv"
TOP_K = 10

# загрузка данных
print("Загружаем файлы...")
chunks_df = pd.read_csv(CHUNKS_PATH)
gold_df   = pd.read_csv(GOLD_PATH)

print(f"chunks_df: {chunks_df.shape}")
print(f"gold_df:   {gold_df.shape}\n")

# исправляем возможное расхождение в названии колонки
if "question_id" in gold_df.columns:
    gold_df = gold_df.rename(columns={"question_id": "query_id"})
    print("Колонка 'question_id' переименована в 'query_id'\n")

# предобработка данных
chunks_df["chunk_id"] = chunks_df["chunk_id"].astype(str).str.strip()
chunks_df["text"]     = chunks_df["text"].fillna("").astype(str).str.strip()

gold_df["query_id"] = gold_df["query_id"].astype(str).str.strip()
gold_df["question"] = gold_df["question"].fillna("").astype(str).str.strip()

# удаляем пустые чанки
chunks_df = chunks_df[chunks_df["text"] != ""].reset_index(drop=True)

print(f"После очистки осталось чанков: {len(chunks_df)}\n")

# токенизация
def simple_tokenize(text: str) -> list[str]:
    """Простая токенизация текста: нижний регистр + удаление спецсимволов"""
    text = str(text).lower()
    text = re.sub(r"[^a-zа-яё0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []

print("Токенизируем все чанки...")
tokenized_corpus = [simple_tokenize(text) for text in chunks_df["text"]]
print(f"Токенизировано {len(tokenized_corpus)} документов\n")

# построение bm25 индекса
print("Строим BM25 индекс... (это может занять некоторое время)")
bm25_index = BM25Okapi(tokenized_corpus)
print("BM25 индекс успешно построен ✓\n")

# функции для поиска
def retrieve_top_k_bm25(query_id, question, bm25_index, chunks_df, top_k=10):
    """Возвращает топ-K самых релевантных чанков для одного вопроса по BM25"""
    tokenized_query = simple_tokenize(question)
    scores = bm25_index.get_scores(tokenized_query)
    
    top_indices = scores.argsort()[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append({
            "query_id": query_id,
            "method": "bm25",
            "rank": rank,
            "chunk_id": chunks_df.iloc[idx]["chunk_id"],
            "score": float(scores[idx])
        })
    return pd.DataFrame(results)


def build_bm25_rankings(gold_df, bm25_index, chunks_df, top_k=10):
    """Строит полную таблицу ранжирования BM25 для всех вопросов"""
    results_list = []
    total = len(gold_df)
    
    print(f"Выполняем BM25 поиск для {total} вопросов...\n")
    
    for i, row in enumerate(gold_df.itertuples(), 1):
        if i % 20 == 0 or i == 1 or i == total:
            print(f"  Прогресс: {i}/{total} вопросов")
        
        res = retrieve_top_k_bm25(
            query_id=row.query_id,
            question=row.question,
            bm25_index=bm25_index,
            chunks_df=chunks_df,
            top_k=top_k
        )
        results_list.append(res)
    
    return pd.concat(results_list, ignore_index=True)


# запуск основного процесса
bm25_rankings_df = build_bm25_rankings(
    gold_df=gold_df,
    bm25_index=bm25_index,
    chunks_df=chunks_df,
    top_k=TOP_K
)

print(f"\nГотово! Создано строк: {len(bm25_rankings_df)}")

# сохранение результата
bm25_rankings_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

full_path = Path(OUTPUT_PATH).resolve()
print(f"\nФайл успешно сохранён:")
print(f"→ {full_path}")
print(f"Размер файла: {full_path.stat().st_size:,} байт")

print("\nПервые 10 строк результата:")
print(bm25_rankings_df.head(10)[["query_id", "rank", "chunk_id", "score"]])

print("\n=== Работа скрипта завершена ===")