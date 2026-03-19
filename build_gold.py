#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd

# импортируем чанки
from chunking import final


# 1. загружаем gold set
df = pd.read_excel("goldenset.xlsx")


new_rows = []

#2. сопоставляем страницы с чанками
for _, row in df.iterrows():
    company = row["company"]
    page = int(row["pdf_page"])

    relevant_chunks = []

    chunks = final.get(company, [])

    for chunk in chunks:
        if chunk.metadata.get("page") == page:
            relevant_chunks.append(chunk.metadata["chunk_id"])

    if not relevant_chunks:
        print(f"нет чанков: {company}, стр {page}")

    new_rows.append({
        "company": company,
        "question": row["question"],
        "answer": row["answer"],
        "pdf_page": page,
        "relevant_chunks": ";".join(relevant_chunks)
    })


# 3. сохраняем
new_df = pd.DataFrame(new_rows)
new_df.to_csv(
    "gold_with_chunks.csv",
    index=False,
    encoding="utf-8-sig",
    sep=";"
)

print("gold_with_chunks.csv создан")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




