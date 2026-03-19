#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install langchain-text-splitters')


# In[13]:


from langchain_text_splitters import RecursiveCharacterTextSplitter
from read_pdf import data
from os.path import splitext
import re
from os.path import splitext

def normalize_name(name):
    name = splitext(name)[0]
    name = name.lower()

    # убираем юр формы
    name = re.sub(r'пao|мкпао|ao', '', name)

    # убираем спецсимволы
    name = re.sub(r'[«»"“”_]', '', name)

    name = name.strip()
    return name

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

final = dict()

for company, pages in data.items():
    all_chunks = []

    company_name = normalize_name(company)
    chunk_counter = 1

    # теперь идем по страницам
    for page_num, text in pages.items():
        chunks = text_splitter.create_documents([text])

        for chunk in chunks:
            chunk_id = f"{company_name}-{chunk_counter}"

            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["page"] = page_num 

            all_chunks.append(chunk)
            chunk_counter += 1

    final[company_name] = all_chunks


# Проверка
for company, chunks in final.items():
    print(f"Компания - {company}, чанков: {len(chunks)}")


# Примеры
count = 0
for company, chunks in final.items():
    for chunk in chunks:
        if count >= 50:
            break
        count += 1
        print(f"{count}. {chunk.metadata['chunk_id']} | page={chunk.metadata['page']}")
    if count >= 50:
        break


# In[5]:


import os
print(os.listdir())


# In[ ]:





# In[ ]:





# In[ ]:




