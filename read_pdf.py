#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pymupdf')


# In[3]:


import fitz  # PyMuPDF
import zipfile

def extract_text(pdf_data: bytes) -> dict:
    doc = fitz.open(stream=pdf_data, filetype="pdf")

    text_dict = {}

    for page_num, page in enumerate(doc, 1):
        text_dict[page_num] = page.get_text()

    doc.close()
    return text_dict


data = {}

with zipfile.ZipFile("DATA.zip", "r") as zip_data:
    file_list = zip_data.namelist()

    for file_name in file_list:
        clean_name = file_name.replace("Дата/", "")
        pdf_data = zip_data.read(file_name)

        page_dict = extract_text(pdf_data)

        # сохраняем dict страниц
        data[clean_name] = page_dict


# Проверка
for key, value in data.items():
    print(f"Ключ: {key}, страниц: {len(value)}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




