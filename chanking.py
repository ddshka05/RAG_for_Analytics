from langchain_text_splitters import RecursiveCharacterTextSplitter
from read_pdf import data # обращаемся к созданному в другом файле словарю
from os.path import splitext # библиотека для того, чтобы убрать расширения при ID

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
final = dict() #создаем словарь, куда будем сохранять ключи - названия документов, значения - чанки
for company, text in data.items():
    chunks = text_splitter.create_documents(text)

    # === Блок создания ID для чанков ===
    company_name = splitext(company)[0] # Получаем название компании без расширения .pdf 
    for idx, chunk in enumerate(chunks, 1): # Проходим по каждому чанку и добавляем ID
        chunk_id = f"{company_name}-{idx}" # Формируем ID для каждого чанка
        chunk.metadata["chunk_id"] = chunk_id # Добавляем айди в метаданные к чанку

    final[company]=chunks

for i,j in final.items(): #тут я проверяла вывод чанков, что в итоге получается по кол-ву
    print(f" Компания - {i}, всего чанков: {len(j)}")


# Вывод для проверки
for company, chunks in final.items():
    print(f"\n{'='*60}")
    print(f"Компания: {company}")
    print(f"Всего чанков: {len(chunks)}")
    print(f"{'='*60}")
    
    # Выводим первые 3 чанка (или все, если их меньше 3)
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n Чанк №{i}:")
        print(f"Длина: {len(chunk.page_content)} символов")
        print(f"Метаданные: {chunk.metadata}")
        print(f"Содержимое:\n{chunk.page_content}")
    
    if len(chunks) > 3:
        print(f"\n... и ещё {len(chunks) - 3} чанков")


# Примеры ID, которые выводятся
count = 0
for company, chunks in final.items():
    for chunk in chunks:
        if count >= 100:
            break
        count += 1
        print(f"{count}. ID: {chunk.metadata['chunk_id']} | Длина: {len(chunk.page_content)}")
    if count >= 100:
        break