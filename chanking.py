from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
final = dict() #создаем словарь, куда будем сохранять ключи - названия документов, значения - чанки
for company, text in data.items():
    chunks = text_splitter.create_documents(text)
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