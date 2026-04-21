import pandas as pd
import re
from pathlib import Path
import pdfplumber
import zipfile
import os

script_dir = Path(__file__).parent.resolve()   # папка, где лежит этот .py файл

zip_path = script_dir / "DATA.zip"
extract_path = script_dir

print(f"Скрипт запущен из: {script_dir}")
print(f"Ищем архив: {zip_path}")
print(f"Распаковываем в: {extract_path}")

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:   #извлекаем файлы из зип архива для дальнейшего прочтения
    zip_ref.extractall(extract_path)

print("Архив распакован")
print()

def make_company_slug(name: str) -> str:   #создаем стабильные имена компаниям для сопоставления с остальными файлами
    name = Path(name).stem.lower()
    name = re.sub(r'[^a-zа-я0-9]+', ' ', name)
    name = name.strip(' ')
    return name

#проходим по всем пдф в папке, открываем каждый файл и извлекаем текст с каждой страницы,
#сохраняя при этом номер страницы для дальнейшего согласования с goldenset
def read_reports_from_folder(folder_path: str) -> dict:      
    reports = {}

    for file in Path(folder_path).rglob("*.pdf"):
        print(f"Читаю файл: {file}")

        pages_dict = {}

        with pdfplumber.open(file) as pdf:
            for i, page in enumerate(pdf.pages, start=1):

                if i % 20 == 0:
                    print(f"страница {i}")

                text = page.extract_text(x_tolerance=2, y_tolerance=2)

                if text:
                    pages_dict[i] = text

        if pages_dict:
            reports[str(file)] = pages_dict

    return reports


reports = read_reports_from_folder(extract_path)

print("Загружено файлов:", len(reports))
print()

def is_table_line(line: str) -> bool:      #эта функция пытается понять, является ли строка частью таблицы
                                           #логика: если много цифр или мало текста, то скорее всего таблица
    if not line:
        return False

    digits = sum(c.isdigit() for c in line)
    letters = sum(c.isalpha() for c in line)

    return digits >= 3 and letters < 50

def split_page_text(page_text: str, max_chunk_size: int = 800):     #чанкинг: не ломаем таблицы, режем только текст
                                                                    #если разделить таблицу, то мы теряем связь показатель+значение
    lines = page_text.split("\n")

    chunks = []
    current_chunk = ""
    current_type = None

    for line in lines:
        line = line.strip()

        if not line:
            continue

        line_type = "table" if is_table_line(line) else "text"

        if current_type is not None and line_type != current_type:
            chunks.append(current_chunk.strip())
            current_chunk = ""

        current_type = line_type
        current_chunk += line + "\n"

        if current_type == "text" and len(current_chunk) > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = ""
            current_type = None

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def normalize_chunks(chunks, min_size=150, max_size=600):         #эта функция убирает мусор(очень короткие остаточные куски), склеивает маленькие чанки, делит слишком большие

    new_chunks = []
    buffer = ""

    for chunk in chunks:
        chunk = chunk.strip()

        # убираем пустые сразу
        if not chunk:
            continue

        # фильтр мусора
        letters = sum(c.isalpha() for c in chunk)
        if len(chunk) < 20 or letters < 5:
            continue

        # маленькие чанки → копим
        if len(chunk) < min_size:
            buffer += " " + chunk
            continue

        # если есть буфер → приклеиваем
        if buffer:
            chunk = buffer + " " + chunk
            buffer = ""

        # режем большие чанки
        if len(chunk) > max_size:
            for i in range(0, len(chunk), max_size):
                part = chunk[i:i+max_size].strip()

                if len(part) > 50:  # защита от мусора
                    new_chunks.append(part)
        else:
            new_chunks.append(chunk)

    # остаток буфера
    if buffer.strip() and len(buffer) > 50:
        new_chunks.append(buffer.strip())

    return new_chunks

def build_chunks_dataframe(reports: dict) -> pd.DataFrame:       #собираем финальный датасет чанков
                                                                 # Важно:
                                                                 #chunk_id уникален
                                                                 #есть привязка к компании и странице
                                                                 #текст очищен от \n
    rows = []

    for source_file in sorted(reports.keys()):
        pages = reports[source_file]

        company_raw = Path(source_file).stem
        company_slug = make_company_slug(source_file)

        for page_num in sorted(pages.keys()):
            page_text = pages[page_num]

            chunks = split_page_text(page_text)
            chunks = normalize_chunks(chunks)

            final_chunks = []
            for chunk in chunks:
              if len(chunk) <= 600:
                final_chunks.append(chunk)
              else:
                for i in range(0, len(chunk), 600):
                  part = chunk[i:i+600].strip()
                  if len(part) > 50:
                    final_chunks.append(part)

            chunks = final_chunks

            for idx, chunk_text in enumerate(chunks, start=1):
              clean_text = chunk_text.replace("\n", " ").strip()
              chunk_id = f"{company_slug}-p{page_num:03d}-c{idx:03d}"
              rows.append({
                  "chunk_id": chunk_id,
                  "company_raw": company_raw,
                  "company_slug": company_slug,
                  "source_file": source_file,
                  "pdf_page": page_num,
                  "chunk_idx_on_page": idx,
                  "text": clean_text,   # ← ВАЖНО
                  "n_chars": len(clean_text)
                  })

    return pd.DataFrame(rows)

chunks_df = build_chunks_dataframe(reports)
chunks_df = chunks_df[chunks_df["text"].str.strip() != ""]
chunks_df["n_chars"] = chunks_df["text"].str.len()

print("Всего чанков:", len(chunks_df))
chunks_df.head()

def validate_chunks_dataframe(df: pd.DataFrame): #проверка

    assert df["chunk_id"].is_unique, "chunk_id не уникальны"    #нет дубликатов
    assert (df["chunk_idx_on_page"] >= 1).all(), "chunk_idx_on_page < 1"
    assert df["text"].fillna("").str.strip().ne("").all(), "пустые чанки" # нет пустых чанков
    assert (df["n_chars"] == df["text"].str.len()).all(), "n_chars ошибка" #длина текста совпадает

    print("Проверка успешна")


validate_chunks_dataframe(chunks_df)

output_path = script_dir  #сохраняем таблицу с чанками в папку, откуда запущен код

chunks_df.to_csv(output_path, index=False)

print("Файл сохранен:", output_path)
