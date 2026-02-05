import fitz # ИМпортируем PyMuPDF
import zipfile # Импортируем модуль для работы с zip-файлом


def extract_text(pdf_data: bytes) -> str:
  """
  Функция написана для преобразования извлеченной информации из PDF-файла в виде байтов

  params:
  - pdf_path: содержимое файлов в виде байтов

  output:
  - text: текст из ПДФ

  """
  doc = fitz.open(stream = pdf_data, filetype="pdf")
  text = ""

  for page in doc:
    text += page.get_text() + "\n"

  doc.close()

  return text


with zipfile.ZipFile("DATA.zip", "r") as zip_data: # Открываем папку
  file_list = zip_data.namelist() # С помощью функции namelist() считываем называния файлов в папке в список


  for file_name in file_list:                # Проходимся циклом по каждому названию
    pdf_data = zip_data.read(file_name)      # Считываем текст из PDF в папке в виде байтов
    text = extract_text(pdf_data)            # Пользуемся функцией для преобразования байтов в текст
    data[file_name] = [text]                 # сохраняем полученный результат в словарь




