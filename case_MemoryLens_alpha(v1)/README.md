# 📹 MemoryLens

**MemoryLens** — это экспериментальная система для поиска по содержимому видео.  
Проект основан на классе `MemoryLens` из файла `main.py` и реализует полный конвейер обработки: от транскрипции аудио до поиска по кадрам.

---

## ⚙️ Основные Возможности

MemoryLens объединяет современные методы компьютерного зрения и обработки естественного языка:

1. **Транскрипция аудио** — преобразование звуковой дорожки в текст с помощью *Whisper* (`faster_whisper`).
2. **Сэмплирование кадров** — извлечение ключевых кадров из видео (`OpenCV`).
3. **Создание эмбеддингов** — генерация векторов для текста и изображений (*CLIP*, *SentenceTransformers*).
4. **Построение поисковых индексов** — гибридный индекс на базе *FAISS* (`DualIndex`).
5. **Поиск и реренкинг** — поиск релевантных кадров с уточнением результатов через *BLIP*-подписи (автоматические описания изображений).

---

## 🖼️ Примеры Работы
![Frontend Demo]("case MemoryLens-alpha(v1)/demo.jpg")
![Search Result Example]("case MemoryLens-alpha(v1)/dem.jpg")
![Extracted Frame Example]("case MemoryLens-alpha(v1)/frame_000250.jpg")

---

## 📂 Структура Проекта

В директории проекта находятся следующие ключевые файлы:

- `main.py` — ядро проекта, класс **MemoryLens** и весь пайплайн.
- `demo.jpg` — скриншот интерфейса (Frontend).
- `dem.jpg` — пример выходного изображения (результат поиска).
- `requirements.txt` — список зависимостей (рекомендуется).
- `demo.py` *(опционально)* — минимальное приложение на **FastAPI** для демонстрации.

---

## 🚀 Запуск Демонстрации (FastAPI)

### 1. Требования

- Python **3.8+**
- Установленные инструменты: `git`, `pip`
- Рекомендуемые пакеты:
  - `uvicorn`, `fastapi`, `pillow`
  - `faster-whisper`, `transformers`, `sentence-transformers`
  - `faiss-cpu`, `opencv-python`, `torch`

Установка зависимостей:

```bash
pip install fastapi uvicorn pillow faster-whisper transformers sentence-transformers faiss-cpu opencv-python torch
