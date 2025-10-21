import os
import json
import gc
import shutil
import uuid
from typing import List, Literal, Optional

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field


from main import MemoryLens
from utils import Embeddings, DualIndex, BlipCaptioner, WhisperModel

# ---Basic Configuration / КОНФИГУРАЦИЯ ---
VIDEO_FOLDER = r"D:\videos\whisper_data"
CACHE_DIR = os.path.join(VIDEO_FOLDER, "_cache")
FRAMES_DIR = os.path.join(VIDEO_FOLDER, "_frames")

app = FastAPI(title="MemoryLens API", version="2.0")
app.mount("/videos", StaticFiles(directory=VIDEO_FOLDER), name="videos")

model_cache = {
    "embs": None, "blip": None, "index": None
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5501"],  # config with your url / настройка под свой адрес 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Pydantic Models / Модели ---
class QueryRequest(BaseModel):
    """Модель для входящего поискового запроса."""
    query: str = Field(..., description="Текстовый запрос для поиска.")
    mode: Literal["auto", "text", "image", "both"] = Field("auto", description="Режим поиска.")
    topk_search: int = Field(25, description="Количество кандидатов для первоначального поиска.")
    topk_rerank: int = Field(3, description="Количество финальных результатов после переранжирования.")
    video_id: Optional[str] = Field(None, description="Опциональный ID видео для поиска только в нем.") # <-- ДОБАВЛЕНО

# --- I edited utils function a little bit with my params ---
# --- Вспомогательная функция (с небольшими доработками) ---
def process_and_index_video(video_path: str, video_id: str):
    """Обработка одного видео и добавление его в индекс"""
    print(f"\nНачало индексации видео ID: {video_id}")
    
    # 1.Transcriptions / Транскрипция
    whisper = WhisperModel("small", device=DEVICE, compute_type="int8")
    # Using unique cache data / Используем уникальный кэш-файл
    transcription_path = os.path.join(CACHE_DIR, f"{video_id}_transcription.json") 
    transcribed = MemoryLens.load_or_transcribe(
        os.path.dirname(video_path), whisper, transcription_path, language="en"
    )
    # Correcting filepath with name / Корректируем, чтобы транскрибировался только нужный файл
    transcribed_item = [item for item in transcribed if item["filename"] == os.path.basename(video_path)][0]
    
    del whisper
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

    # 2.Frames / Кадры
    frames_info = MemoryLens.load_or_extract_frames(os.path.dirname(video_path), FRAMES_DIR, fps_sample=1.0)
    # Filtering frames for current ID / Фильтруем кадры только для текущего видео ID
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_info = [f for f in frames_info if f[0].startswith(os.path.join(FRAMES_DIR, video_base_name))]

    # 3.Embeddings / Эмбеддинги
    text_vecs, text_meta = MemoryLens.builder_text_embeddings(
        transcribed=[transcribed_item], embs=model_cache["embs"]
    )
    for m in text_meta: m["video_id"] = video_id

    image_vecs, image_meta = MemoryLens.builder_image_embeddings(
        frames_info=frames_info, embs=model_cache["embs"]
    )
    for m in image_meta: m["video_id"] = video_id
    
    # 4.Indexing / Добавляем в индекс
    model_cache["index"].add_text(text_vecs, text_meta)
    model_cache["index"].add_image(image_vecs, image_meta)
    print(f"✓ Видео ID: {video_id} успешно проиндексировано. Текст: {len(text_meta)}, Визуал: {len(image_meta)}")


# --- Startup ---
@app.on_event("startup")
async def startup_event():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    # Caching models / Загружаем модели один раз
    model_cache["embs"] = Embeddings(device=DEVICE)
    model_cache["blip"] = BlipCaptioner(device=DEVICE)
    model_cache["index"] = DualIndex(dim_text=model_cache["embs"].text_model.get_sentence_embedding_dimension(), 
                                     dim_image=512) # 512 - размерность CLIP Base
    
    print("✓ Модели загружены, индекс инициализирован пустым.")

    # Step for loading video / ШАГ ЗАГРУЗКИ ВИДЕО    
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith((".mp4", ".mkv", ".mov", ".avi"))]
    if video_files:
        print(f"\n[!] Обнаружено {len(video_files)} видео в папке. Начинаем первичную индексацию...")
        for file in video_files:
            video_path = os.path.join(VIDEO_FOLDER, file)
            video_id = os.path.splitext(file)[0]
            process_and_index_video(video_path, video_id)
        print("✓ Первичная индексация завершена.")


# --- Upload Endpoints/ Эндпоинты  ---
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not model_cache["index"]:
         raise HTTPException(status_code=503, detail="Индекс еще не инициализирован. Попробуйте позже.")
         
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(status_code=400, detail="Поддерживаются только видеофайлы")

    #Saving / Сохраняем файл в папку
    video_id = uuid.uuid4().hex
    save_path = os.path.join(VIDEO_FOLDER, f"{video_id}_{file.filename}")

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении файла: {e}")

    # Simple processing / Обработка и индексация
    try:
        process_and_index_video(save_path, video_id)
    except Exception as e:
        # NOTICE: in this step we deleting file
        # Важно: если индексация упала, удаляем файл
        os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке и индексации видео: {e}")

    return {"status": "ok", "video_id": video_id, "message": "Видео загружено и проиндексировано"}


# --- Search Эндпоинт (обновлен для video_id) ---
@app.post("/search", summary="Поиск по видео")
async def search(request: QueryRequest):
    if not model_cache["index"]:
         raise HTTPException(status_code=503, detail="Сервис недоступен: индекс еще не инициализирован.")
         
    print(f"\nПолучен запрос: query='{request.query}', mode='{request.mode}', video_id='{request.video_id}'")
    
    results = MemoryLens.run_query(
        query=request.query,
        embs=model_cache["embs"],
        index=model_cache["index"],
        blip=model_cache["blip"],
        mode=request.mode,
        topk_search=request.topk_search,
        topk_rerank=request.topk_rerank,
        video_id=request.video_id
    )
    
    print(f"Отправка {len(results)} результатов.")
    return {"query": request.query, "results": results}


# --- Running uvicorn / Запуск сервера ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5500)