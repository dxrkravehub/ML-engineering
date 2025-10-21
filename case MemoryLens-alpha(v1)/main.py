from faster_whisper import WhisperModel
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from utils import Embeddings, DualIndex, BlipCaptioner # utils.py остается без изменений

import requests
import os, cv2
import torch, faiss
import numpy as np
from typing import Any, List, Tuple, Literal,Optional
from collections import defaultdict
import json
import gc

class MemoryLens:
    @staticmethod
    def load_or_transcribe(video_folder: str, whisper_model: WhisperModel, cache_path: str, language: str = "ru") -> list[dict]:
        if os.path.exists(cache_path):
            print(f"✓ Загрузка транскрипции из кэша: {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f: return json.load(f)
        else:
            print("Транскрибирование видео (сохранение в кэш)...")
            results = MemoryLens.transcribe_video_paths(video_folder, whisper_model, language)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f: json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✓ Транскрипция сохранена в: {cache_path}")
            return results

    @staticmethod
    def load_or_extract_frames(video_folder: str, frames_root_dir: str, fps_sample: float = 1.0) -> List[Tuple[str, float]]:
        frames_info_all = []
        print("\nПроверка и извлечение кадров из видео...")
        video_files = [f for f in sorted(os.listdir(video_folder)) if f.lower().endswith((".mp4", ".mkv", ".mov", ".avi"))]
        for file in video_files:
            video_name = os.path.splitext(file)[0]
            out_dir = os.path.join(frames_root_dir, video_name)
            if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
                print(f"✓ Кадры для '{file}' уже существуют. Загрузка путей...")
                frames = [(os.path.join(out_dir, frame_file), int(frame_file.split('_')[1].split('.')[0]) / 30) for frame_file in sorted(os.listdir(out_dir)) if frame_file.endswith(".jpg")]
                frames_info_all.extend(frames)
            else:
                print(f"Извлечение кадров для '{file}'...")
                vpath = os.path.join(video_folder, file)
                frames = MemoryLens.sample_frames_cv2(vpath, out_dir, fps_sample)
                frames_info_all.extend(frames)
        print(f"\nВсего найдено и извлечено кадров: {len(frames_info_all)}")
        return frames_info_all

    @staticmethod
    def load_or_build_embeddings(name: str, cache_dir: str, builder_fn, builder_args: dict) -> Tuple[np.ndarray, List[dict]]:
        vecs_path, meta_path = os.path.join(cache_dir, f"{name}_vecs.npy"), os.path.join(cache_dir, f"{name}_meta.json")
        if os.path.exists(vecs_path) and os.path.exists(meta_path):
            print(f"✓ Загрузка {name} эмбеддингов из кэша...")
            vecs, meta = np.load(vecs_path), json.load(open(meta_path, 'r', encoding='utf-8'))
            return vecs, meta
        else:
            print(f"Создание {name} эмбеддингов...")
            vecs, meta = builder_fn(**builder_args)
            os.makedirs(cache_dir, exist_ok=True)
            np.save(vecs_path, vecs)
            with open(meta_path, 'w', encoding='utf-8') as f: json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"✓ {name} эмбеддинги сохранены.")
            return vecs, meta
    
    @staticmethod
    def transcribe_video_paths(video_folder: str, model: WhisperModel, language: str = "ru") -> list[dict]:
        results = []
        video_files = [f for f in sorted(os.listdir(video_folder)) if f.lower().endswith((".mp4", ".mkv", ".mov", ".avi"))]
        for file in video_files:
            path = os.path.join(video_folder, file)
            print(f"  Транскрибирование: {file}")
            segments, info = model.transcribe(path, language=language, temperature=0, beam_size=5, best_of=5)
            segs = [{"start": seg.start, "end": seg.end, "text": seg.text.strip()} for seg in segments]
            results.append({"filename": file, "segments": segs})
        return results

    @staticmethod
    def chunk_segments(segments: List[dict], min_len: float = 5.0, max_len: float = 15.0) -> List[dict]:
        chunks, cur, cur_start, cur_end, cur_len = [], [], None, None, 0.0
        for seg in segments:
            if not cur:
                cur_start, cur_end, cur_len, cur = seg["start"], seg["end"], seg["end"] - seg["start"], [seg["text"]]
            else:
                proposed_end, proposed_len = seg["end"], seg["end"] - cur_start
                if proposed_len <= max_len or (cur_len < min_len and seg["end"] - seg["start"] < min_len):
                    cur.append(seg["text"])
                    cur_end, cur_len = proposed_end, proposed_len
                else:
                    chunks.append({"start": cur_start, "end": cur_end, "text": " ".join(cur).strip()})
                    cur_start, cur_end, cur_len, cur = seg["start"], seg["end"], seg["end"] - seg["start"], [seg["text"]]
        if cur:
            chunks.append({"start": cur_start, "end": cur_end, "text": " ".join(cur).strip()})
        return chunks

    @staticmethod
    def sample_frames_cv2(video_path, out_dir, fps_sample=1.0):
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise RuntimeError(f"Не удалось открыть {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(int(round(fps / fps_sample)), 1)
        saved, frame_idx = [], 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if frame_idx % step == 0:
                fpath = os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg")
                if cv2.imwrite(fpath, frame): saved.append((fpath, frame_idx / fps))
            frame_idx += 1
        cap.release()
        print(f"  Извлечено кадров: {len(saved)}")
        return saved

    @staticmethod
    def builder_text_embeddings(transcribed: List[dict], embs: Embeddings) -> Tuple[np.ndarray, List[dict]]:
        text_items, metas = [], []
        for item in transcribed:
            chunks = MemoryLens.chunk_segments(item["segments"])
            for ch in chunks:
                text_items.append(ch["text"])
                metas.append({"filename": item["filename"], "start": ch["start"], "end": ch["end"], "text": ch["text"]})
        text_vecs = embs.encode_text(text_items)
        return text_vecs, metas

    @staticmethod
    def builder_image_embeddings(frames_info: List[Tuple[str, float]], embs: Embeddings) -> Tuple[np.ndarray, List[dict]]:
        frame_paths = [p for p, _ in frames_info]
        img_vecs = embs.encode_images(frame_paths)
        metas = [{"frame_path": p, "time_sec": t} for p, t in frames_info]
        return img_vecs, metas

    @staticmethod
    def is_visual_query(query: str) -> bool:
        visual_keywords = ["покажи", "картинка", "кадр", "видно", "экран", "frame", "image", "визуально"]
        return any(k in query.lower() for k in visual_keywords)

    @staticmethod
    def search_pipeline(query: str, embs: Embeddings, index: DualIndex, topk: int = 10, mode: Literal["auto", "text", "image", "both"] = "auto") -> List[dict]:
        results_text, results_image = [], []
        use_text = mode in ["text", "both"] or (mode == "auto" and not MemoryLens.is_visual_query(query))
        use_image = mode in ["image", "both"] or (mode == "auto" and MemoryLens.is_visual_query(query))
        if use_text:
            qvec = embs.encode_text([query])
            results_text = index.search_text(qvec, topk=topk)
            print(f"Найдено {len(results_text)} кандидатов в текстовом индексе.")
        if use_image:
            inputs = embs.clip_proc(text=[query], return_tensors="pt")
            with torch.no_grad():
                txt_feat = embs.clip_model.get_text_features(**{k: v.to(embs.device) for k, v in inputs.items()})
                txt_feat /= txt_feat.norm(p=2, dim=-1, keepdim=True)
            qvec = txt_feat.cpu().numpy().astype("float32")
            results_image = index.search_image(qvec, topk=topk)
            print(f"Найдено {len(results_image)} кандидатов в визуальном индексе.")
        return sorted(results_text + results_image, key=lambda x: x['score'], reverse=True)

    @staticmethod
    def format_timecode(seconds: float) -> str:
        h, rem = divmod(seconds, 3600); m, s = divmod(rem, 60); ms = (s % 1) * 1000
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{int(ms):03d}"

    @staticmethod
    def filter_by_threshold(results: List[dict], threshold: float) -> List[dict]:
        return [r for r in results if r.get("score", 0) >= threshold]

    @staticmethod
    def group_by_time(results: List[dict], window_sec: float = 3.0) -> List[dict]:
        if not results: return []
        sorted_results = sorted(results, key=lambda x: x.get("start", x.get("time_sec", 0)))
        grouped, current_group = [], [sorted_results[0]]
        for r in sorted_results[1:]:
            r_time = r.get("start", r.get("time_sec", 0))
            last_time = current_group[-1].get("start", current_group[-1].get("time_sec", 0))
            if abs(r_time - last_time) <= window_sec: current_group.append(r)
            else:
                grouped.append(max(current_group, key=lambda x: x.get("score", 0)))
                current_group = [r]
        if current_group: grouped.append(max(current_group, key=lambda x: x.get("score", 0)))
        return grouped

    # =================================================================================
    # НОВЫЙ МЕТОД ДЛЯ ПЕРЕРАНЖИРОВАНИЯ
    # =================================================================================
    @staticmethod
    def rerank_with_blip(
        query: str, 
        candidates: List[dict], 
        blip: BlipCaptioner, 
        embs: Embeddings,
        top_k: int = 3
    ) -> List[dict]:
        """Переранжирует кандидатов-изображений с помощью BLIP."""
        print(f"\nПереранжирование топ-{len(candidates)} кандидатов с помощью BLIP...")
        if not candidates:
            return []

        # 1. Генерируем кэпшены для всех кандидатов
        paths = [c["frame_path"] for c in candidates]
        captions = blip.caption_paths(paths)

        # 2. Считаем семантическую близость между запросом и каждым кэпшеном
        q_vec = embs.encode_text([query])
        caption_vecs = embs.encode_text(captions)
        
        # Косинусная близость (т.к. векторы нормализованы, это просто скалярное произведение)
        rerank_scores = (q_vec @ caption_vecs.T)[0]

        # 3. Добавляем кэпшены и новые скоры к результатам
        for i, candidate in enumerate(candidates):
            candidate["blip_caption"] = captions[i]
            candidate["rerank_score"] = float(rerank_scores[i])

        # 4. Сортируем по новому скору и возвращаем top_k
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    # =================================================================================
    # ОБНОВЛЕННЫЙ `run_query`
    # =================================================================================


    @staticmethod
    def run_query(
        query: str, embs: Embeddings, index: DualIndex, blip: BlipCaptioner = None, 
        topk_search: int = 20, 
        threshold: float = 0.25, 
        group_window: float = 3.0,
        mode: Literal["auto", "text", "image", "both"] = "auto",
        topk_rerank: int = 3,
        video_id: Optional[str] = None
    ):
        # 1. Поиск (грубый поиск по всему индексу)
        # Мы оставляем topk_search как параметр, но помни, что 20-50 может быть мало.
        raw = MemoryLens.search_pipeline(query, embs, index, topk=topk_search, mode=mode)
        
        # 2. Фильтрация по video_id
        if video_id:
            print(f"Фильтрация результатов по video_id: {video_id}")
            # Важно: Используем str() для безопасного сравнения, если типы ID не совпадают
            raw = [r for r in raw if str(r.get("video_id", "")) == str(video_id)]

        # --- DEBUG PRINT: СКОЛЬКО КАНДИДАТОВ ОСТАЛОСЬ ---
        print(f"Кандидатов после фильтрации по video_id: {len(raw)}") 
        
        # Если кандидатов нет, возвращаем пустой список, чтобы избежать ошибок
        if not raw:
            return [] 

        # 3. Фильтрация и группировка
        # ... (остальная логика остается прежней)
        
        filtered = MemoryLens.filter_by_threshold(raw, threshold)
        grouped = MemoryLens.group_by_time(filtered, window_sec=group_window)

        # 4. BLIP Re-ranking 
        # ... (логика BLIP и финальное форматирование)
        if blip:
            image_candidates = [g for g in grouped if "frame_path" in g]
            if image_candidates:
                final_results_data = MemoryLens.rerank_with_blip(
                    query, image_candidates, blip, embs, top_k=topk_rerank
                )
            else:
                final_results_data = sorted(grouped, key=lambda x: x.get('score', 0), reverse=True)[:topk_rerank]
        else:
            final_results_data = sorted(grouped, key=lambda x: x.get('score', 0), reverse=True)[:topk_rerank]


        # 5. Финальное форматирование вывода
        results = []
        for g in final_results_data:
            timestamp = g.get("start", g.get("time_sec", 0))

            result = {
                "timestamp_sec": round(timestamp, 3), 
                "timecode": MemoryLens.format_timecode(timestamp),
                "text": g.get("text"),
                "frame_path": g.get("frame_path"),
                "initial_score": round(g.get("score", 0), 4),
                "blip_caption": g.get("blip_caption"),
                "rerank_score": round(g.get("rerank_score", 0), 4) if blip and "rerank_score" in g else None,
                "video_id": g.get("video_id")
            }
            if result["rerank_score"] is None:
                del result["rerank_score"]
            
            results.append(result)

        return results


def free_memory(obj_name: str = ""):
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    if obj_name: print(f"✓ {obj_name} освобожден из памяти")