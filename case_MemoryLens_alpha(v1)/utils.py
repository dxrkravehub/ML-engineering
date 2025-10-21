from faster_whisper import WhisperModel
import torch
import faiss
from typing import Any,List,Tuple
from sentence_transformers import SentenceTransformer
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np

class Embeddings:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.text_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=self.device, cache_folder="./models")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode_text(self, texts: List[str]) -> np.ndarray:
        vecs = self.text_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype("float32")

    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.clip_proc(images=images, return_tensors="pt")
        with torch.no_grad():
            img_emb = self.clip_model.get_image_features(**{k: v.to(self.device) for k, v in inputs.items()})
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        return img_emb.cpu().numpy().astype("float32")
    

class DualIndex:
    def __init__(self, dim_text: int, dim_image: int):
        self.faiss_text = faiss.IndexFlatIP(dim_text)   # cosine via dot since normalized
        self.faiss_image = faiss.IndexFlatIP(dim_image)
        self.text_meta: List[dict[str, Any]] = []
        self.image_meta: List[dict[str, Any]] = []

    def add_text(self, vectors: np.ndarray, metas: List[dict]):
        assert vectors.shape[0] == len(metas)
        self.faiss_text.add(vectors)
        self.text_meta.extend(metas)

    def add_image(self, vectors: np.ndarray, metas: List[dict]):
        assert vectors.shape[0] == len(metas)
        self.faiss_image.add(vectors)
        self.image_meta.extend(metas)

    def search_text(self, qvec: np.ndarray, topk: int = 10):
        D, I = self.faiss_text.search(qvec.astype("float32"), topk)
        results = []
        for i, score in zip(I[0], D[0]):
            if i == -1: continue
            meta = self.text_meta[i]
            results.append({**meta, "score": float(score)})
        return results

    def search_image(self, qvec: np.ndarray, topk: int = 10):
        D, I = self.faiss_image.search(qvec.astype("float32"), topk)
        results = []
        for i, score in zip(I[0], D[0]):
            if i == -1: continue
            meta = self.image_meta[i]
            results.append({**meta, "score": float(score)})
        return results


class BlipCaptioner:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

    def caption_paths(self, image_paths: List[str]) -> List[str]:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=30)
        return self.processor.batch_decode(out, skip_special_tokens=True)
