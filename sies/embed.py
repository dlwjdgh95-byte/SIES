"""임베딩 백엔드 — 로컬 sentence-transformers 레지스트리. 순수 벡터화 단계(원칙 2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 한국어 의미검색 후보 모델. 키는 짧은 별칭.
# 기본은 KURE — BGE-m3를 한국어로 추가 학습한 모델이라 한국어 뉘앙스에서 우위(Phase 0 벤치).
DEFAULT_MODEL = "kure"
MODELS = {
    # 한국어 특화(KURE, 고려대). BGE-m3 기반 한국어 fine-tune. 1024차원. ★ 기본
    "kure": "nlpai-lab/KURE-v1",
    # 다국어 강자(BAAI), 한국어 준수. 비교용 베이스라인. 1024차원.
    "bge-m3": "BAAI/bge-m3",
    # 가볍고 빠른 다국어 베이스라인. 384차원.
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}


@dataclass
class Embedder:
    alias: str
    model_name: str
    _model: object = None
    dim: int = 0

    def load(self) -> "Embedder":
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        return self

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if self._model is None:
            self.load()
        # 정규화하면 코사인 = 내적 → sqlite-vec L2/내적과 호환 단순화
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)


def get_embedder(alias: str) -> Embedder:
    if alias not in MODELS:
        raise KeyError(f"알 수 없는 모델 별칭: {alias}. 가능: {list(MODELS)}")
    return Embedder(alias=alias, model_name=MODELS[alias])
