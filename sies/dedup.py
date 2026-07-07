"""인덱싱 단계 근접 중복 제거 — 임베딩 코사인 ≥ 임계(0.97)면 하나만 남긴다.

정본은 포맷 우선순위(md/txt > docx > pdf > hwp)로 메타 좋은 쪽.
주제만 비슷한 글(0.7~0.9)은 보존 — 테마 반복은 신호다. 학습·LLM 없음(원칙 2).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .chunk import Chunk

DEFAULT_DEDUP_THRESHOLD = 0.97

# 같은 내용이면 메타가 풍부한 포맷을 정본으로 남긴다(숫자 작을수록 우선 보존).
_FMT_PRIORITY = {".md": 0, ".markdown": 0, ".txt": 0, ".docx": 1, ".pdf": 2, ".hwp": 3}


def _priority(doc_path: str) -> int:
    return _FMT_PRIORITY.get(Path(doc_path).suffix.lower(), 9)


def dedup_chunks(
    chunks: list[Chunk], vecs: np.ndarray, threshold: float = DEFAULT_DEDUP_THRESHOLD
) -> tuple[list[Chunk], np.ndarray, list[tuple[Chunk, Chunk, float]]]:
    """근접 중복을 제거. (남긴 청크, 남긴 벡터, [(버린 청크, 정본 청크, 코사인)]) 반환.

    그리디: 포맷 우선순위 → 경로 → chunk_index 순으로 훑으며, 이미 남긴 청크와
    코사인 ≥ threshold면 버린다. 정본은 먼저 훑힌(우선순위 높은) 쪽.
    반환 청크/벡터는 원래 순서를 유지한다(저장 안정성).
    """
    n = len(chunks)
    if n == 0:
        return [], vecs, []
    V = np.asarray(vecs)  # 원본 dtype(float32) 유지 — 저장 시 차원 어긋남 방지
    Vf = V.astype(np.float64)  # 유사도 계산용 사본만 float64
    Vn = Vf / np.clip(np.linalg.norm(Vf, axis=1, keepdims=True), 1e-12, None)

    order = sorted(
        range(n),
        key=lambda i: (_priority(chunks[i].doc_path), chunks[i].doc_path, chunks[i].chunk_index),
    )
    kept_idx: list[int] = []
    kept_vn: list[np.ndarray] = []
    dropped: list[tuple[Chunk, Chunk, float]] = []
    for i in order:
        if kept_vn:
            sims = np.asarray(kept_vn) @ Vn[i]
            j = int(np.argmax(sims))
            if sims[j] >= threshold:
                dropped.append((chunks[i], chunks[kept_idx[j]], float(sims[j])))
                continue
        kept_idx.append(i)
        kept_vn.append(Vn[i])

    keep_sorted = sorted(kept_idx)
    return [chunks[i] for i in keep_sorted], V[keep_sorted], dropped
