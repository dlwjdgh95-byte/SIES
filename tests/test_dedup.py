"""dedup.py — 근접 중복 제거의 순수 로직(임베딩 모델 없이 합성 벡터로)."""
import numpy as np

from sies.chunk import Chunk
from sies.dedup import dedup_chunks


def _c(path, idx, text):
    return Chunk(doc_path=path, source="s", title=path, timestamp="2025-01-01",
                 chunk_index=idx, text=text)


def test_drops_near_identical_keeps_richer_format():
    # 거의 같은 벡터: md(노션)와 hwp(raw) 중 md를 정본으로 남긴다.
    chunks = [_c("a.hwp", 0, "x"), _c("b.md", 0, "x 복사본")]
    vecs = np.array([[1.0, 0.0], [0.999, 0.0447]])  # cos ≈ 0.999
    kept, kv, dropped = dedup_chunks(chunks, vecs, 0.97)
    assert len(kept) == 1 and len(dropped) == 1
    assert kept[0].doc_path == "b.md"          # md > hwp 우선 보존
    assert dropped[0][0].doc_path == "a.hwp"   # 버려진 쪽
    assert dropped[0][2] > 0.97                # 코사인


def test_keeps_distinct_topics():
    # 주제만 다른(직교) 벡터는 보존 — 테마 반복은 신호.
    chunks = [_c("a.md", 0, "x"), _c("b.md", 0, "y")]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
    kept, kv, dropped = dedup_chunks(chunks, vecs, 0.97)
    assert len(kept) == 2 and not dropped


def test_preserves_original_order():
    chunks = [_c("a.md", 0, "p"), _c("b.md", 0, "q"), _c("c.md", 0, "r")]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    kept, kv, dropped = dedup_chunks(chunks, vecs, 0.97)
    assert [c.doc_path for c in kept] == ["a.md", "b.md", "c.md"]
    assert kv.shape == (3, 2)


def test_empty():
    kept, kv, dropped = dedup_chunks([], np.zeros((0, 2)), 0.97)
    assert kept == [] and dropped == []


def test_preserves_dtype():
    # float32 임베딩이 그대로 float32로 반환돼야 한다(저장 차원 어긋남 방지).
    chunks = [_c("a.md", 0, "x"), _c("b.md", 0, "y")]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    kept, kv, dropped = dedup_chunks(chunks, vecs, 0.97)
    assert kv.dtype == np.float32
