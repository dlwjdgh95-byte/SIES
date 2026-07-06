"""문단 단위 청킹 — 통찰은 글 전체가 아니라 문단에 박혀 있다.

빈 줄로 나누고, MIN_CHARS 미만 문단은 앞 문단에 병합.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Document
from .normalize import strip_leading_marker

MIN_CHARS = 40  # 이보다 짧은 문단은 독립 청크로 두지 않고 병합


@dataclass
class Chunk:
    doc_path: str
    source: str
    title: str
    timestamp: str       # ISO date 문자열
    chunk_index: int
    text: str


def split_paragraphs(body: str) -> list[str]:
    # 빈 줄(공백만 있는 줄 포함) 기준 분할
    raw = re.split(r"\n\s*\n", body.strip())
    paras = [p.strip() for p in raw if p.strip()]

    merged: list[str] = []
    for p in paras:
        if merged and len(p) < MIN_CHARS:
            merged[-1] = merged[-1] + "\n" + p
        else:
            merged.append(p)
    return merged


def chunk_document(doc: Document) -> list[Chunk]:
    ts = doc.timestamp.isoformat() if doc.timestamp else ""
    chunks = []
    for i, para in enumerate(split_paragraphs(doc.body)):
        chunks.append(
            Chunk(
                doc_path=str(doc.path),
                source=doc.source,
                title=doc.title,
                timestamp=ts,
                chunk_index=i,
                text=strip_leading_marker(para),
            )
        )
    return chunks


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from .corpus import iter_corpus

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("corpus")
    total = 0
    for doc in iter_corpus(root):
        cs = chunk_document(doc)
        total += len(cs)
        print(f"[{doc.title}] {len(cs)}개 문단")
    print(f"\n총 {total}개 청크")
