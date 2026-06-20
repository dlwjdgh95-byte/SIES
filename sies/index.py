"""인덱싱 CLI — corpus/ 를 읽어 청킹·임베딩·저장.

사용:
    uv run python -m sies.index --model bge-m3
    uv run python -m sies.index --model kure --corpus corpus --db sies.db
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .chunk import chunk_document
from .corpus import iter_corpus
from .embed import DEFAULT_MODEL, MODELS, get_embedder
from .store import (
    connect,
    ensure_vec_table,
    init_schema,
    store_embeddings,
    upsert_chunks,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="SIES 코퍼스 인덱싱")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--db", default="sies.db")
    args = ap.parse_args()

    docs = iter_corpus(Path(args.corpus))
    chunks = [c for d in docs for c in chunk_document(d)]
    print(f"{len(docs)}개 문서 → {len(chunks)}개 청크")
    if not chunks:
        print("청크 없음. corpus/ 에 글을 넣어라.")
        return

    print(f"모델 로딩: {args.model} ({MODELS[args.model]}) …")
    emb = get_embedder(args.model).load()
    print(f"  차원={emb.dim}")

    vecs = emb.encode([c.text for c in chunks])

    conn = connect(args.db)
    init_schema(conn)
    ensure_vec_table(conn, args.model, emb.dim)
    ids = upsert_chunks(conn, chunks)
    store_embeddings(conn, args.model, ids, vecs)
    conn.close()
    print(f"저장 완료 → {args.db}  (vec_{args.model.replace('-', '_')})")


if __name__ == "__main__":
    main()
