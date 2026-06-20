"""검색 CLI — 순수 의미검색(베이스라인). Phase 0의 "그럴듯하게 작동" 확인용.

Phase 1에서 여기 위에 역전 재순위(활성도·밴드패스)를 얹는다. 지금은 베이스라인만.

사용:
    uv run python -m sies.search "할머니에 대한 기억" --model bge-m3
"""
from __future__ import annotations

import argparse

from .embed import DEFAULT_MODEL, MODELS, get_embedder
from .store import connect, search


def _preview(text: str, n: int = 90) -> str:
    text = " ".join(text.split())
    return text[:n] + ("…" if len(text) > n else "")


def main() -> None:
    ap = argparse.ArgumentParser(description="SIES 의미검색 (베이스라인)")
    ap.add_argument("query")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--db", default="sies.db")
    ap.add_argument("-k", type=int, default=8)
    args = ap.parse_args()

    emb = get_embedder(args.model).load()
    qv = emb.encode([args.query], is_query=True)[0]

    conn = connect(args.db)
    results = search(conn, args.model, qv, k=args.k)
    conn.close()

    print(f'질의: "{args.query}"  [{args.model}]\n')
    for rank, r in enumerate(results, 1):
        print(f"{rank:>2}. ({r['distance']:.3f}) [{r['title']}] {r['timestamp']}")
        print(f"    {_preview(r['text'])}")


if __name__ == "__main__":
    main()
