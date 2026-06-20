"""Phase 0 임베딩 벤치 — 같은 질의를 여러 모델에 돌려 눈으로 비교.

PLAN Phase 0: "감각어·함축 많은 네 문장에서 의미검색이 실제로 닿는지 눈으로 확인."
정답 라벨이 아직 없으므로 자동 점수가 아니라 나란히 보여주는 게 목적이다.

사용:
    uv run python -m sies.bench --models bge-m3 kure
    uv run python -m sies.bench --queries queries.txt
"""
from __future__ import annotations

import argparse

from .embed import DEFAULT_MODEL, MODELS, get_embedder
from .store import connect, search

# 감각어·함축이 많은 기본 질의 — 코퍼스(개인 에세이) 성격에 맞춤
DEFAULT_QUERIES = [
    "할머니와 관련된 냄새의 기억",
    "AI가 인간의 사고를 대체한다는 두려움",
    "어린 시절 반복했던 집안일",
    "금기를 넘는 순간의 망설임",
]


def _preview(text: str, n: int = 70) -> str:
    return " ".join(text.split())[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="SIES 임베딩 모델 비교 벤치")
    ap.add_argument("--models", nargs="+", default=[DEFAULT_MODEL], choices=list(MODELS))
    ap.add_argument("--db", default="sies.db")
    ap.add_argument("--queries", help="질의 1줄 1개 파일 (없으면 기본 질의)")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    if args.queries:
        with open(args.queries, encoding="utf-8") as f:
            queries = [ln.strip() for ln in f if ln.strip()]
    else:
        queries = DEFAULT_QUERIES

    embedders = {m: get_embedder(m).load() for m in args.models}
    conn = connect(args.db)

    for q in queries:
        print("=" * 72)
        print(f"질의: {q}")
        for m, emb in embedders.items():
            qv = emb.encode([q], is_query=True)[0]
            res = search(conn, m, qv, k=args.k)
            print(f"\n  [{m}]")
            if not res:
                print("    (결과 없음 — 이 모델로 먼저 인덱싱했나?)")
            for rank, r in enumerate(res, 1):
                print(f"    {rank}. ({r['distance']:.3f}) {r['title']} :: {_preview(r['text'])}")
        print()
    conn.close()


if __name__ == "__main__":
    main()
