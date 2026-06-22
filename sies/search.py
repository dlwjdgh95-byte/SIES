"""검색 CLI — 베이스라인(순수 유사도) 또는 역전 재순위(--invert).

사용:
    uv run python -m sies.search "할머니에 대한 기억"            # 베이스라인
    uv run python -m sies.search "관성에 대하여" --invert        # 역전 재순위
"""
from __future__ import annotations

import argparse
import datetime as dt

from .embed import DEFAULT_MODEL, MODELS, get_embedder
from .rank import (
    DEFAULT_HALF_LIFE,
    DEFAULT_LAMBDA,
    rank_baseline,
    rank_gated,
    rank_inverted,
    rank_relz_fv,
)
from .retrieve import candidate_pool
from .store import connect


def _preview(text: str, n: int = 90) -> str:
    text = " ".join(text.split())
    return text[:n] + ("…" if len(text) > n else "")


def main() -> None:
    ap = argparse.ArgumentParser(description="SIES 의미검색")
    ap.add_argument("query")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--db", default="sies.db")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--invert", action="store_true", help="재순위 사용(--ranker 적용)")
    ap.add_argument("--ranker", choices=["relz_fv", "gated", "inverted"], default="relz_fv",
                    help="재순위 랭커: relz_fv(E, 기본) / gated(B) / inverted")
    ap.add_argument("--half-life", type=float, default=DEFAULT_HALF_LIFE,
                    help="활성도 반감기(일). 작을수록 옛 글을 더 띄움")
    ap.add_argument("--lam", type=float, default=DEFAULT_LAMBDA,
                    help="relz_fv의 망각 보너스 강도 λ")
    args = ap.parse_args()

    emb = get_embedder(args.model).load()
    qv = emb.encode([args.query], is_query=True)[0]

    conn = connect(args.db)
    pool = candidate_pool(conn, args.model, qv)
    conn.close()

    mode = args.ranker if args.invert else "베이스라인"
    print(f'질의: "{args.query}"  [{args.model} · {mode}]\n')

    if args.invert:
        rankers = {
            "relz_fv": lambda: rank_relz_fv(pool, dt.date.today(), args.half_life, lam=args.lam),
            "gated": lambda: rank_gated(pool, dt.date.today(), args.half_life),
            "inverted": lambda: rank_inverted(pool, dt.date.today(), args.half_life),
        }
        ranked = rankers[args.ranker]()[: args.k]
        for i, s in enumerate(ranked, 1):
            c = s.candidate
            extra = f"밴드 {s.band_weight:.2f}" if s.fv != s.fv else f"fv {s.fv:.2f}"
            print(f"{i:>2}. 점수 {s.score:.3f} (유사도 {s.similarity:.3f} · 활성도 {s.activity:.2f}"
                  f"[시{s.activity_time:.2f}/볼{s.activity_vol:.2f},V={s.volume}]"
                  f" · {extra}) [{c['title']}] {c['timestamp']}")
            print(f"    {_preview(c['text'])}")
    else:
        ranked = rank_baseline(pool)[: args.k]
        for i, s in enumerate(ranked, 1):
            c = s.candidate
            print(f"{i:>2}. 유사도 {s.similarity:.3f} [{c['title']}] {c['timestamp']}")
            print(f"    {_preview(c['text'])}")


if __name__ == "__main__":
    main()
