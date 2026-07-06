"""검색 CLI — 베이스라인(순수 유사도) 또는 역전 재순위(--invert).

사용:
    uv run python -m sies.search "할머니에 대한 기억"            # 베이스라인
    uv run python -m sies.search "관성에 대하여" --invert        # 역전 재순위
"""
from __future__ import annotations

import argparse
import datetime as dt

from .embed import DEFAULT_MODEL, MODELS
from .rank import DEFAULT_HALF_LIFE, rank_baseline, rank_inverted
from .retrieve import query_pool
from .util import preview


def main() -> None:
    ap = argparse.ArgumentParser(description="SIES 의미검색")
    ap.add_argument("query")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--db", default="sies.db")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--invert", action="store_true", help="역전 재순위 사용")
    ap.add_argument("--half-life", type=float, default=DEFAULT_HALF_LIFE,
                    help="활성도 반감기(일). 작을수록 옛 글을 더 띄움")
    args = ap.parse_args()

    pool = query_pool(args.db, args.model, args.query)

    mode = "역전" if args.invert else "베이스라인"
    print(f'질의: "{args.query}"  [{args.model} · {mode}]\n')

    if args.invert:
        ranked = rank_inverted(pool, dt.date.today(), args.half_life)[: args.k]
        for i, s in enumerate(ranked, 1):
            c = s.candidate
            print(f"{i:>2}. 점수 {s.score:.3f} (유사도 {s.similarity:.3f} · 활성도 {s.activity:.2f}"
                  f"[시{s.activity_time:.2f}/볼{s.activity_vol:.2f},V={s.volume}]"
                  f" · 밴드 {s.band_weight:.2f}) [{c['title']}] {c['timestamp']}")
            print(f"    {preview(c['text'], 90)}")
    else:
        ranked = rank_baseline(pool)[: args.k]
        for i, s in enumerate(ranked, 1):
            c = s.candidate
            print(f"{i:>2}. 유사도 {s.similarity:.3f} [{c['title']}] {c['timestamp']}")
            print(f"    {preview(c['text'], 90)}")


if __name__ == "__main__":
    main()
