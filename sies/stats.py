"""A/B 로그 집계 — 킬 테스트 판정.

각 세션의 적중률을 모아 베이스라인 대비 역전이 이기는지 본다.
- 매크로: 세션별 적중률의 평균
- 마이크로: 전체 후보를 합쳐 hit/(hit+miss)

사용:
    uv run python -m sies.stats
    uv run python -m sies.stats --log search_log.jsonl
"""
from __future__ import annotations

import argparse
import json

from .ab import hit_rate


def aggregate(records: list[dict]) -> dict:
    """판정이 있는 세션만 집계."""
    macro = {"baseline": [], "inversion": []}
    micro = {"baseline": [0, 0], "inversion": [0, 0]}  # [hit, total]
    judged_sessions = 0

    for r in records:
        br, ir = hit_rate(r, "baseline"), hit_rate(r, "inversion")
        if br is None and ir is None:
            continue
        judged_sessions += 1
        for method, rate in (("baseline", br), ("inversion", ir)):
            if rate is not None:
                macro[method].append(rate)
            for c in r[method]:
                v = r["verdicts"].get(str(c["id"]))
                if v == "hit":
                    micro[method][0] += 1
                    micro[method][1] += 1
                elif v == "miss":
                    micro[method][1] += 1

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    def ratio(pair):
        return pair[0] / pair[1] if pair[1] else None

    return {
        "sessions_total": len(records),
        "sessions_judged": judged_sessions,
        "macro": {m: mean(macro[m]) for m in macro},
        "micro": {m: ratio(micro[m]) for m in micro},
    }


def _fmt(x) -> str:
    return f"{x:.1%}" if isinstance(x, float) else "—"


def main() -> None:
    ap = argparse.ArgumentParser(description="SIES A/B 로그 집계")
    ap.add_argument("--log", default="search_log.jsonl")
    args = ap.parse_args()

    try:
        with open(args.log, encoding="utf-8") as f:
            records = [json.loads(ln) for ln in f if ln.strip()]
    except FileNotFoundError:
        print(f"로그 없음: {args.log}. 먼저 `python -m sies.ab \"질의\" --judge` 로 쌓아라.")
        return

    agg = aggregate(records)
    print(f"세션: 총 {agg['sessions_total']}개 / 판정된 것 {agg['sessions_judged']}개\n")
    print(f"{'':12} {'베이스라인':>10} {'역전':>10}")
    print(f"{'매크로 적중률':12} {_fmt(agg['macro']['baseline']):>10} {_fmt(agg['macro']['inversion']):>10}")
    print(f"{'마이크로 적중률':12} {_fmt(agg['micro']['baseline']):>10} {_fmt(agg['micro']['inversion']):>10}")

    bi, ii = agg["macro"]["baseline"], agg["macro"]["inversion"]
    if agg["sessions_judged"] == 0:
        print("\n판정된 세션이 없다. --judge 로 적중/헛것을 표시해야 킬 테스트가 돈다.")
    elif bi is not None and ii is not None:
        verdict = "역전 우세 ✓" if ii > bi else ("동률" if ii == bi else "베이스라인 우세 ✗")
        print(f"\n판정(매크로): {verdict}  (한 달치 누적 후 최종 판단)")


if __name__ == "__main__":
    main()
