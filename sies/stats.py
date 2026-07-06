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

from .ab import hit_rate
from .rank import LOW_ACTIVITY  # '잊힌' 판정 기준 — rank가 단일 소스
from .util import fmt_pct, read_log


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def aggregate(records: list[dict]) -> dict:
    """판정이 있는 세션만 집계."""
    macro = {"baseline": [], "inversion": []}
    micro = {"baseline": [0, 0], "inversion": [0, 0]}  # [hit, total]
    judged_sessions = 0
    # I-only: 역전에만 있고 베이스라인엔 없는 후보 — 도구의 진짜 본전(재현율)
    i_only = {"hit": 0, "judged": 0, "candidates": 0, "sessions_with_hit": 0}
    # 적중의 잊힘도(method 무관): 적중한 후보가 실제로 '잊혔던' 글인가
    hit_total = 0
    hit_acts: list[float] = []  # 활성도 기록이 있는 적중 후보의 활성도

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

        baseline_ids = {str(c["id"]) for c in r["baseline"]}
        session_i_only_hits = 0
        for c in r["inversion"]:
            cid = str(c["id"])
            if cid in baseline_ids:
                continue
            i_only["candidates"] += 1
            v = r["verdicts"].get(cid)
            if v == "hit":
                i_only["hit"] += 1
                i_only["judged"] += 1
                session_i_only_hits += 1
            elif v == "miss":
                i_only["judged"] += 1
        if session_i_only_hits:
            i_only["sessions_with_hit"] += 1

        # 적중의 잊힘도 — B/I 무관, 후보를 id로 합쳐 적중만 본다.
        act_by_id: dict[str, float] = {}
        for c in r["baseline"] + r["inversion"]:
            if c.get("activity") is not None:
                act_by_id.setdefault(str(c["id"]), c["activity"])
        union_ids = {str(c["id"]) for c in r["baseline"] + r["inversion"]}
        for cid in union_ids:
            if r["verdicts"].get(cid) == "hit":
                hit_total += 1
                a = act_by_id.get(cid)
                if a is not None:
                    hit_acts.append(a)

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    def ratio(pair):
        return pair[0] / pair[1] if pair[1] else None

    return {
        "sessions_total": len(records),
        "sessions_judged": judged_sessions,
        "macro": {m: mean(macro[m]) for m in macro},
        "micro": {m: ratio(micro[m]) for m in micro},
        "i_only": i_only,
        "hits_forgotten": {
            "hit_total": hit_total,
            "with_activity": len(hit_acts),
            "low_activity": sum(1 for a in hit_acts if a < LOW_ACTIVITY),
            "activity_median": _median(hit_acts),
            "low_threshold": LOW_ACTIVITY,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SIES A/B 로그 집계")
    ap.add_argument("--log", default="search_log.jsonl")
    args = ap.parse_args()

    try:
        records = read_log(args.log)
    except FileNotFoundError:
        print(f"로그 없음: {args.log}. 먼저 `python -m sies.ab \"질의\" --judge` 로 쌓아라.")
        return

    agg = aggregate(records)
    print(f"세션: 총 {agg['sessions_total']}개 / 판정된 것 {agg['sessions_judged']}개\n")
    print(f"{'':12} {'베이스라인':>10} {'역전':>10}")
    print(f"{'매크로 적중률':12} {fmt_pct(agg['macro']['baseline']):>10} {fmt_pct(agg['macro']['inversion']):>10}")
    print(f"{'마이크로 적중률':12} {fmt_pct(agg['micro']['baseline']):>10} {fmt_pct(agg['micro']['inversion']):>10}")

    io = agg["i_only"]
    io_rate = io["hit"] / io["judged"] if io["judged"] else None
    print(
        f"\n역전 단독(I-only) 적중: {io['hit']}개"
        f" / 판정 {io['judged']}개 (적중률 {fmt_pct(io_rate)})"
        f" · 후보 {io['candidates']}개 · 건진 세션 {io['sessions_with_hit']}개"
    )
    print("  └ 베이스라인이 절대 안 올렸을 후보를 역전이 건져 적중시킨 수 — 도구의 본전.")

    hf = agg["hits_forgotten"]
    med = f"{hf['activity_median']:.2f}" if hf["activity_median"] is not None else "—"
    print(
        f"\n적중의 잊힘도(method 무관): 적중 {hf['hit_total']}개"
        f" · 활성도 기록 {hf['with_activity']}개"
        f" · 저활성(<{hf['low_threshold']}) 적중 {hf['low_activity']}개"
        f" · 활성도 중앙값 {med}"
    )
    print("  └ 잊힌 통찰을 실제로 몇 개 건졌나 — B/I 무관(정확한 베이스라인 적중도 포함). 활성도↓ = 더 잊힘.")

    bi, ii = agg["macro"]["baseline"], agg["macro"]["inversion"]
    if agg["sessions_judged"] == 0:
        print("\n판정된 세션이 없다. --judge 로 적중/헛것을 표시해야 킬 테스트가 돈다.")
    elif bi is not None and ii is not None:
        verdict = "역전 우세 ✓" if ii > bi else ("동률" if ii == bi else "베이스라인 우세 ✗")
        note = ""
        if ii is not None and bi is not None and ii <= bi and io["hit"] > 0:
            note = f" — 단, 역전 단독 적중 {io['hit']}개(재현율 본전)는 따로 본다"
        print(f"\n판정(매크로): {verdict}{note}  (한 달치 누적 후 최종 판단)")


if __name__ == "__main__":
    main()
