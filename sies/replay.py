"""오프라인 재실행 — 새 판정 없이, 기존 로그로 점수식을 비교한다.

각 세션의 *판정된 후보 합집합*을 여러 점수식으로 재랭킹해 top-k 적중률을 비교한다.
'어떤 점수식이 적중을 위로 더 올리나'를 같은 후보집합 위에서 본다.

한계:
- 후보별 (유사도, 활성도)만 쓰는 식만 정확히 재현(밴드패스는 풀 분포가 필요한데
  로그엔 top-5만 있음). 그래서 밴드패스 제거/게이팅 계열을 비교한다.
- 합집합은 '그때 보여준 후보'라, 안 보여준 후보의 잠재 적중은 알 수 없다.
- 표본이 작으면(세션 몇 개) 경향만. 세션이 쌓일수록 신뢰도가 오른다.

사용:
    uv run python -m sies.replay
    uv run python -m sies.replay --k 5 --log search_log.jsonl
"""
from __future__ import annotations

import argparse

from .rank import gated_score  # B의 점수식 — rank가 단일 소스
from .util import fmt_pct, read_log


# 점수식: (유사도, 활성도) -> 점수. 활성도↓ = 더 잊힘.
def f_baseline(sim: float, act: float) -> float:
    return sim


def f_forgotten(sim: float, act: float) -> float:        # A: 밴드패스 제거
    return sim * (1.0 - act)


def f_gated(sim: float, act: float) -> float:            # B: 저활성은 통과, 최근만 페널티
    return gated_score(sim, act)


def f_mild(sim: float, act: float) -> float:             # C: 약한 잊힘 보상
    return sim * (1.0 - 0.5 * act)


FORMULAS = {
    "baseline(sim)": f_baseline,
    "A: sim×(1−act)": f_forgotten,
    "B: gated": f_gated,
    "C: mild": f_mild,
}


def judged_candidates(record: dict) -> list[dict]:
    """판정된(hit/miss) 후보를 id로 합쳐 (sim, act, verdict) 리스트로."""
    by_id: dict[str, dict] = {}
    for c in record["baseline"] + record["inversion"]:
        cid = str(c["id"])
        d = by_id.setdefault(cid, {})
        if c.get("similarity") is not None:
            d["sim"] = c["similarity"]
        if c.get("activity") is not None:
            d["act"] = c["activity"]
        d["verdict"] = record["verdicts"].get(cid)
    return [
        d for d in by_id.values()
        if d.get("verdict") in ("hit", "miss") and "sim" in d and "act" in d
    ]


def evaluate(records: list[dict], formulas: dict, k: int = 5) -> dict:
    """식별로 top-k 적중률 집계. micro=전체 슬롯 합, macro=세션별 평균."""
    out = {name: {"micro": [0, 0], "macro": []} for name in formulas}
    sessions = 0
    for r in records:
        cands = judged_candidates(r)
        if not cands:
            continue
        sessions += 1
        for name, fn in formulas.items():
            ranked = sorted(cands, key=lambda d: fn(d["sim"], d["act"]), reverse=True)
            top = ranked[:k]
            hits = sum(1 for d in top if d["verdict"] == "hit")
            out[name]["micro"][0] += hits
            out[name]["micro"][1] += len(top)
            out[name]["macro"].append(hits / len(top))
    return {"sessions": sessions, "k": k, "by_formula": out}


def main() -> None:
    ap = argparse.ArgumentParser(description="SIES 오프라인 점수식 재실행 비교")
    ap.add_argument("--log", default="search_log.jsonl")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    try:
        records = read_log(args.log)
    except FileNotFoundError:
        print(f"로그 없음: {args.log}")
        return

    res = evaluate(records, FORMULAS, args.k)
    print(f"세션 {res['sessions']}개 · 판정 후보 합집합을 top-{res['k']} 재랭킹\n")
    print(f"{'점수식':<16}{'마이크로':>10}{'매크로':>10}")
    for name, agg in res["by_formula"].items():
        micro = agg["micro"][0] / agg["micro"][1] if agg["micro"][1] else None
        macro = sum(agg["macro"]) / len(agg["macro"]) if agg["macro"] else None
        print(f"{name:<16}{fmt_pct(micro):>10}{fmt_pct(macro):>10}")
    print("\n(같은 후보집합 위 랭킹 비교 — 표본 작으면 경향만. 밴드패스 계열은 풀 분포가 없어 제외.)")


if __name__ == "__main__":
    main()
