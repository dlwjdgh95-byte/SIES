"""replay.py — 오프라인 점수식 재실행의 순수 로직."""
from sies.replay import (
    evaluate,
    f_baseline,
    f_forgotten,
    f_gated,
    judged_candidates,
)


def _rec(baseline, inversion, verdicts):
    return {"baseline": baseline, "inversion": inversion, "verdicts": verdicts}


def _c(cid, sim, act=None):
    d = {"id": cid, "similarity": sim}
    if act is not None:
        d["activity"] = act
    return d


def test_formulas_math():
    assert f_baseline(0.8, 0.9) == 0.8
    assert abs(f_forgotten(0.8, 0.25) - 0.6) < 1e-9
    # gated: 저활성(0.2<0.4)은 sim 그대로, 고활성(0.9)은 페널티
    assert f_gated(0.8, 0.2) == 0.8
    assert abs(f_gated(0.8, 0.9) - 0.8 * 0.1) < 1e-9


def test_judged_candidates_filters_and_merges():
    rec = _rec(
        baseline=[_c(1, 0.9, 0.8), _c(2, 0.7, 0.2)],
        inversion=[_c(3, 0.6, 0.3), _c(1, 0.9, 0.8)],
        verdicts={"1": "hit", "2": "miss", "3": "skip"},
    )
    cands = judged_candidates(rec)
    ids = sorted(d for d in [c.get("sim") for c in cands])
    # 1(hit),2(miss)만 — 3은 skip 제외. 1은 중복이지만 하나로.
    assert len(cands) == 2


def test_evaluate_topk_hits():
    # 저활성 고유사도가 적중, 고활성 고유사도가 헛것인 세션.
    rec = _rec(
        baseline=[_c(1, 0.9, 0.95), _c(2, 0.85, 0.2)],   # 1: 최근·헛것, 2: 잊힘·적중
        inversion=[_c(3, 0.5, 0.1)],                       # 3: 잊힘·적중
        verdicts={"1": "miss", "2": "hit", "3": "hit"},
    )
    res = evaluate([rec], {"baseline(sim)": f_baseline, "A": f_forgotten}, k=2)
    # baseline top2 = id1(0.9),id2(0.85) → 적중 1/2
    assert res["by_formula"]["baseline(sim)"]["micro"] == [1, 2]
    # A: 점수 1=0.9*.05=.045, 2=.85*.8=.68, 3=.5*.9=.45 → top2 = id2,id3 → 적중 2/2
    assert res["by_formula"]["A"]["micro"] == [2, 2]
