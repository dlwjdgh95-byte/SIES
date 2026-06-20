"""ab.py / stats.py — 세션 레코드, 적중률, 집계의 순수 로직."""
from types import SimpleNamespace

from sies.ab import candidate_key, hit_rate, make_record
from sies.stats import aggregate


def _scored(cid, title, sim, score=0.0, activity=0.5, band_weight=1.0):
    cand = {
        "id": cid,
        "title": title,
        "chunk_index": 0,
        "timestamp": "2025-01-01",
        "text": f"본문 {cid}",
    }
    return SimpleNamespace(
        candidate=cand, similarity=sim, score=score, activity=activity,
        activity_time=activity, activity_vol=1.0, volume=0, band_weight=band_weight,
    )


def test_candidate_key_is_str_id():
    assert candidate_key({"id": 7}) == "7"


def test_make_record_shape():
    base = [_scored(1, "A", 0.9), _scored(2, "B", 0.8)]
    inv = [_scored(3, "C", 0.6, score=0.3), _scored(1, "A", 0.9, score=0.2)]
    rec = make_record("질의", "kure", 365.0, 2, base, inv)
    assert rec["query"] == "질의"
    assert len(rec["baseline"]) == 2 and len(rec["inversion"]) == 2
    # 역전 후보에는 활성도/점수/밴드 필드가 붙는다
    assert "score" in rec["inversion"][0] and "activity" in rec["inversion"][0]
    # 베이스라인엔 안 붙는다
    assert "score" not in rec["baseline"][0]
    assert rec["verdicts"] == {}


def test_make_record_respects_k():
    base = [_scored(i, f"t{i}", 0.9 - i * 0.1) for i in range(5)]
    rec = make_record("q", "kure", 365.0, 3, base, base)
    assert len(rec["baseline"]) == 3


def test_hit_rate_none_without_verdicts():
    base = [_scored(1, "A", 0.9)]
    rec = make_record("q", "kure", 365.0, 1, base, base)
    assert hit_rate(rec, "baseline") is None


def test_hit_rate_counts_only_hit_miss():
    base = [_scored(1, "A", 0.9), _scored(2, "B", 0.8), _scored(3, "C", 0.7)]
    rec = make_record("q", "kure", 365.0, 3, base, [])
    rec["verdicts"] = {"1": "hit", "2": "miss", "3": "skip"}
    # skip은 분모에서 제외 → 1/2
    assert hit_rate(rec, "baseline") == 0.5


def test_hit_rate_separates_methods():
    base = [_scored(1, "A", 0.9)]
    inv = [_scored(2, "C", 0.6, score=0.3)]
    rec = make_record("q", "kure", 365.0, 1, base, inv)
    rec["verdicts"] = {"1": "miss", "2": "hit"}
    assert hit_rate(rec, "baseline") == 0.0
    assert hit_rate(rec, "inversion") == 1.0


def test_aggregate_skips_unjudged():
    base = [_scored(1, "A", 0.9)]
    rec = make_record("q", "kure", 365.0, 1, base, base)
    agg = aggregate([rec])
    assert agg["sessions_total"] == 1
    assert agg["sessions_judged"] == 0
    assert agg["macro"]["baseline"] is None


def test_aggregate_macro_and_micro():
    # 세션1: 베이스라인 1/2, 역전 2/2 / 세션2: 베이스라인 0/1, 역전 1/1
    base1 = [_scored(1, "A", 0.9), _scored(2, "B", 0.8)]
    inv1 = [_scored(3, "C", 0.6, score=0.3), _scored(4, "D", 0.6, score=0.3)]
    rec1 = make_record("q1", "kure", 365.0, 2, base1, inv1)
    rec1["verdicts"] = {"1": "hit", "2": "miss", "3": "hit", "4": "hit"}

    base2 = [_scored(5, "E", 0.9)]
    inv2 = [_scored(6, "F", 0.6, score=0.3)]
    rec2 = make_record("q2", "kure", 365.0, 1, base2, inv2)
    rec2["verdicts"] = {"5": "miss", "6": "hit"}

    agg = aggregate([rec1, rec2])
    assert agg["sessions_judged"] == 2
    # 매크로: 베이스라인 평균(0.5, 0.0)=0.25, 역전 평균(1.0,1.0)=1.0
    assert agg["macro"]["baseline"] == 0.25
    assert agg["macro"]["inversion"] == 1.0
    # 마이크로: 베이스라인 1/3, 역전 3/3
    assert abs(agg["micro"]["baseline"] - 1 / 3) < 1e-9
    assert agg["micro"]["inversion"] == 1.0


def test_aggregate_i_only_excludes_overlap():
    # 역전 후보 3,4 중 4는 베이스라인에도 있음(overlap) → I-only는 3뿐.
    # 3=hit → I-only 적중 1, 4는 베이스라인과 겹쳐 카운트 제외.
    base = [_scored(1, "A", 0.9), _scored(4, "D", 0.7)]
    inv = [_scored(3, "C", 0.6, score=0.3), _scored(4, "D", 0.7, score=0.2)]
    rec = make_record("q", "kure", 365.0, 2, base, inv)
    rec["verdicts"] = {"1": "miss", "3": "hit", "4": "hit"}
    io = aggregate([rec])["i_only"]
    assert io["candidates"] == 1          # 3만 I-only(4는 overlap)
    assert io["hit"] == 1
    assert io["judged"] == 1
    assert io["sessions_with_hit"] == 1


def test_aggregate_hits_forgotten():
    # 전체 풀(inverted)이 활성도를 제공 → 베이스라인 후보에도 활성도가 박힌다.
    base = [_scored(1, "A", 0.90, activity=0.80), _scored(2, "B", 0.85, activity=0.20)]
    inv = [
        _scored(1, "A", 0.90, score=0.1, activity=0.80),
        _scored(2, "B", 0.85, score=0.2, activity=0.20),
        _scored(3, "C", 0.60, score=0.3, activity=0.90),
    ]
    rec = make_record("q", "kure", 365.0, 3, base, inv)
    rec["verdicts"] = {"1": "hit", "2": "hit", "3": "miss"}
    hf = aggregate([rec])["hits_forgotten"]
    assert hf["hit_total"] == 2          # 1,2 적중(합집합)
    assert hf["with_activity"] == 2      # 둘 다 활성도 기록 있음(베이스라인 1 포함)
    assert hf["low_activity"] == 1       # 2만 저활성(0.20 < 0.4)
    assert hf["activity_median"] == 0.5  # [0.20, 0.80] 중앙값


def test_aggregate_i_only_skip_not_judged():
    # I-only 후보가 skip이면 judged 분모에서 빠지고 적중도 아님.
    base = [_scored(1, "A", 0.9)]
    inv = [_scored(2, "C", 0.6, score=0.3)]
    rec = make_record("q", "kure", 365.0, 1, base, inv)
    rec["verdicts"] = {"1": "hit", "2": "skip"}
    io = aggregate([rec])["i_only"]
    assert io["candidates"] == 1
    assert io["hit"] == 0
    assert io["judged"] == 0
    assert io["sessions_with_hit"] == 0
