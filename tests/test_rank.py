"""rank.py — 역전 재순위의 결정론적 산수. 제품의 뇌라 가장 촘촘히 검증."""
import datetime as dt
import math

import numpy as np

from sies.rank import (
    activity,
    bandpass_weight,
    cosine_from_l2,
    double_sigmoid,
    forgetting_value,
    newer_counts,
    rank_baseline,
    rank_gated,
    rank_inverted,
    rank_relz_fv,
    volume_activity,
)

NOW = dt.date(2026, 6, 20)


# ── cosine_from_l2 ────────────────────────────────────────────
def test_cosine_identical_vectors():
    assert cosine_from_l2(0.0) == 1.0


def test_cosine_orthogonal():
    # 정규화 벡터 직교 → L2 = sqrt(2), cos = 0
    assert cosine_from_l2(math.sqrt(2)) == 0.0


def test_cosine_half():
    assert cosine_from_l2(1.0) == 0.5


def test_cosine_clamps_negative():
    # 거리 2(반대 방향) → cos = -1 이지만 [0,1]로 클램프
    assert cosine_from_l2(2.0) == 0.0


# ── activity ─────────────────────────────────────────────────
def test_activity_today_is_one():
    assert activity(NOW, NOW) == 1.0


def test_activity_one_half_life():
    ts = NOW - dt.timedelta(days=365)
    assert activity(ts, NOW, half_life_days=365) == 0.5


def test_activity_two_half_lives():
    ts = NOW - dt.timedelta(days=730)
    assert activity(ts, NOW, half_life_days=365) == 0.25


def test_activity_missing_is_neutral():
    assert activity(None, NOW) == 0.5


def test_activity_future_clamped():
    # 미래 날짜(나이 음수)는 0으로 클램프 → 활성도 1
    assert activity(NOW + dt.timedelta(days=30), NOW) == 1.0


def test_activity_older_is_lower():
    a_old = activity(NOW - dt.timedelta(days=540), NOW)
    a_new = activity(NOW - dt.timedelta(days=30), NOW)
    assert a_old < a_new


# ── double_sigmoid (소프트 밴드 곡선) ─────────────────────────
def test_double_sigmoid_high_in_middle():
    # 폭이 넓으면(1/k보다 훨씬) 중앙은 ~1
    assert double_sigmoid(0.5, 0.4, 0.6, k=50)[()] > 0.95


def test_double_sigmoid_half_at_boundaries():
    # 경계에서 ~0.5 (한쪽 시그모이드가 0.5, 다른 쪽 ~0)
    assert abs(double_sigmoid(0.4, 0.4, 0.6, k=50)[()] - 0.5) < 0.01
    assert abs(double_sigmoid(0.6, 0.4, 0.6, k=50)[()] - 0.5) < 0.01


def test_double_sigmoid_near_zero_outside():
    assert double_sigmoid(0.1, 0.4, 0.6, k=50)[()] < 0.01   # 하위(잡음)
    assert double_sigmoid(0.9, 0.4, 0.6, k=50)[()] < 0.01   # 상위(뻔함)


def test_double_sigmoid_no_cliff():
    # 경계 양옆 0.001 차이가 통째 증발(1→0)이 아니라 미세 변화여야 한다
    a = double_sigmoid(0.399, 0.4, 0.6, k=50)[()]
    b = double_sigmoid(0.401, 0.4, 0.6, k=50)[()]
    assert abs(a - b) < 0.05
    assert 0.4 < a < 0.6 and 0.4 < b < 0.6  # 둘 다 ~0.5 부근, 누구도 0이 아님


def test_double_sigmoid_no_overflow_on_extremes():
    # 아주 먼 값에도 inf/경고 없이 [0,1)
    w = double_sigmoid(np.array([-10.0, 10.0]), 0.4, 0.6, k=50)
    assert np.all((w >= 0) & (w < 1))


# ── bandpass_weight (퍼센타일 배선) ───────────────────────────
def test_bandpass_weight_drops_top_and_bottom():
    sims = np.arange(0, 101, 10.0)  # 0,10,...,100
    w = bandpass_weight(sims, lo=60, hi=85)  # L=60, U=85
    assert w[sims.tolist().index(0)] < 0.01     # 최하위 눌림
    assert w[sims.tolist().index(100)] < 0.01   # 최상위 눌림
    assert w[sims.tolist().index(70)] > 0.4     # 밴드 안 살아남음


def test_bandpass_weight_peak_normalized_to_one():
    # 밴드 중앙 부근 후보는 정규화로 가중치 ~1, 절대 1을 넘지 않는다
    sims = np.array([0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0])
    w = bandpass_weight(sims)
    assert w.max() <= 1.0 + 1e-9
    assert w.max() > 0.95


def test_bandpass_weight_normalize_off_keeps_raw_peak():
    # 좁은 밴드 + k=50이면 정규화 끄면 피크가 1에 못 미친다(겹침)
    sims = np.array([0.40, 0.444, 0.47, 0.497, 0.55])
    raw = bandpass_weight(sims, normalize=False)
    norm = bandpass_weight(sims, normalize=True)
    assert raw.max() < 0.95          # 정규화 전엔 피크<1
    assert abs(norm.max() - 1.0) < 0.05  # 정규화 후 피크~1


def test_bandpass_weight_is_continuous():
    # 하드 마스크와 달리 0/1 이분이 아니라 연속값을 돌려준다
    sims = np.linspace(0, 1, 50)
    w = bandpass_weight(sims)
    assert np.any((w > 0.01) & (w < 0.99))  # 중간값 존재 = 절벽 아님


def test_bandpass_weight_single_candidate_passes():
    assert bandpass_weight(np.array([0.7])).tolist() == [1.0]


def test_bandpass_weight_empty():
    assert bandpass_weight(np.array([])).tolist() == []


# ── volume_activity / newer_counts ───────────────────────────
def test_volume_activity_no_burial_is_one():
    assert volume_activity(0) == 1.0


def test_volume_activity_one_half_life():
    assert volume_activity(50, half_life=50) == 0.5


def test_volume_activity_two_half_lives():
    assert volume_activity(100, half_life=50) == 0.25


def test_newer_counts_basic():
    cands = [
        _cand(0.7, "2022-01-01", doc="c"),
        _cand(0.7, "2020-01-01", doc="a"),
        _cand(0.7, "2021-01-01", doc="b"),
    ]
    counts = newer_counts(cands)
    assert counts["a"] == 2  # b,c 가 더 새로움
    assert counts["b"] == 1  # c
    assert counts["c"] == 0  # 가장 새로움


def test_newer_counts_dedupes_chunks_per_doc():
    # 같은 문서의 청크 2개는 한 문서로 센다
    cands = [
        _cand(0.7, "2020-01-01", doc="a"),
        _cand(0.6, "2020-01-01", doc="a"),
        _cand(0.7, "2021-01-01", doc="b"),
    ]
    counts = newer_counts(cands)
    assert counts["a"] == 1  # b 하나만 더 새로움 (a의 두 청크는 한 문서)


def test_newer_counts_missing_date_not_buried():
    cands = [_cand(0.7, "", doc="a"), _cand(0.7, "2021-01-01", doc="b")]
    counts = newer_counts(cands)
    assert counts["a"] == 0  # 날짜 없으면 안 덮인 것으로


# ── rank_inverted ────────────────────────────────────────────
def _cand(cos, ts, doc=None):
    """코사인 유사도와 날짜로 후보 dict 생성 (distance 역산)."""
    distance = math.sqrt(max(0.0, 2.0 * (1.0 - cos)))
    return {
        "distance": distance,
        "timestamp": ts,
        "title": f"t{cos}",
        "doc_path": doc or f"{cos}@{ts}",
        "text": "x",
    }


def test_inverted_extremes_get_near_zero_score():
    # 명확한 상위(뻔함)/하위(잡음)는 밴드 밖 → 가중치~0 → 점수~0
    cands = [_cand(c, "2025-01-01") for c in (0.95, 0.75, 0.70, 0.68, 0.40)]
    ranked = rank_inverted(cands, NOW)
    by_cos = {round(s.similarity, 2): s for s in ranked}
    assert by_cos[0.95].score < 0.02   # 최상위
    assert by_cos[0.40].score < 0.02   # 최하위
    # 밴드 안 후보는 살아있다
    assert max(s.score for s in ranked) > 0.05


def test_inverted_prefers_older_within_band():
    # 유사도 동일, 날짜만 다른 두 후보가 밴드 안일 때 더 오래된 글이 높은 점수
    cands = [
        _cand(0.70, "2026-06-01"),  # 최근
        _cand(0.70, "2024-06-01"),  # 오래됨
        _cand(0.95, "2025-01-01"),  # 상위(밴드 밖 유도)
        _cand(0.40, "2025-01-01"),  # 하위(밴드 밖 유도)
    ]
    ranked = rank_inverted(cands, NOW, half_life_days=365)
    # 전체에서 가장 점수 높은 건 밴드 안 + 더 오래된(2024) 글
    top = ranked[0]
    assert top.candidate["timestamp"] == "2024-06-01"


def test_inverted_sorted_descending():
    cands = [_cand(c, "2025-01-01") for c in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)]
    scores = [s.score for s in rank_inverted(cands, NOW)]
    assert scores == sorted(scores, reverse=True)


def test_inverted_empty():
    assert rank_inverted([], NOW) == []


def test_inverted_volume_buries_recent_burst():
    # 몰아쓰기: 6/01~6/10 열 편 — 시간으론 다 최근이지만 볼륨으론 d1이 9편에 덮임
    cands = [_cand(0.7, f"2026-06-{i:02d}", doc=f"d{i}") for i in range(1, 11)]
    ranked = rank_inverted(cands, NOW, volume_half_life=2)
    by = {s.candidate["doc_path"]: s for s in ranked}
    # 시간 활성도는 다 높음(최근)
    assert by["d1"].activity_time > 0.9
    # 그러나 d1은 뒤에 9편 → 볼륨 활성도 낮음 → 최종 = min 이 볼륨 채택
    assert by["d1"].activity_vol < 0.1
    assert by["d1"].activity == min(by["d1"].activity_time, by["d1"].activity_vol)
    assert by["d1"].volume == 9
    # 가장 최신 d10은 안 덮임 → 볼륨 활성도 1, 최종은 시간이 결정
    assert by["d10"].activity_vol == 1.0
    assert by["d10"].activity == by["d10"].activity_time


def test_inverted_old_but_newest_uses_time_axis():
    # 오래됐지만 그 뒤로 아무것도 안 쓴 글: 볼륨=0(A_vol=1)이라 시간축이 최종을 결정
    cands = [
        _cand(0.7, "2020-01-01", doc="old_newest"),  # 가장 오래됐고 = 가장 최신(뒤에 없음)...
        _cand(0.7, "2019-01-01", doc="older"),
    ]
    ranked = rank_inverted(cands, NOW, half_life_days=365, volume_half_life=2)
    by = {s.candidate["doc_path"]: s for s in ranked}
    # old_newest: 뒤에 덮인 게 없음 → 볼륨활성도 1, 최종 = 시간활성도(오래돼 낮음)
    assert by["old_newest"].volume == 0
    assert by["old_newest"].activity_vol == 1.0
    assert by["old_newest"].activity == by["old_newest"].activity_time
    assert by["old_newest"].activity < 0.1  # 6년 전 → 시간상 충분히 잊힘


# ── rank_baseline ────────────────────────────────────────────
def test_baseline_sorts_by_similarity():
    cands = [_cand(0.5, "2025-01-01"), _cand(0.9, "2025-01-01"), _cand(0.7, "2025-01-01")]
    sims = [s.similarity for s in rank_baseline(cands)]
    assert sims == sorted(sims, reverse=True)


# ── rank_gated (B) ───────────────────────────────────────────
def test_gated_low_activity_keeps_pure_similarity():
    # 저활성(오래됨)이면 유사도 그대로 → 관련 깊은 옛 글이 위로.
    cands = [
        _cand(0.55, "2024-01-01"),  # 잊힘 + 고유사도
        _cand(0.70, "2024-01-01"),  # 잊힘 + 더 고유사도
    ]
    ranked = rank_gated(cands, NOW, half_life_days=365)
    assert ranked[0].similarity > ranked[1].similarity     # 잊힌 것끼리는 유사도 순
    assert abs(ranked[0].score - ranked[0].similarity) < 1e-9  # 저활성은 페널티 없음
    assert all(s.band_weight == 1.0 for s in ranked)       # 밴드패스 없음


def test_gated_penalizes_recent():
    # 최근(고활성) 글은 유사도가 높아도 ×(1−활성도)로 눌린다.
    cands = [
        _cand(0.80, "2026-06-19"),  # 어제 = 고활성 → 큰 페널티
        _cand(0.60, "2024-01-01"),  # 잊힘 → 유사도 그대로
    ]
    ranked = rank_gated(cands, NOW, half_life_days=365)
    assert ranked[0].candidate["timestamp"] == "2024-01-01"  # 잊힌 글이 최근 고유사도를 이김
    recent = next(s for s in ranked if s.candidate["timestamp"] == "2026-06-19")
    assert recent.score < recent.similarity        # 최근은 페널티 받음


def test_gated_empty():
    assert rank_gated([], NOW) == []


# ── forgetting_value (망각가치 밴드) ──────────────────────────
def test_fv_peak_at_035():
    assert forgetting_value(0.35) == 1.0


def test_fv_recent_excluded():
    # 거의 최신(A≥0.9)은 0 — '완전 최신만 배제'
    assert forgetting_value(0.90) == 0.0
    assert forgetting_value(1.0) == 0.0


def test_fv_old_is_gentle_not_zero():
    # 가장 잊힌 글(A→0)도 0이 아니라 OLD_FLOOR(0.6)까지만 — '옛것엔 관대'
    assert forgetting_value(0.0) == 0.6
    assert forgetting_value(0.18) > 0.5


def test_fv_asymmetric_left_gentler_than_right():
    # 피크에서 같은 거리(±0.2)면 옛것 쪽(좌)이 최신 쪽(우)보다 가중치가 높다
    assert forgetting_value(0.35 - 0.20) > forgetting_value(0.35 + 0.20)


# ── rank_relz_fv (전략 E) ────────────────────────────────────
def test_relz_fv_prefers_mid_forgotten_over_recent():
    # 유사도 동일 → rel_z 0 → fv가 결정. 중간-망각(피크)이 최신을 이긴다.
    mid = NOW - dt.timedelta(days=552)   # A_time ≈ 0.35
    cands = [
        _cand(0.70, NOW.isoformat(), doc="recent"),      # 최신 → fv ~0
        _cand(0.70, mid.isoformat(), doc="mid"),         # 중간-망각 → fv ~1
    ]
    ranked = rank_relz_fv(cands, NOW, half_life_days=365)
    assert ranked[0].candidate["doc_path"] == "mid"
    assert ranked[0].fv > ranked[1].fv


def test_relz_fv_relevance_dominates_among_forgotten():
    # 같은(잊힘) 날짜면 fv 동일 → rel_z(관련성)가 줄세운다.
    old = (NOW - dt.timedelta(days=552)).isoformat()
    cands = [
        _cand(0.50, old, doc="lo"),
        _cand(0.70, old, doc="hi"),
        _cand(0.60, old, doc="mid"),
    ]
    ranked = rank_relz_fv(cands, NOW, half_life_days=365)
    assert ranked[0].candidate["doc_path"] == "hi"
    assert [s.candidate["doc_path"] for s in ranked] == ["hi", "mid", "lo"]


def test_relz_fv_records_fv_and_no_band():
    cands = [_cand(c, "2025-01-01") for c in (0.5, 0.7)]
    ranked = rank_relz_fv(cands, NOW)
    assert all(s.fv == s.fv for s in ranked)          # fv 채워짐(NaN 아님)
    assert all(s.band_weight == 1.0 for s in ranked)  # 유사도 밴드패스 없음


def test_relz_fv_sorted_descending():
    cands = [_cand(c, "2024-06-01") for c in (0.9, 0.8, 0.7, 0.6, 0.5)]
    scores = [s.score for s in rank_relz_fv(cands, NOW)]
    assert scores == sorted(scores, reverse=True)


def test_relz_fv_empty():
    assert rank_relz_fv([], NOW) == []
