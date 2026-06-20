"""rank.py — 역전 재순위의 결정론적 산수. 제품의 뇌라 가장 촘촘히 검증."""
import datetime as dt
import math

import numpy as np

from sies.rank import (
    activity,
    bandpass_weight,
    cosine_from_l2,
    double_sigmoid,
    rank_baseline,
    rank_inverted,
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


def test_bandpass_weight_is_continuous():
    # 하드 마스크와 달리 0/1 이분이 아니라 연속값을 돌려준다
    sims = np.linspace(0, 1, 50)
    w = bandpass_weight(sims)
    assert np.any((w > 0.01) & (w < 0.99))  # 중간값 존재 = 절벽 아님


def test_bandpass_weight_single_candidate_passes():
    assert bandpass_weight(np.array([0.7])).tolist() == [1.0]


def test_bandpass_weight_empty():
    assert bandpass_weight(np.array([])).tolist() == []


# ── rank_inverted ────────────────────────────────────────────
def _cand(cos, ts):
    """코사인 유사도와 날짜로 후보 dict 생성 (distance 역산)."""
    distance = math.sqrt(max(0.0, 2.0 * (1.0 - cos)))
    return {"distance": distance, "timestamp": ts, "title": f"t{cos}", "text": "x"}


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


# ── rank_baseline ────────────────────────────────────────────
def test_baseline_sorts_by_similarity():
    cands = [_cand(0.5, "2025-01-01"), _cand(0.9, "2025-01-01"), _cand(0.7, "2025-01-01")]
    sims = [s.similarity for s in rank_baseline(cands)]
    assert sims == sorted(sims, reverse=True)
