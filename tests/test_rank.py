"""rank.py — 역전 재순위의 결정론적 산수. 제품의 뇌라 가장 촘촘히 검증."""
import datetime as dt
import math

import numpy as np

from sies.rank import (
    activity,
    bandpass_mask,
    cosine_from_l2,
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


# ── bandpass_mask ────────────────────────────────────────────
def test_bandpass_keeps_middle_band():
    sims = np.arange(0, 101, 10)  # 0,10,...,100
    mask = bandpass_mask(sims, lo=60, hi=85)
    kept = set(sims[mask])
    # P60=60, P85=85 → 60,70,80 통과 / 90,100(상위) 0,...,50(하위) 탈락
    assert kept == {60, 70, 80}


def test_bandpass_drops_top_and_bottom():
    sims = np.arange(0, 101, 10)
    mask = bandpass_mask(sims, lo=60, hi=85)
    assert not mask[sims.tolist().index(100)]  # 최상위(뻔함) 탈락
    assert not mask[sims.tolist().index(0)]     # 최하위(잡음) 탈락


def test_bandpass_single_candidate_passes():
    assert bandpass_mask(np.array([0.7])).tolist() == [True]


def test_bandpass_empty():
    assert bandpass_mask(np.array([])).tolist() == []


# ── rank_inverted ────────────────────────────────────────────
def _cand(cos, ts):
    """코사인 유사도와 날짜로 후보 dict 생성 (distance 역산)."""
    distance = math.sqrt(max(0.0, 2.0 * (1.0 - cos)))
    return {"distance": distance, "timestamp": ts, "title": f"t{cos}", "text": "x"}


def test_inverted_masked_get_zero_score():
    # 명확한 상위/하위는 밴드 밖 → 점수 0
    cands = [_cand(c, "2025-01-01") for c in (0.95, 0.75, 0.70, 0.68, 0.40)]
    ranked = rank_inverted(cands, NOW)
    for s in ranked:
        if not s.in_band:
            assert s.score == 0.0


def test_inverted_prefers_older_within_band():
    # 유사도 동일, 날짜만 다른 두 후보가 밴드 안일 때 더 오래된 글이 높은 점수
    cands = [
        _cand(0.70, "2026-06-01"),  # 최근
        _cand(0.70, "2024-06-01"),  # 오래됨
        _cand(0.95, "2025-01-01"),  # 상위(밴드 밖 유도)
        _cand(0.40, "2025-01-01"),  # 하위(밴드 밖 유도)
    ]
    ranked = rank_inverted(cands, NOW, half_life_days=365)
    banded = [s for s in ranked if s.in_band]
    # 밴드 안에서 가장 점수 높은 건 더 오래된(2024) 글
    top = max(banded, key=lambda s: s.score)
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
