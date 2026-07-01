"""잊힌 인물 스코어 — SIES 원칙 2 계승: 결정론적 산수, LLM 없음.

    점수 = 정점_저명도(풀 내 상대) × (1 − 활성도) × 밴드패스가중치

- 정점_저명도: 후보의 피크 월간 조회수. 절대값이 아니라 풀 전체 분포에서의 상대 위치가
  중요하므로(밴드패스가 퍼센타일 기반이라 자동 상대화됨), raw peak_views를 그대로
  bandpass_weight에 태운다. 표시용으로만 [0,1] 정규화한 필드를 별도로 둔다.
- 활성도: sies.rank.activity() 그대로 재사용 — ts=정점이 있었던 월, now=오늘.
  "정점이 최근일수록 아직 안 잊힘". half_life는 SIES 기본값(365일, 글 하나의 스케일)이
  아니라 PERSON_HALF_LIFE_DAYS(3년)로 별도 튜닝 — 유명인의 '잊힘'은 훨씬 느린 시간축이다.
- 밴드패스: sies.rank.bandpass_weight() 그대로. 상위(여전히 압도적 유명 — 재발견 가치
  없음)와 하위(애초에 유명한 적 없음 — 노이즈) 둘 다 눌러, "한때 확실히 유명했지만 지금은
  조용한" 인물만 남긴다. SIES 기본(60~85)보다 상단을 타이트하게(60~90 → 상위 10%만 컷이
  아니라 90 자체가 컷 시작점) 잡을 이유는 없어 보였으나, 인물 도메인은 "여전히 활동 중인
  현역 스타"가 코퍼스 도메인의 "뻔한 1위 글"보다 훨씬 극단적으로 조회수를 독점하므로
  상단 컷을 더 세게(BAND_HI=90) 건다.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from sies.rank import activity, bandpass_weight
from people.pageviews import MonthlyViews
from people.wikidata import Candidate

PERSON_HALF_LIFE_DAYS = 1095.0  # 인물 재발견 반감기(3년) — 글 한 편(365일)보다 느린 스케일.
BAND_LO = 60.0
BAND_HI = 90.0  # SIES(85)보다 타이트 — 현역 초유명인의 조회수 독점을 더 세게 누른다.
BAND_K = 50.0


@dataclass
class PersonScore:
    candidate: Candidate
    peak_views: int
    peak_month: str | None       # "YYYY-MM", 시계열이 비어있으면 None
    peak_prominence: float       # [0,1] 표시용 정규화(풀 내 최댓값=1)
    person_activity: float
    band_weight: float
    trend_signal: float | None   # pytrends 참고용, 점수엔 미반영
    score: float


def peak_month_series(views: list[MonthlyViews]) -> tuple[str | None, int]:
    """월별 조회수에서 (피크 월, 피크 조회수)를 뽑는다. 스무딩 없음(스파이크 단계).

    동률(같은 조회수)이면 더 이른 달을 택한다 — max()는 첫 최댓값을 유지하는 안정 정렬이라
    입력 순서(시간순)를 그대로 따르면 자연히 '가장 이른 정점'이 뽑힌다.
    """
    if not views:
        return None, 0
    peak = max(views, key=lambda v: v.views)
    return peak.month, peak.views


def _month_to_date(month: str) -> dt.date:
    year, mon = month.split("-")
    return dt.date(int(year), int(mon), 1)


def score_candidates(
    series_by_qid: dict[str, list[MonthlyViews]],
    candidates: list[Candidate],
    now: dt.date,
    half_life_days: float = PERSON_HALF_LIFE_DAYS,
    lo: float = BAND_LO,
    hi: float = BAND_HI,
    k: float = BAND_K,
    trend_signals: dict[str, float | None] | None = None,
) -> list[PersonScore]:
    """후보 전원의 피크 저명도 분포에 밴드패스를 걸고, 정점 이후 경과로 활성도를 매겨 정렬.

    시계열이 없는(peak_views=0) 후보는 밴드패스 하위컷에서 자연히 눌린다 — 별도 예외처리 불필요.
    """
    if not candidates:
        return []
    trend_signals = trend_signals or {}

    peaks = [peak_month_series(series_by_qid.get(c.qid, [])) for c in candidates]
    peak_views = np.array([pv for _, pv in peaks], dtype=float)
    weights = bandpass_weight(peak_views, lo, hi, k)
    max_views = float(peak_views.max()) if peak_views.size and peak_views.max() > 0 else 1.0

    out: list[PersonScore] = []
    for c, (peak_month, pv), w in zip(candidates, peaks, weights):
        a = activity(_month_to_date(peak_month) if peak_month else None, now, half_life_days)
        score = float(pv * (1.0 - a) * float(w))
        out.append(
            PersonScore(
                candidate=c,
                peak_views=pv,
                peak_month=peak_month,
                peak_prominence=pv / max_views,
                person_activity=a,
                band_weight=float(w),
                trend_signal=trend_signals.get(c.qid),
                score=score,
            )
        )
    out.sort(key=lambda s: s.score, reverse=True)
    return out
