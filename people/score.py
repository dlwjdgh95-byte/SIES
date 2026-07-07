"""잊힌 인물 스코어 — SIES 원칙 2 계승: 결정론적 산수, LLM 없음.

    점수 = 정점_저명도(풀 내 상대) × (1 − 활성도) × 밴드패스가중치
    단, 활성도 ≥ max_activity 인 후보는 점수 이전에 풀에서 제외한다.

- 정점_저명도: 후보의 피크 월간 조회수. 절대값이 아니라 풀 전체 분포에서의 상대 위치가
  중요하므로(밴드패스가 퍼센타일 기반이라 자동 상대화됨), raw peak_views를 그대로
  bandpass_weight에 태운다. 표시용으로만 [0,1] 정규화한 필드를 별도로 둔다.
- 활성도: 세 축의 max — 어느 한 축이라도 '아직 활발'이면 활발로 본다.
    A_time = 0.5^(정점 이후 경과/H)   정점이 최근 → 아직 안 잊힘
    A_now  = 최근 12개월 평균 조회수 / 정점 조회수   지금도 정점급으로 보임 → 안 잊힘
    A_abs  = 1 − 0.5^(최근 평균/H_뷰)   절대 규모로 지금도 많이 보임 → 안 잊힘
    A = max(A_time, A_now, A_abs)
  A_time만 쓰면 "정점은 2016년이지만 지금도 매달 수십만 뷰"인 레이건·아인슈타인류가
  잊힌 것으로 잘못 잡힌다. A_now(자기 정점 대비 궤적)를 더해도, 바이럴 스파이크로 정점이
  극단적으로 높았던 초유명인(비욘세: 정점 187만, 최근 23만/월 → 비율 0.12)은 새어 들어온다
  — 그래서 절대 규모 축 A_abs가 필요하다: 지금도 월 수십만 명이 찾아보는 사람은 자기
  정점이 얼마였든 잊힌 게 아니다. 잊힘은 '정점이 오래됨 AND 자기 정점 대비 미미함 AND
  절대적으로도 조용함' 셋 다 필요하므로 활성은 그 부정인 max로 결합한다.
  (sies.rank의 min과 방향이 반대인 이유: 저쪽은 '잊힘 축들의 OR', 여기는 '활성 축들의 OR.)
  half_life는 SIES 기본값(365일, 글 하나의 스케일)이 아니라 PERSON_HALF_LIFE_DAYS(3년)로
  별도 튜닝 — 유명인의 '잊힘'은 훨씬 느린 시간축이다.
- 활성도 게이트: A ≥ MAX_ACTIVITY(기본 0.4)면 후보에서 탈락. 밴드패스가 정점 조회수
  분포의 퍼센타일로 잡히므로, 여전히 활발한 초대형 인물을 풀에 남겨두면 분포만 왜곡하고
  결과에는 어차피 못 들어온다 — 먼저 빼고 남은 '잊힌 풀'에서 밴드를 잡는 게 정직하다.
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
RECENT_WINDOW_MONTHS = 12  # '지금도 보이는가'(A_now/A_abs)의 관측 창.
MAX_ACTIVITY = 0.4  # 이 이상이면 '아직 활발' — 후보에서 제외. sies.rank.LOW_ACTIVITY와 같은 눈금.
# 절대 주목도 반감 눈금(월간 뷰). 최근 평균이 이만큼이면 A_abs=0.5.
# 게이트(0.4)와 조합하면 월 ~3.7만 뷰 이상은 '아직 활발'로 컷 — 반감기류 상수처럼 튜닝 노브.
RECENT_FAME_HALF_VIEWS = 50_000.0


@dataclass
class PersonScore:
    candidate: Candidate
    peak_views: int
    peak_month: str | None       # "YYYY-MM", 시계열이 비어있으면 None
    peak_prominence: float       # [0,1] 표시용 정규화(풀 내 최댓값=1)
    person_activity: float       # 최종 = max(시간, 현재주목)
    band_weight: float
    trend_signal: float | None   # pytrends 참고용, 점수엔 미반영
    score: float
    activity_time: float = float("nan")   # 정점 경과 축
    activity_now: float = float("nan")    # 현재 주목 축(최근 평균/정점 — 자기 궤적)
    activity_abs: float = float("nan")    # 현재 주목 축(절대 규모)
    recent_views: float = 0.0             # 최근 12개월 월평균 조회수


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


def recent_mean_views(
    views: list[MonthlyViews], now: dt.date, window_months: int = RECENT_WINDOW_MONTHS
) -> float:
    """now 기준 최근 window_months개월의 월평균 조회수 — A_now의 분자.

    "YYYY-MM" 문자열은 사전순 = 시간순이라 컷오프 문자열 비교로 창을 자른다.
    창 안에 데이터가 한 달도 없으면 0(= 요즘 아무도 안 봄)으로 본다.
    평균의 분모는 '창 안에 실제로 존재하는 달 수' — pageviews API는 조회 0인 달을
    아예 빼고 주기도 해서, 고정 분모(window_months)로 나누면 결측이 많은 옛 인물의
    현재 주목도가 과소평가될 수 있어 관측된 달만으로 평균한다(보수적인 쪽).
    """
    if not views:
        return 0.0
    now_idx = now.year * 12 + (now.month - 1)
    cy, cm = divmod(now_idx - window_months, 12)
    cutoff = f"{cy:04d}-{cm + 1:02d}"
    recent = [v.views for v in views if v.month >= cutoff]
    return float(sum(recent)) / len(recent) if recent else 0.0


def score_candidates(
    series_by_qid: dict[str, list[MonthlyViews]],
    candidates: list[Candidate],
    now: dt.date,
    half_life_days: float = PERSON_HALF_LIFE_DAYS,
    lo: float = BAND_LO,
    hi: float = BAND_HI,
    k: float = BAND_K,
    trend_signals: dict[str, float | None] | None = None,
    max_activity: float | None = MAX_ACTIVITY,
) -> list[PersonScore]:
    """활성도 게이트를 통과한 후보만 남기고, 그 풀의 피크 분포에 밴드패스를 걸어 정렬.

    순서가 중요하다: 게이트 → 밴드패스. 여전히 활발한 초대형 인물을 먼저 빼야
    밴드 퍼센타일이 '잊힌 풀' 안에서 잡힌다(안 빼면 그들이 분포 상단을 다 차지해
    중간 밴드가 위로 쏠린다). max_activity=None이면 게이트 없이 전원 스코어링(비교·디버그용).

    시계열이 없는(peak_views=0) 후보는 A_time이 결측 중립값(0.5)이라 기본 게이트(0.4)에서
    탈락한다 — 데이터가 없으면 '잊힘'을 주장할 근거도 없다는 뜻이라 의도된 동작.
    """
    if not candidates:
        return []
    trend_signals = trend_signals or {}

    survivors: list[tuple[Candidate, str | None, int, float, float, float, float, float]] = []
    for c in candidates:
        series = series_by_qid.get(c.qid, [])
        peak_month, pv = peak_month_series(series)
        a_time = activity(_month_to_date(peak_month) if peak_month else None, now, half_life_days)
        rmean = recent_mean_views(series, now)
        a_now = min(rmean / pv, 1.0) if pv > 0 else 0.0
        a_abs = 1.0 - 0.5 ** (rmean / RECENT_FAME_HALF_VIEWS)
        a = max(a_time, a_now, a_abs)
        if max_activity is not None and a >= max_activity:
            continue
        survivors.append((c, peak_month, pv, a_time, a_now, a_abs, a, rmean))

    if not survivors:
        return []

    peak_views = np.array([s[2] for s in survivors], dtype=float)
    weights = bandpass_weight(peak_views, lo, hi, k)
    max_views = float(peak_views.max()) if peak_views.size and peak_views.max() > 0 else 1.0

    out: list[PersonScore] = []
    for (c, peak_month, pv, a_time, a_now, a_abs, a, rmean), w in zip(survivors, weights):
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
                activity_time=a_time,
                activity_now=a_now,
                activity_abs=a_abs,
                recent_views=rmean,
            )
        )
    out.sort(key=lambda s: s.score, reverse=True)
    return out
