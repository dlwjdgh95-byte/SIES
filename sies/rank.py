"""역전 재순위 — 제품의 뇌. 순수 결정론적 산수, ML·LLM 없음(원칙 2).

    점수 = 관련성 × (1 − 활성도) × 밴드패스마스크

- 관련성: 질의-청크 코사인 유사도 (벡터 거리에서 환산)
- 활성도: 시간 감쇠(최근일수록 1). (1−활성도) = '잊힌 정도'를 가중
- 밴드패스: 유사도 상위 15%(뻔함)·하위(잡음) 버리고 60~85 퍼센타일만 통과

의도: "뻔한 1위"도 "관련 없는 잡음"도 아닌, *적당히 관련되면서 오래 잊힌* 글을 끌어올린다.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

# 활성도 기본 반감기(일). 이 일수만큼 지나면 활성도 0.5.
DEFAULT_HALF_LIFE = 365.0
# 밴드패스 통과 구간(퍼센타일)
BAND_LO = 60.0
BAND_HI = 85.0
# 타임스탬프가 없는 글의 활성도(중립)
MISSING_ACTIVITY = 0.5


def cosine_from_l2(distance: float) -> float:
    """정규화 벡터의 L2 거리 d → 코사인 유사도. d²=2(1−cos). [0,1]로 클램프."""
    cos = 1.0 - (distance * distance) / 2.0
    return float(min(1.0, max(0.0, cos)))


def activity(ts: dt.date | None, now: dt.date, half_life_days: float = DEFAULT_HALF_LIFE) -> float:
    """시간 감쇠 활성도 ∈ (0,1]. 최근=1에 가깝고 오래될수록 0에 수렴."""
    if ts is None:
        return MISSING_ACTIVITY
    age_days = max((now - ts).days, 0)
    return float(0.5 ** (age_days / half_life_days))


def bandpass_mask(sims: np.ndarray, lo: float = BAND_LO, hi: float = BAND_HI) -> np.ndarray:
    """유사도 분포에서 [lo, hi] 퍼센타일만 True. 후보가 1개면 통과시킨다."""
    sims = np.asarray(sims, dtype=float)
    if sims.size <= 1:
        return np.ones_like(sims, dtype=bool)
    lo_v, hi_v = np.percentile(sims, [lo, hi])
    return (sims >= lo_v) & (sims <= hi_v)


def _parse_ts(s: str | None) -> dt.date | None:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s)
    except ValueError:
        return None


@dataclass
class Scored:
    candidate: dict      # store.search가 돌려준 원본 행
    similarity: float
    activity: float
    in_band: bool
    score: float


def rank_inverted(
    candidates: list[dict],
    now: dt.date,
    half_life_days: float = DEFAULT_HALF_LIFE,
    lo: float = BAND_LO,
    hi: float = BAND_HI,
) -> list[Scored]:
    """후보(각 dict에 'distance','timestamp')에 역전 점수를 매겨 내림차순 정렬.

    밴드 밖(mask=0) 후보는 점수 0 → 자연히 뒤로 밀린다.
    """
    if not candidates:
        return []
    sims = np.array([cosine_from_l2(c["distance"]) for c in candidates])
    mask = bandpass_mask(sims, lo, hi)
    out: list[Scored] = []
    for c, sim, in_band in zip(candidates, sims, mask):
        act = activity(_parse_ts(c.get("timestamp")), now, half_life_days)
        score = float(sim * (1.0 - act) * (1.0 if in_band else 0.0))
        out.append(Scored(c, float(sim), act, bool(in_band), score))
    out.sort(key=lambda s: s.score, reverse=True)
    return out


def rank_baseline(candidates: list[dict]) -> list[Scored]:
    """순수 유사도 정렬(거리 오름차순). 비교용 베이스라인."""
    out = [
        Scored(c, cosine_from_l2(c["distance"]), float("nan"), True, cosine_from_l2(c["distance"]))
        for c in candidates
    ]
    out.sort(key=lambda s: s.similarity, reverse=True)
    return out
