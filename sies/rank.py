"""역전 재순위 — 제품의 뇌. 순수 결정론적 산수, ML·LLM 없음(원칙 2).

    점수 = 관련성 × (1 − 활성도) × 밴드패스가중치

- 관련성: 질의-청크 코사인 유사도 (벡터 거리에서 환산)
- 활성도: 시간 감쇠(최근일수록 1). (1−활성도) = '잊힌 정도'를 가중
- 밴드패스: 유사도 상위(뻔함)·하위(잡음)를 누르고 60~85 퍼센타일 부근만 살린다.
  하드 0/1 마스크는 경계에서 0.001 차이로 글이 증발하는 절벽이 있어,
  이중 시그모이드로 매끄럽게 만든다(soft bandpass):
      Mask(x) = σ(k(x − L)) − σ(k(x − U))
  L=P60 값, U=P85 값. 경계에서 ~0.5를 지나고 밴드 한참 밖이면 ~0.

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
# 소프트 밴드패스 시그모이드 가파름(클수록 하드 마스크에 가까움)
BAND_K = 50.0
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


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """수치 안정 시그모이드(오버플로 회피)."""
    z = np.asarray(z, dtype=float)
    pos = z >= 0
    out = np.empty_like(z)
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def double_sigmoid(
    x: np.ndarray, lo_val: float, hi_val: float, k: float = BAND_K
) -> np.ndarray:
    """이중 시그모이드 밴드 가중치 ∈ [0,1). σ(k(x−L)) − σ(k(x−U)).

    L~U 사이에서 높고(밴드 폭이 1/k보다 넓으면 ~1), 경계에서 ~0.5, 밖에서는 ~0.
    하드 마스크와 달리 경계에서 연속이라 0.001 차이로 글이 증발하지 않는다.
    """
    x = np.asarray(x, dtype=float)
    return _sigmoid(k * (x - lo_val)) - _sigmoid(k * (x - hi_val))


def bandpass_weight(
    sims: np.ndarray, lo: float = BAND_LO, hi: float = BAND_HI, k: float = BAND_K
) -> np.ndarray:
    """유사도 분포의 [lo,hi] 퍼센타일을 L,U로 잡아 소프트 밴드 가중치를 매긴다.

    후보가 1개면 분포가 없으므로 가중치 1로 통과시킨다.
    """
    sims = np.asarray(sims, dtype=float)
    if sims.size <= 1:
        return np.ones_like(sims, dtype=float)
    lo_val, hi_val = np.percentile(sims, [lo, hi])
    return double_sigmoid(sims, lo_val, hi_val, k)


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
    band_weight: float   # 소프트 밴드패스 가중치 ∈ [0,1]
    score: float


def rank_inverted(
    candidates: list[dict],
    now: dt.date,
    half_life_days: float = DEFAULT_HALF_LIFE,
    lo: float = BAND_LO,
    hi: float = BAND_HI,
    k: float = BAND_K,
) -> list[Scored]:
    """후보(각 dict에 'distance','timestamp')에 역전 점수를 매겨 내림차순 정렬.

    밴드 밖 후보는 가중치가 ~0이라 점수도 ~0 → 자연히 뒤로 밀린다.
    """
    if not candidates:
        return []
    sims = np.array([cosine_from_l2(c["distance"]) for c in candidates])
    weights = bandpass_weight(sims, lo, hi, k)
    out: list[Scored] = []
    for c, sim, w in zip(candidates, sims, weights):
        act = activity(_parse_ts(c.get("timestamp")), now, half_life_days)
        score = float(sim * (1.0 - act) * float(w))
        out.append(Scored(c, float(sim), act, float(w), score))
    out.sort(key=lambda s: s.score, reverse=True)
    return out


def rank_baseline(candidates: list[dict]) -> list[Scored]:
    """순수 유사도 정렬(거리 오름차순). 비교용 베이스라인."""
    out = [
        Scored(c, cosine_from_l2(c["distance"]), float("nan"), 1.0, cosine_from_l2(c["distance"]))
        for c in candidates
    ]
    out.sort(key=lambda s: s.similarity, reverse=True)
    return out
