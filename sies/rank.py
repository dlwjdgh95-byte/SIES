"""역전 재순위 — 제품의 뇌. 순수 결정론적 산수, ML·LLM 없음(원칙 2). 상수·점수식 단일 소스.

    점수 = 관련성 × (1 − 활성도) × 밴드패스가중치

- 관련성: 질의-청크 코사인 유사도(L2 거리에서 환산)
- 활성도 = min(A_time, A_vol) — "오래됐거나 OR 덮였거나" 더 잊힌 쪽 채택.
    A_time = 0.5^(나이/H_time),  A_vol = 0.5^(V/H_vol)  (V = 뒤에 덮은 새 글 수)
- 밴드패스: 유사도 상위(뻔함)·하위(잡음)를 누르고 P60~P85 부근만 살린다.
  하드 0/1 마스크의 경계 절벽(0.001 차이로 증발)을 피해 이중 시그모이드:
      Mask(x) = σ(k(x−L)) − σ(k(x−U)),  L=P60값, U=P85값, 중앙 피크=1로 정규화

의도: *적당히 관련되면서 오래 잊힌* 글을 끌어올린다.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

# 활성도 기본 반감기(일). 이 일수만큼 지나면 시간 활성도 0.5.
DEFAULT_HALF_LIFE = 365.0
# 볼륨 반감기(편). 뒤에 이만큼 새 글이 덮이면 볼륨 활성도 0.5.
DEFAULT_VOLUME_HALF_LIFE = 50.0
# 밴드패스 통과 구간(퍼센타일)
BAND_LO = 60.0
BAND_HI = 85.0
# 소프트 밴드패스 시그모이드 가파름(클수록 하드 마스크에 가까움)
BAND_K = 50.0
# 타임스탬프가 없는 글의 활성도(중립)
MISSING_ACTIVITY = 0.5
# 이 활성도 미만이면 '잊힌' 것으로 본다(gated 랭커의 통과 기준).
# stats·replay도 이 값을 import — 여기가 단일 소스.
LOW_ACTIVITY = 0.4


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


def volume_activity(newer_count: int, half_life: float = DEFAULT_VOLUME_HALF_LIFE) -> float:
    """볼륨 감쇠 활성도 ∈ (0,1]. 뒤에 덮인 글이 없으면 1, 많을수록 0에 수렴."""
    return float(0.5 ** (max(newer_count, 0) / half_life))


def newer_counts(candidates: list[dict]) -> dict:
    """문서별로 '자기보다 날짜가 새로운 문서 수'를 센다. 몰아쓰기(볼륨) 축의 V.

    문서 식별자는 doc_path(없으면 title). 날짜 없는 문서는 V=0(안 덮임)으로 둔다.
    한계: 날짜 단위 해상도라 *같은 날* 몰아쓴 글끼리는 서로 안 덮인 것(V 동률)으로 본다.
    여러 날에 걸친 버스트는 정상 포착. 같은 날 해소가 필요하면 datetime 정밀도가 있어야 한다.
    """
    doc_date: dict = {}
    for c in candidates:
        key = c.get("doc_path") or c.get("title")
        doc_date[key] = _parse_ts(c.get("timestamp"))
    dated = [d for d in doc_date.values() if d is not None]
    counts: dict = {}
    for key, d in doc_date.items():
        counts[key] = 0 if d is None else sum(1 for o in dated if o > d)
    return counts


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
    sims: np.ndarray,
    lo: float = BAND_LO,
    hi: float = BAND_HI,
    k: float = BAND_K,
    normalize: bool = True,
) -> np.ndarray:
    """유사도 분포의 [lo,hi] 퍼센타일을 L,U로 잡아 소프트 밴드 가중치를 매긴다.

    L,U는 매 질의 분포에서 np.percentile로 뽑은 *유사도 값*(순위가 아니라 값).
    normalize=True면 곡선 최대값(대칭이라 중앙 (L+U)/2)으로 나눠 밴드 중앙 = 1로 맞춘다.
    좁은 밴드 + 큰 k로 두 시그모이드가 겹쳐 피크가 1에 못 미치는 것을 보정한다.
    후보가 1개면 분포가 없으므로 가중치 1로 통과시킨다.
    """
    sims = np.asarray(sims, dtype=float)
    if sims.size <= 1:
        return np.ones_like(sims, dtype=float)
    lo_val, hi_val = np.percentile(sims, [lo, hi])
    w = double_sigmoid(sims, lo_val, hi_val, k)
    if normalize:
        peak = float(double_sigmoid((lo_val + hi_val) / 2.0, lo_val, hi_val, k))
        if peak > 1e-9:
            w = w / peak
    return w


def _parse_ts(s: str | None) -> dt.date | None:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s)
    except ValueError:
        return None


def gated_score(sim: float, act: float, low: float = LOW_ACTIVITY) -> float:
    """게이팅 점수식(B) — replay의 오프라인 비교와 rank_gated가 공유하는 커널.

        저활성(잊힘): 유사도 그대로 → 관련 깊은 옛 금은 줄세움만
        그 외:       유사도 × (1−활성도) → 최근일수록 누름
    """
    return sim if act < low else sim * (1.0 - act)


@dataclass
class Scored:
    candidate: dict      # store.search가 돌려준 원본 행
    similarity: float
    activity: float      # 최종 = min(시간, 볼륨)
    band_weight: float   # 소프트 밴드패스 가중치 ∈ [0,1]
    score: float
    activity_time: float = float("nan")
    activity_vol: float = float("nan")
    volume: int = 0


def _measure(
    candidates: list[dict],
    now: dt.date,
    half_life_days: float,
    volume_half_life: float,
) -> list[tuple[dict, float, float, float, float, int]]:
    """후보별 (원본, 유사도, 시간·볼륨·최종 활성도, V) — 두 랭커가 공유하는 계측."""
    counts = newer_counts(candidates)
    rows = []
    for c in candidates:
        sim = cosine_from_l2(c["distance"])
        a_time = activity(_parse_ts(c.get("timestamp")), now, half_life_days)
        vol = counts[c.get("doc_path") or c.get("title")]
        a_vol = volume_activity(vol, volume_half_life)
        rows.append((c, sim, a_time, a_vol, min(a_time, a_vol), vol))
    return rows


def rank_inverted(
    candidates: list[dict],
    now: dt.date,
    half_life_days: float = DEFAULT_HALF_LIFE,
    lo: float = BAND_LO,
    hi: float = BAND_HI,
    k: float = BAND_K,
    volume_half_life: float = DEFAULT_VOLUME_HALF_LIFE,
) -> list[Scored]:
    """후보(각 dict에 'distance','timestamp','doc_path')에 역전 점수를 매겨 내림차순 정렬.

    활성도 = min(시간, 볼륨). 밴드 밖 후보는 가중치 ~0이라 점수 ~0 → 뒤로 밀린다.
    """
    if not candidates:
        return []
    rows = _measure(candidates, now, half_life_days, volume_half_life)
    weights = bandpass_weight(np.array([r[1] for r in rows]), lo, hi, k)
    out = [
        Scored(c, float(sim), a_final, float(w), float(sim * (1.0 - a_final) * float(w)),
               activity_time=a_time, activity_vol=a_vol, volume=vol)
        for (c, sim, a_time, a_vol, a_final, vol), w in zip(rows, weights)
    ]
    out.sort(key=lambda s: s.score, reverse=True)
    return out


def rank_gated(
    candidates: list[dict],
    now: dt.date,
    half_life_days: float = DEFAULT_HALF_LIFE,
    volume_half_life: float = DEFAULT_VOLUME_HALF_LIFE,
    low: float = LOW_ACTIVITY,
) -> list[Scored]:
    """게이팅 랭커(B) — 밴드패스 없이 '잊힘'만 조건부로 거든다(점수식은 gated_score).

    오프라인 재실행에서 밴드패스 역전(공격적)·순수 잊힘보상(A)보다 나았던 식.
    밴드 억제가 없으니 band_weight=1.0으로 둔다.
    """
    if not candidates:
        return []
    out = [
        Scored(c, sim, a_final, 1.0, float(gated_score(sim, a_final, low)),
               activity_time=a_time, activity_vol=a_vol, volume=vol)
        for c, sim, a_time, a_vol, a_final, vol
        in _measure(candidates, now, half_life_days, volume_half_life)
    ]
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
