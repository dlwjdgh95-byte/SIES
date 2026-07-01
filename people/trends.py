"""Google Trends(pytrends) — 참고용 부가 신호, 점수식에는 반영하지 않는다.

pytrends는 비공식 스크래핑 기반이라 같은 질의도 실행마다 값이 흔들리고 레이트리밋(429)에
쉽게 걸린다. SIES 원칙 2(결정론적 산수)와 정면으로 충돌하는 성질이라, 여기서는 실패를
전부 삼켜 None으로 낙하시키고 CLI 출력의 sanity-check 필드로만 곁들인다 — 점수 계산은
이 값 없이도 항상 완결된다.
"""
from __future__ import annotations


def fetch_trend_signal(query: str) -> float | None:
    """최근 관심도 지수(0~100, pytrends 기준)를 최선 노력으로 가져온다. 실패 시 None."""
    try:
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl="ko-KR", tz=540)
        pytrends.build_payload([query], timeframe="today 3-m")
        df = pytrends.interest_over_time()
        if df.empty or query not in df.columns:
            return None
        return float(df[query].mean())
    except Exception:
        return None
