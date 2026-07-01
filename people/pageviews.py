"""Wikimedia Pageviews REST — 인물별 월간 조회수 시계열 + 파일 캐시.

'현재 활성도' 신호의 원천. 데이터는 2015-07부터 제공된다(그 이전 정점은 관측 불가 —
구조적 한계이며 이 스파이크 단계에서는 있는 데이터로만 판단한다).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

import httpx

PAGEVIEWS_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
USER_AGENT = "SIES-people-spike/0.1 (feasibility spike; personal project)"
DEFAULT_START = "2015070100"

# 공유 커넥션 — 후보 1천+명을 순차 조회하므로 요청마다 TLS 핸드셰이크를 새로 하지 않는다
# (프록시 경유 환경에서 간헐적 SSL EOF의 주원인이기도 했다).
_CLIENT: httpx.Client | None = None


def _client() -> httpx.Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0)
    return _CLIENT


def _get_with_retry(url: str, retries: int = 4) -> httpx.Response | None:
    """GET — 일시적 실패(네트워크 오류·5xx)는 지수 백오프로 재시도.

    재시도를 소진하면 None — 대량 순차 조회 중 한 명의 실패로 전체 실행이
    죽지 않도록 호출자가 '이 사람만 건너뜀'을 택할 수 있게 한다.
    """
    for attempt in range(retries):
        try:
            resp = _client().get(url)
            if resp.status_code >= 500:
                resp.raise_for_status()
            return resp
        except httpx.HTTPError as exc:
            if attempt == retries - 1:
                print(f"  ! pageviews 실패({url.rsplit('/metrics/', 1)[-1][:80]}) — 건너뜀: {exc}",
                      file=sys.stderr)
                return None
            time.sleep(2 ** (attempt + 1))  # 2s, 4s, 8s
    return None


@dataclass
class MonthlyViews:
    month: str  # "YYYY-MM"
    views: int


def _cache_path(cache_dir: Path, project: str, article: str) -> Path:
    safe = quote(article, safe="").replace("%", "_")
    return cache_dir / f"{project}_{safe}.json"


def fetch_pageviews(
    project: str,
    article: str,
    cache_dir: Path,
    start: str = DEFAULT_START,
    end: str | None = None,
    refresh: bool = False,
) -> list[MonthlyViews]:
    """project(예: 'en.wikipedia'/'ko.wikipedia')·article(위키 타이틀)의 월별 조회수.

    문서/데이터가 없으면(404) 빈 리스트 — 예외로 취급하지 않는다.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, project, article)
    if path.exists() and not refresh:
        raw = json.loads(path.read_text())
        return [MonthlyViews(**r) for r in raw]

    end = end or _this_month_start()
    url = f"{PAGEVIEWS_BASE}/{project}/all-access/user/{quote(article, safe='')}/monthly/{start}/{end}"
    resp = _get_with_retry(url)
    if resp is None:
        # 지속 실패 — 캐시에 쓰지 않고 빈 시계열 반환(다음 실행에서 자연 재시도되도록).
        return []
    if resp.status_code == 404:
        result: list[MonthlyViews] = []
    else:
        resp.raise_for_status()
        items = resp.json().get("items", [])
        result = [
            MonthlyViews(month=f"{it['timestamp'][:4]}-{it['timestamp'][4:6]}", views=it["views"])
            for it in items
        ]

    path.write_text(json.dumps([asdict(r) for r in result], ensure_ascii=False))
    return result


def _this_month_start() -> str:
    import datetime as dt

    today = dt.date.today()
    return f"{today.year:04d}{today.month:02d}0100"
