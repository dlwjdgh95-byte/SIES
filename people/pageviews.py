"""Wikimedia Pageviews REST — 인물별 월간 조회수 시계열 + 파일 캐시.

'현재 활성도' 신호의 원천. 데이터는 2015-07부터 제공된다(그 이전 정점은 관측 불가 —
구조적 한계이며 이 스파이크 단계에서는 있는 데이터로만 판단한다).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

import httpx

PAGEVIEWS_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
USER_AGENT = "SIES-people-spike/0.1 (feasibility spike; personal project)"
DEFAULT_START = "2015070100"


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
    resp = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=30.0)
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
