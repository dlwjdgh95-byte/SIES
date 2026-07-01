"""Wikidata SPARQL로 '잊힌 인물' 후보 풀을 모은다.

직업(occupation) QID로 후보를 좁히고, en/ko 위키 sitelink 중 최소 하나가 있으면서
sitelink 총수가 문턱값 이상인 사람만 남긴다 — "한때 실제로 유의미하게 알려졌는가"의
1차 게이트. 점수 계산(정점 저명도·활성도)은 people.score가 pageviews 시계열로 담당하고,
여기서는 순수히 *후보 풀 구성*만 한다.
"""
from __future__ import annotations

import httpx
from dataclasses import dataclass

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "SIES-people-spike/0.1 (feasibility spike; personal project)"

# 직업 QID 큐레이션 — 넓히면 SPARQL이 60초 제한에 걸리기 쉬워 스파이크 범위로 제한한다.
OCCUPATION_QIDS: dict[str, list[str]] = {
    "politician": ["Q82955"],
    "entrepreneur": ["Q131524"],
    "actor": ["Q33999"],
    "musician": ["Q177220", "Q639669"],  # 가수, 음악가
    "influencer": ["Q17125263"],  # 유튜버
}

DEFAULT_MIN_SITELINKS = 15
DEFAULT_LIMIT = 500


@dataclass
class Candidate:
    qid: str
    label_en: str | None
    label_ko: str | None
    enwiki_title: str | None
    kowiki_title: str | None
    sitelinks: int
    birth_year: int | None
    occupation: str


def _build_query(occupation_qids: list[str], limit: int) -> str:
    values = " ".join(f"wd:{q}" for q in occupation_qids)
    return f"""
SELECT ?person ?labelEn ?labelKo ?enArticle ?koArticle ?sitelinks ?birth WHERE {{
  ?person wdt:P31 wd:Q5 ;
          wdt:P106 ?occupation ;
          wikibase:sitelinks ?sitelinks .
  VALUES ?occupation {{ {values} }}
  OPTIONAL {{ ?person rdfs:label ?labelEn . FILTER(LANG(?labelEn) = "en") }}
  OPTIONAL {{ ?person rdfs:label ?labelKo . FILTER(LANG(?labelKo) = "ko") }}
  OPTIONAL {{ ?enArticle schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> . }}
  OPTIONAL {{ ?koArticle schema:about ?person ; schema:isPartOf <https://ko.wikipedia.org/> . }}
  OPTIONAL {{ ?person wdt:P569 ?birth . }}
  FILTER(BOUND(?enArticle) || BOUND(?koArticle))
}}
ORDER BY DESC(?sitelinks)
LIMIT {limit}
""".strip()


def _title_from_article_url(url: str | None) -> str | None:
    """schema:about 결과는 풀 위키 URL(예: https://en.wikipedia.org/wiki/Barack_Obama).
    pageviews API가 요구하는 'article' 파라미터(=타이틀만)로 잘라낸다."""
    if not url:
        return None
    return url.rsplit("/wiki/", 1)[-1] if "/wiki/" in url else None


def fetch_candidates(
    occupation: str,
    min_sitelinks: int = DEFAULT_MIN_SITELINKS,
    limit: int = DEFAULT_LIMIT,
) -> list[Candidate]:
    """occupation(=OCCUPATION_QIDS 키, 'all'이면 전체 합집합) 후보 풀을 SPARQL로 가져온다.

    sitelink 문턱값은 SPARQL FILTER가 아니라 클라이언트 사이드에서 거른다 — 재쿼리 없이
    임계값만 바꿔 재실행할 수 있게(캐시된 응답을 그대로 재사용할 여지를 남겨둔다).
    """
    if occupation == "all":
        qids = [q for group in OCCUPATION_QIDS.values() for q in group]
    else:
        qids = OCCUPATION_QIDS[occupation]

    query = _build_query(qids, limit)
    resp = httpx.post(
        SPARQL_ENDPOINT,
        data={"query": query},
        headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT},
        timeout=60.0,
    )
    resp.raise_for_status()
    bindings = resp.json()["results"]["bindings"]

    out: list[Candidate] = []
    for b in bindings:
        sitelinks = int(b["sitelinks"]["value"])
        if sitelinks < min_sitelinks:
            continue
        qid = b["person"]["value"].rsplit("/", 1)[-1]
        birth_raw = b.get("birth", {}).get("value")
        birth_year = int(birth_raw[:4]) if birth_raw else None
        out.append(
            Candidate(
                qid=qid,
                label_en=b.get("labelEn", {}).get("value"),
                label_ko=b.get("labelKo", {}).get("value"),
                enwiki_title=_title_from_article_url(b.get("enArticle", {}).get("value")),
                kowiki_title=_title_from_article_url(b.get("koArticle", {}).get("value")),
                sitelinks=sitelinks,
                birth_year=birth_year,
                occupation=occupation,
            )
        )
    return out
