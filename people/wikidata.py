"""Wikidata SPARQL로 '잊힌 인물' 후보 풀을 모은다.

직업(occupation) QID로 후보를 좁히고, en/ko 위키 sitelink 중 최소 하나가 있으면서
sitelink 총수가 문턱값 이상인 사람만 남긴다 — "한때 실제로 유의미하게 알려졌는가"의
1차 게이트. 점수 계산(정점 저명도·활성도)은 people.score가 pageviews 시계열로 담당하고,
여기서는 순수히 *후보 풀 구성*만 한다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

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


def _build_query(occupation_qids: list[str], min_sitelinks: int, limit: int) -> str:
    values = " ".join(f"wd:{q}" for q in occupation_qids)
    # 후보 풀이 큰 직업(예: politician)은 ORDER BY가 정치인 집합 전체를 훑어 WDQS 60초
    # 제한(504)에 걸린다. 두 가지로 비용을 낮춘다:
    #   1) sitelink 문턱을 서버측 FILTER로 밀어 넣어 정렬 대상 집합을 먼저 줄인다
    #      (대다수 인물은 sitelink가 적어 이 필터 하나로 후보가 크게 줄어든다).
    #   2) 정렬·상한은 ?person/?sitelinks 두 컬럼만 다루는 서브쿼리에서 끝내고,
    #      라벨·문서 URL 같은 무거운 OPTIONAL 조인은 상위 N명에 대해서만 바깥에서 푼다.
    # en/ko 문서 요구(FILTER BOUND)로 상위권 일부가 탈락할 수 있어 서브쿼리에서 3배
    # 과다-추출한 뒤 바깥에서 limit으로 자른다.
    inner_limit = limit * 3
    return f"""
SELECT ?person ?labelEn ?labelKo ?enArticle ?koArticle ?sitelinks ?birth WHERE {{
  {{
    SELECT ?person ?sitelinks WHERE {{
      ?person wdt:P31 wd:Q5 ;
              wdt:P106 ?occupation ;
              wikibase:sitelinks ?sitelinks .
      VALUES ?occupation {{ {values} }}
      FILTER(?sitelinks >= {min_sitelinks})
    }}
    ORDER BY DESC(?sitelinks)
    LIMIT {inner_limit}
  }}
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


def _post_sparql(query: str, retries: int = 4) -> httpx.Response:
    """SPARQL POST — 일시적 실패(네트워크 오류·5xx)는 지수 백오프로 재시도한다.

    WDQS는 비싼 쿼리로 타임아웃을 반복하면 잠시 연결을 끊거나 5xx를 돌려주므로,
    스파이크 실행 한 번이 일시적 흔들림에 통째로 죽지 않도록 감싼다.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = httpx.post(
                SPARQL_ENDPOINT,
                data={"query": query},
                headers={
                    "Accept": "application/sparql-results+json",
                    "User-Agent": USER_AGENT,
                },
                timeout=90.0,
            )
            if resp.status_code >= 500:
                resp.raise_for_status()
            return resp
        except (httpx.HTTPError,) as exc:
            last_exc = exc
            if attempt == retries - 1:
                break
            time.sleep(2 ** (attempt + 1))  # 2s, 4s, 8s, ...
    assert last_exc is not None
    raise last_exc


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

    sitelink 문턱값은 서버측 FILTER로 밀어 넣어 정렬 비용을 낮춘다(_build_query 참고).
    클라이언트 사이드 재확인은 안전망 — 서브쿼리 과다-추출 경계에서 들어온 값을 한 번 더 거른다.
    """
    if occupation == "all":
        qids = [q for group in OCCUPATION_QIDS.values() for q in group]
    else:
        qids = OCCUPATION_QIDS[occupation]

    query = _build_query(qids, min_sitelinks, limit)
    resp = _post_sparql(query)
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
