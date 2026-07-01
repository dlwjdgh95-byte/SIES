"""Wikidata SPARQL로 '잊힌 인물' 후보 풀을 모은다.

직업(occupation) QID로 후보를 좁히고, en/ko 위키 sitelink 중 최소 하나가 있으면서
sitelink 총수가 문턱값 이상인 사람만 남긴다 — "한때 실제로 유의미하게 알려졌는가"의
1차 게이트. 점수 계산(정점 저명도·활성도)은 people.score가 pageviews 시계열로 담당하고,
여기서는 순수히 *후보 풀 구성*만 한다.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from urllib.parse import unquote

import httpx

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "SIES-people-spike/0.1 (feasibility spike; personal project)"

# 직업 QID 큐레이션. 서브쿼리+sitelink 서버측 필터(_build_query 참고)로 정렬 비용을
# 낮춘 뒤로는 넓은 직업군도 60초 제한 안에서 돈다. '역사 속 인물'은 별도 카테고리가
# 아니라 — 생존 여부로 거르지 않으므로 — 아래 직업군 안에 자연히 섞여 나온다
# (예: 철학자·군주·군인·화가에는 근현대 이전 인물이 다수).
OCCUPATION_QIDS: dict[str, list[str]] = {
    # 정치·권력
    "politician": ["Q82955"],
    "monarch": ["Q116"],              # 군주 — 역사 속 왕/황제
    "military": ["Q47064"],           # 군인
    # 사업·미디어
    "entrepreneur": ["Q131524"],
    "journalist": ["Q1930187"],       # 언론인
    "influencer": ["Q17125263"],      # 유튜버
    # 연예·대중문화(=유명인)
    "actor": ["Q33999"],
    "musician": ["Q177220", "Q639669"],  # 가수, 음악가
    "athlete": ["Q2066131"],          # 선수
    # 학문·사상
    "scientist": ["Q901", "Q170790", "Q188094"],  # 과학자, 수학자, 경제학자
    "philosopher": ["Q4964182"],      # 철학자
    "writer": ["Q36180", "Q49757"],   # 작가, 시인
    # 예술·종교
    "artist": ["Q483501", "Q1028181"],  # 예술가, 화가
    "religious": ["Q2259532", "Q42603"],  # 성직자, 사제
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


def _build_query(qid: str, min_sitelinks: int, limit: int) -> str:
    # 후보 풀이 큰 직업(예: politician)은 ORDER BY가 그 집합 전체를 훑어 WDQS 60초
    # 제한(504)에 걸린다. 세 가지로 비용을 낮춘다:
    #   1) sitelink 문턱을 서버측 FILTER로 밀어 넣어 정렬 대상 집합을 먼저 줄인다
    #      (대다수 인물은 sitelink가 적어 이 필터 하나로 후보가 크게 줄어든다).
    #   2) 정렬·상한은 ?person/?sitelinks 두 컬럼만 다루는 서브쿼리에서 끝내고,
    #      라벨·문서 URL 같은 무거운 OPTIONAL 조인은 상위 N명에 대해서만 바깥에서 푼다.
    #   3) P106을 VALUES 세트가 아니라 '단일 바운드'(wdt:P106 wd:{qid})로 박는다 —
    #      여러 QID를 VALUES로 묶으면 P106 인덱스를 못 타 같은 규모라도 504가 난다.
    #      그래서 직업군에 QID가 여러 개면 QID마다 쿼리를 따로 돌려 상위에서 병합한다.
    # en/ko 문서 요구(FILTER BOUND)로 상위권 일부가 탈락할 수 있어 서브쿼리에서 3배
    # 과다-추출한 뒤 바깥에서 limit으로 자른다.
    inner_limit = limit * 3
    return f"""
SELECT ?person ?labelEn ?labelKo ?enArticle ?koArticle ?sitelinks ?birth WHERE {{
  {{
    SELECT ?person ?sitelinks WHERE {{
      ?person wdt:P31 wd:Q5 ;
              wdt:P106 wd:{qid} ;
              wikibase:sitelinks ?sitelinks .
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
    pageviews API가 요구하는 'article' 파라미터(=타이틀만)로 잘라낸다.

    비ASCII 타이틀(예: ko '버락_오바마')은 URL에서 퍼센트 인코딩된 채로 온다. 여기서
    한 번 디코드해 '순수 타이틀'로 보관해야, pageviews가 요청 시 quote()로 정확히 한 번만
    인코딩한다(디코드를 생략하면 이중 인코딩되어 ko pageviews가 전부 404가 난다)."""
    if not url:
        return None
    if "/wiki/" not in url:
        return None
    return unquote(url.rsplit("/wiki/", 1)[-1])


def _fetch_one(
    qid: str,
    occupation_label: str,
    min_sitelinks: int,
    limit: int,
) -> list[Candidate]:
    """단일 직업 QID 후보 풀을 SPARQL 한 번으로 가져온다."""
    query = _build_query(qid, min_sitelinks, limit)
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
                occupation=occupation_label,
            )
        )
    return out


def fetch_candidates(
    occupation: str,
    min_sitelinks: int = DEFAULT_MIN_SITELINKS,
    limit: int = DEFAULT_LIMIT,
) -> list[Candidate]:
    """occupation(=OCCUPATION_QIDS 키, 'all'이면 전 직업군) 후보 풀을 SPARQL로 가져온다.

    거대 UNION 한 방으로 뽑지 않는다 — 여러 직업 QID를 묶어 ORDER BY 하면 WDQS 60초
    제한에 걸린다(_build_query 참고). 대신 직업 QID마다 (검증된) 가벼운 쿼리를 각각 돌려
    병합한다. limit은 'QID당' 상한으로 동작하며, 한 인물이 여러 직업을 가지면 sitelink가
    가장 큰 쪽으로 중복 제거한다(occupation 라벨도 그때의 직업으로 확정).

    개별 QID 쿼리가 재시도까지 소진하고 실패하면(드문 타임아웃) 그 QID만 건너뛰고 경고를
    남긴다 — 한 직업의 실패로 'all' 전체가 죽지 않게 한다.
    """
    if occupation == "all":
        groups = list(OCCUPATION_QIDS.items())
    else:
        groups = [(occupation, OCCUPATION_QIDS[occupation])]

    merged: dict[str, Candidate] = {}
    first = True
    for name, qids in groups:
        for qid in qids:
            if not first:
                time.sleep(1.0)  # WDQS 예의상 간격 — 연속 쿼리 스로틀 완화
            first = False
            try:
                found = _fetch_one(qid, name, min_sitelinks, limit)
            except httpx.HTTPError as exc:
                print(f"  ! {name}({qid}) 조회 실패 — 건너뜀: {exc}", file=sys.stderr)
                continue
            for c in found:
                existing = merged.get(c.qid)
                if existing is None or c.sitelinks > existing.sitelinks:
                    merged[c.qid] = c
    return sorted(merged.values(), key=lambda c: c.sitelinks, reverse=True)
