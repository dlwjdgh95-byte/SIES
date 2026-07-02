"""enrich.py 순수함수 단위테스트 — 네트워크 없음."""
from people.enrich import (
    _claim_time_month,
    _months_apart,
    _title_from_url,
    extract_facts,
    peak_reason,
)


# ── 시간 파싱 ────────────────────────────────────────────────
def test_claim_time_month():
    snak = {"datavalue": {"value": {"time": "+2016-02-19T00:00:00Z"}}}
    assert _claim_time_month(snak) == "2016-02"


def test_claim_time_month_malformed():
    assert _claim_time_month({}) is None
    assert _claim_time_month({"datavalue": {"value": {"time": "somevalue"}}}) is None


def test_months_apart():
    assert _months_apart("2016-02", "2016-02") == 0
    assert _months_apart("2016-02", "2016-03") == 1
    assert _months_apart("2015-12", "2016-01") == 1
    assert _months_apart("2016-01", "2018-01") == 24


# ── 정점 사유 휴리스틱 ────────────────────────────────────────
def test_peak_reason_death_match():
    r = peak_reason("2016-02", "2016-02", [], {})
    assert r["type"] == "death"


def test_peak_reason_death_adjacent_month():
    r = peak_reason("2016-03", "2016-02", [], {})
    assert r["type"] == "death"


def test_peak_reason_award_match():
    r = peak_reason("2015-10", None, [("Q35637", "2015-10")], {"Q35637": "노벨 문학상"})
    assert r["type"] == "award"
    assert "노벨 문학상" in r["detail"]


def test_peak_reason_death_beats_award():
    r = peak_reason("2016-02", "2016-02", [("Q35637", "2016-02")], {})
    assert r["type"] == "death"


def test_peak_reason_no_match():
    r = peak_reason("2016-03", "2000-11", [("Q35637", "1990-01")], {})
    assert r["type"] == "unknown"


def test_peak_reason_no_peak():
    assert peak_reason(None, "2016-02", [], {})["type"] == "unknown"


# ── 클레임 추출 ──────────────────────────────────────────────
def test_extract_facts():
    claims = {
        "P18": [{"mainsnak": {"datavalue": {"value": "Foo.jpg"}}}],
        "P570": [{"mainsnak": {"datavalue": {"value": {"time": "+2003-11-10T00:00:00Z"}}}}],
        "P166": [{
            "mainsnak": {"datavalue": {"value": {"id": "Q35637"}}},
            "qualifiers": {"P585": [{"datavalue": {"value": {"time": "+2015-10-08T00:00:00Z"}}}]},
        }],
    }
    f = extract_facts(claims)
    assert f["image"] == "Foo.jpg"
    assert f["death_month"] == "2003-11"
    assert f["awards"] == [("Q35637", "2015-10")]


def test_extract_facts_empty():
    f = extract_facts({})
    assert f == {"image": None, "death_month": None, "awards": []}


def test_title_from_url():
    assert _title_from_url("https://ko.wikipedia.org/wiki/%EC%B9%B4%EB%82%9C") == "카난"
    assert _title_from_url("https://ko.wikipedia.org/wiki/카난_바나나") == "카난_바나나"
    assert _title_from_url(None) is None
    assert _title_from_url("https://example.com/nope") is None
