"""score.py 순수함수 단위테스트 — test_rank.py와 같은 스타일(고정 fixture, 네트워크 없음)."""
import datetime as dt

from people.pageviews import MonthlyViews
from people.score import peak_month_series, score_candidates
from people.wikidata import Candidate

NOW = dt.date(2026, 6, 20)


def _person(qid: str, occupation: str = "politician") -> Candidate:
    return Candidate(
        qid=qid,
        label_en=qid,
        label_ko=None,
        enwiki_title=qid,
        kowiki_title=None,
        sitelinks=20,
        birth_year=1970,
        occupation=occupation,
    )


def _series(*month_views: tuple[str, int]) -> list[MonthlyViews]:
    return [MonthlyViews(month=m, views=v) for m, v in month_views]


# ── peak_month_series ────────────────────────────────────────
def test_peak_month_series_picks_max():
    views = _series(("2020-01", 100), ("2020-06", 900), ("2021-01", 300))
    month, views_at_peak = peak_month_series(views)
    assert month == "2020-06"
    assert views_at_peak == 900


def test_peak_month_series_tie_picks_earliest():
    views = _series(("2020-01", 500), ("2020-06", 500))
    month, _ = peak_month_series(views)
    assert month == "2020-01"


def test_peak_month_series_empty():
    assert peak_month_series([]) == (None, 0)


# ── score_candidates ─────────────────────────────────────────
def test_score_candidates_empty():
    assert score_candidates({}, [], NOW) == []


def test_recent_peak_is_suppressed_by_activity():
    # 정점이 최근일수록 아직 안 잊힘 → 활성도 높음 → (1-활성도) 작음
    old_peak = _person("old")
    recent_peak = _person("recent")
    series = {
        "old": _series(("2018-01", 5000)),
        "recent": _series(("2026-05", 5000)),  # NOW 한 달 전
    }
    ranked = score_candidates(series, [old_peak, recent_peak], NOW)
    by_qid = {s.candidate.qid: s for s in ranked}
    assert by_qid["recent"].person_activity > by_qid["old"].person_activity


def test_bandpass_suppresses_extremes_within_pool():
    # test_rank.py의 bandpass 테스트와 동형: 정점 조회수가 0~100(만) 고르게 퍼진 11명,
    # 같은 시점에 정점을 찍었다고 두면(활성도 동일) 밴드패스만으로 상/하위가 눌리는지 확인.
    people = [_person(f"p{i}") for i in range(11)]
    series = {f"p{i}": _series(("2019-01", i * 10)) for i in range(11)}  # 0,10,...,100
    ranked = score_candidates(series, people, NOW)
    by_qid = {s.candidate.qid: s for s in ranked}
    assert by_qid["p0"].band_weight < 0.01    # 최하위 눌림
    assert by_qid["p10"].band_weight < 0.01   # 최상위 눌림
    assert by_qid["p7"].band_weight > 0.4     # 밴드 안(70) 후보는 살아남는다


def test_recent_dominant_scores_lower_than_forgotten_midband():
    # 최근에 정점 찍은 압도적 인물은 활성도로, 무명은 밴드패스 하위로 각각 눌리고,
    # '오래전에 적당히 유명했던' 인물이 최고 점수를 받는다 — 여러 후보로 분포를 넓혀 검증.
    people = [_person(f"q{i}") for i in range(9)]
    series = {f"q{i}": _series(("2019-01", (i + 1) * 10_000)) for i in range(8)}
    series["q8"] = _series(("2026-05", 1_000_000))  # 최근 + 압도적 조회수
    ranked = score_candidates(series, people, NOW)
    by_qid = {s.candidate.qid: s for s in ranked}
    assert ranked[0].candidate.qid != "q8"  # 압도적 현역이 1위를 차지하지 않는다


def test_single_candidate_bandpass_passes():
    p = _person("solo")
    series = {"solo": _series(("2019-01", 1000))}
    ranked = score_candidates(series, [p], NOW)
    assert ranked[0].band_weight == 1.0


def test_missing_pageviews_series_scores_near_zero():
    p = _person("no_data")
    ranked = score_candidates({}, [p], NOW)
    assert ranked[0].peak_views == 0
    assert ranked[0].score == 0.0


def test_trend_signal_none_does_not_crash():
    p = _person("no_trend")
    series = {"no_trend": _series(("2019-01", 1000))}
    ranked = score_candidates(series, [p], NOW, trend_signals={"no_trend": None})
    assert ranked[0].trend_signal is None


def test_sorted_descending():
    people = [_person(f"p{i}") for i in range(5)]
    series = {f"p{i}": _series((f"20{18+i}-01", 1000 * (i + 1))) for i in range(5)}
    ranked = score_candidates(series, people, NOW)
    scores = [s.score for s in ranked]
    assert scores == sorted(scores, reverse=True)
