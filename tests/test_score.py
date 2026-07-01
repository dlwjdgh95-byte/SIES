"""score.py 순수함수 단위테스트 — test_rank.py와 같은 스타일(고정 fixture, 네트워크 없음)."""
import datetime as dt

from people.pageviews import MonthlyViews
from people.score import peak_month_series, recent_mean_views, score_candidates
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
    # (게이트를 꺼야 recent가 결과에 남아 비교 가능)
    old_peak = _person("old")
    recent_peak = _person("recent")
    series = {
        "old": _series(("2018-01", 5000)),
        "recent": _series(("2026-05", 5000)),  # NOW 한 달 전
    }
    ranked = score_candidates(series, [old_peak, recent_peak], NOW, max_activity=None)
    by_qid = {s.candidate.qid: s for s in ranked}
    assert by_qid["recent"].person_activity > by_qid["old"].person_activity


# ── 활성도 게이트 (max_activity) ─────────────────────────────
def test_recent_peak_is_gated_out_by_default():
    # 기본 게이트(0.4)에서는 최근 정점 인물이 아예 결과에서 빠진다.
    series = {
        "old": _series(("2018-01", 5000)),
        "recent": _series(("2026-05", 5000)),
    }
    ranked = score_candidates(series, [_person("old"), _person("recent")], NOW)
    qids = {s.candidate.qid for s in ranked}
    assert "recent" not in qids
    assert "old" in qids


def test_still_viewed_person_is_gated_out():
    # 정점은 오래전이지만 지금도 정점급 조회수(A_now 높음) → '잊힘' 아님 → 제외.
    # 아인슈타인류: A_time만 보면 잊힌 걸로 오판되던 케이스.
    still_viewed = _series(("2016-03", 100_000), ("2026-04", 80_000), ("2026-05", 90_000))
    forgotten = _series(("2016-03", 100_000), ("2026-04", 500), ("2026-05", 400))
    series = {"evergreen": still_viewed, "gone": forgotten}
    ranked = score_candidates(series, [_person("evergreen"), _person("gone")], NOW)
    qids = {s.candidate.qid for s in ranked}
    assert "evergreen" not in qids
    assert "gone" in qids


def test_activity_axes_are_reported():
    series = {"gone": _series(("2016-03", 100_000), ("2026-05", 1_000))}
    ranked = score_candidates(series, [_person("gone")], NOW)
    s = ranked[0]
    assert s.activity_now == 0.01           # 1000/100000
    assert 0.0 < s.activity_time < 0.4      # 10년 전 정점, 반감기 3년
    assert s.person_activity == max(s.activity_time, s.activity_now)
    assert s.recent_views == 1_000


# ── recent_mean_views ────────────────────────────────────────
def test_recent_mean_views_window():
    views = _series(("2018-01", 9_999), ("2025-08", 100), ("2026-02", 300))
    # NOW=2026-06 기준 창은 2025-06부터 — 2018-01은 창 밖.
    assert recent_mean_views(views, NOW) == 200.0


def test_recent_mean_views_empty_window():
    views = _series(("2018-01", 9_999))
    assert recent_mean_views(views, NOW) == 0.0
    assert recent_mean_views([], NOW) == 0.0


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


def test_missing_pageviews_series_is_gated_out():
    # 시계열이 없으면 A_time이 결측 중립값(0.5) → 기본 게이트(0.4)에서 탈락.
    # 데이터가 없으면 '잊힘'을 주장할 근거도 없다는 의도된 동작.
    p = _person("no_data")
    assert score_candidates({}, [p], NOW) == []
    # 게이트를 끄면 종전처럼 0점으로 남는다.
    ranked = score_candidates({}, [p], NOW, max_activity=None)
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
