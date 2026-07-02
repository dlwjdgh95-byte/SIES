"""잊힌 인물 발굴 CLI.

사용:
    uv run python -m people.discover --occupation politician --limit 50
    uv run python -m people.discover --occupation all --limit 200 --out people_out/result.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from people.pageviews import fetch_pageviews
from people.score import MAX_ACTIVITY, PersonScore, score_candidates
from people.trends import fetch_trend_signal
from people.wikidata import DEFAULT_MIN_SITELINKS, OCCUPATION_QIDS, fetch_candidates

CACHE_DIR = Path("people_cache/pageviews")


def _wiki_url(project: str, title: str | None) -> str | None:
    if not title:
        return None
    domain = "en.wikipedia.org" if project == "en" else "ko.wikipedia.org"
    return f"https://{domain}/wiki/{title}"


def _to_json(scores: list[PersonScore]) -> list[dict]:
    out = []
    for s in scores:
        c = s.candidate
        out.append(
            {
                "qid": c.qid,
                "name_en": c.label_en,
                "name_ko": c.label_ko,
                "occupation": c.occupation,
                "score": s.score,
                "peak_prominence": s.peak_prominence,
                "peak_views": s.peak_views,
                "peak_month": s.peak_month,
                "activity": s.person_activity,
                "activity_time": s.activity_time,
                "activity_now": s.activity_now,
                "activity_abs": s.activity_abs,
                "recent_monthly_views": s.recent_views,
                "band_weight": s.band_weight,
                "trend_signal": s.trend_signal,
                "wiki_en": _wiki_url("en", c.enwiki_title),
                "wiki_ko": _wiki_url("ko", c.kowiki_title),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="잊힌 인물 발굴 (feasibility spike)")
    ap.add_argument("--occupation", default="politician",
                     choices=[*OCCUPATION_QIDS.keys(), "all"])
    ap.add_argument("--limit", type=int, default=50, help="후보 풀 상한(SPARQL LIMIT)")
    ap.add_argument("--min-sitelinks", type=int, default=DEFAULT_MIN_SITELINKS)
    ap.add_argument("--max-activity", type=float, default=MAX_ACTIVITY,
                     help="이 활성도 이상(아직 활발)인 후보 제외. 1.01 이상이면 사실상 게이트 해제")
    ap.add_argument("--refresh", action="store_true", help="pageviews 캐시 무시하고 재요청")
    ap.add_argument("--with-trends", action="store_true",
                     help="Google Trends 부가 신호도 조회(느리고 불안정 — 기본 꺼짐)")
    ap.add_argument("--out", help="JSON 저장 경로(생략 시 표준출력에 텍스트 랭킹만)")
    args = ap.parse_args()

    print(f"[1/3] Wikidata 후보 풀 조회 ({args.occupation}, limit={args.limit})...")
    candidates = fetch_candidates(args.occupation, args.min_sitelinks, args.limit)
    print(f"  → 후보 {len(candidates)}명")

    print("[2/3] Wikimedia pageviews 조회...")
    series_by_qid = {}
    for i, c in enumerate(candidates, 1):
        project = "en.wikipedia" if c.enwiki_title else "ko.wikipedia"
        title = c.enwiki_title or c.kowiki_title
        series_by_qid[c.qid] = fetch_pageviews(project, title, CACHE_DIR, refresh=args.refresh)
        if i % 100 == 0:
            print(f"  ... {i}/{len(candidates)}", flush=True)

    trend_signals = None
    if args.with_trends:
        print("[2.5/3] Google Trends 부가 신호(best-effort)...")
        trend_signals = {
            c.qid: fetch_trend_signal(c.label_en or c.label_ko or c.qid) for c in candidates
        }

    print("[3/3] 스코어링...")
    ranked = score_candidates(series_by_qid, candidates, dt.date.today(),
                              trend_signals=trend_signals, max_activity=args.max_activity)
    print(f"  → 활성도 게이트(<{args.max_activity}) 통과 {len(ranked)}명 / 후보 {len(candidates)}명")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(_to_json(ranked), ensure_ascii=False, indent=2)
        )
        print(f"저장: {args.out}")

    print()
    for i, s in enumerate(ranked, 1):
        c = s.candidate
        name = c.label_ko or c.label_en or c.qid
        link = _wiki_url("ko", c.kowiki_title) or _wiki_url("en", c.enwiki_title)
        print(
            f"{i:>2}. 점수 {s.score:.1f} (정점저명 {s.peak_prominence:.2f} · "
            f"활성도 {s.person_activity:.2f}"
            f"[시간{s.activity_time:.2f}/궤적{s.activity_now:.2f}/절대{s.activity_abs:.2f}] · "
            f"밴드 {s.band_weight:.2f} · 최근 {s.recent_views:,.0f}/월) "
            f"{name} [{s.peak_month}] {link}"
        )


if __name__ == "__main__":
    main()
