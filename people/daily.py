"""일간 쓰레드 업로드 CLI — prepare(다음 인물+작성지시) / publish(검증+업로드).

Claude Code 루틴이 이 두 명령 사이에서 서사·포스트를 작성한다(people/ROUTINE.md).
LLM 호출 없음 — Anthropic API 키 불필요, Threads 토큰만 있으면 실발행.

사용:
    uv run python -m people.daily prepare [--qid Qxxx]
    uv run python -m people.daily publish [--qid Qxxx] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from people.threads import POST_CHAR_LIMIT, ThreadsClient, validate_posts

ENRICHED = Path("people_out/enriched.json")
STORIES = Path("people_out/stories.json")
STATE = Path("people_out/publish_state.json")
THREADS_DIR = Path("people_out/threads")


def load_state(path: Path = STATE) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"pending": None, "posted": {}}


def save_state(state: dict, path: Path = STATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2))


def pick_next(state: dict, ordered_qids: list[str]) -> str | None:
    """pending이 있으면 그것부터(어제 못 끝낸 것), 없으면 랭킹 순서상 첫 미발행."""
    if state.get("pending") and state["pending"] not in state["posted"]:
        return state["pending"]
    for qid in ordered_qids:
        if qid not in state["posted"]:
            return qid
    return None


def _load_entry(qid: str) -> dict:
    enriched = json.load(open(ENRICHED))
    by_qid = {e["qid"]: e for e in enriched}
    if qid not in by_qid:
        sys.exit(f"오류: {qid}가 {ENRICHED}에 없다 — enrich를 먼저 실행하라")
    return by_qid[qid]


def cmd_prepare(args) -> None:
    state = load_state()
    enriched = json.load(open(ENRICHED))
    qid = args.qid or pick_next(state, [e["qid"] for e in enriched])
    if qid is None:
        sys.exit("큐 소진 — enrich로 다음 배치를 채우거나 --qid로 지정하라")
    e = _load_entry(qid)
    stories = json.loads(STORIES.read_text()) if STORIES.exists() else {}
    story = stories.get(qid)
    name = e.get("name_ko") or e.get("name_en") or qid

    state["pending"] = qid
    save_state(state)

    posts_path = THREADS_DIR / f"{qid}.json"
    print(f"=== 오늘의 인물: {name} ({qid}) ===\n")
    if story:
        print("[서사 있음 — stories.json에서 로드]")
        print(json.dumps(story, ensure_ascii=False, indent=1))
    else:
        print("[서사 없음] 먼저 people_out/prompts/_SYSTEM.txt 원칙과 "
              f"people_out/prompts/{qid}.txt 팩트로 서사를 작성해 "
              f"{STORIES}의 \"{qid}\" 키에 저장하라(스키마: prompts/_SCHEMA.json).")
        print(f"\n<팩트 요약>\n발췌: {(e.get('summary') or '')[:300]}\n"
              f"정점: {e.get('peak_month')} 월 {e.get('peak_views'):,}회 / "
              f"최근 월평균 {e.get('recent_monthly_views', 0):,.0f}회\n"
              f"정점 사유: {(e.get('peak_reason') or {}).get('detail') or '미상'}")
    print(f"""
=== 쓰레드 포스트 작성 지시 ===
위 서사를 바탕으로 Threads 글타래 포스트를 작성해 {posts_path} 에 저장하라.
형식: {{"posts": ["...", "..."]}} — UTF-8 JSON.

규칙:
- 4~6개 포스트, 각 450자 이하(하드 한도 {POST_CHAR_LIMIT}자, 넘버링 포함 길이).
- 각 포스트 끝에 (n/총) 넘버링.
- 1번: 훅 — 스크롤을 멈추게. 사진이 함께 첨부된다는 전제.
- 중간: 서사 → 개념 렌즈 → 일하는 사람에게 (카드의 보이스 그대로: 재치 있는 해요체).
- 마지막: 오늘 바꿔볼 한 가지 + 위키 링크({e.get('wiki_ko') or e.get('wiki_en')}).
- 팩트 규율 유지: 카드/팩트에 없는 사실 추가 금지.

작성 후: uv run python -m people.daily publish
""")


def cmd_publish(args) -> None:
    state = load_state()
    qid = args.qid or state.get("pending")
    if not qid:
        sys.exit("발행 대상 없음 — prepare를 먼저 실행하거나 --qid를 지정하라")
    e = _load_entry(qid)
    posts_path = THREADS_DIR / f"{qid}.json"
    if not posts_path.exists():
        sys.exit(f"오류: {posts_path} 없음 — prepare의 지시대로 포스트를 먼저 작성하라")
    posts = validate_posts(json.loads(posts_path.read_text())["posts"])
    name = e.get("name_ko") or e.get("name_en") or qid
    image_url = e.get("image_url")

    client = None if args.dry_run else ThreadsClient.from_env()
    if client is None:
        mode = "dry-run" if args.dry_run else "THREADS_ACCESS_TOKEN 없음 → dry-run"
        print(f"[{mode}] {name} ({qid}) — 포스트 {len(posts)}개, 사진 {'있음' if image_url else '없음'}\n")
        for i, p in enumerate(posts, 1):
            print(f"--- 포스트 {i} ({len(p)}자) ---\n{p}\n")
        print("(실발행 안 됨 — 상태 미변경)")
        return

    print(f"{name} ({qid}) 발행 중 — 포스트 {len(posts)}개...")
    ids = client.publish_thread(posts, image_url=image_url)
    state["posted"][qid] = {"date": dt.date.today().isoformat(), "post_ids": ids}
    state["pending"] = None
    save_state(state)
    print(f"완료 — 루트 포스트 id {ids[0]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="일간 쓰레드 업로드")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prepare", help="다음 인물 선택 + 작성 지시 출력")
    p1.add_argument("--qid")
    p1.set_defaults(func=cmd_prepare)
    p2 = sub.add_parser("publish", help="포스트 검증 + Threads 업로드")
    p2.add_argument("--qid")
    p2.add_argument("--dry-run", action="store_true")
    p2.set_defaults(func=cmd_publish)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
