"""오늘의 잊힌 나 — 결정론적 산수로 잊힌 글 1편을 골라 원문 발췌를 내미는 일간 푸시 CLI.

    점수 = 1 − 활성도,  활성도 = min(A_time, A_vol)  — 산수는 전부 rank.py에서 import.

LLM·ML 없음(원칙 1). 검색과 달리 *질의가 없으므로 관련성·밴드패스는 성립하지 않는다* —
남는 축은 잊힘뿐이라 점수는 (1−활성도) 단독. 동점은 sha1(now+doc_path) 오름차순: 같은
날엔 항상 같은 문서(결정론), 날이 바뀌면 동점끼리 섞인다. 확정(냉각 개시)은 mark 시에만
— prepare 후 세션이 죽어도 다음 prepare가 pending을 재제시한다.
"""
from __future__ import annotations

import argparse, datetime as dt, hashlib, json, os, sys  # noqa: E401 — 줄 예산

import numpy as np

from sies.embed import DEFAULT_MODEL
from sies.rank import LOW_ACTIVITY, activity, newer_counts, volume_activity
from sies.store import connect
from sies.util import read_log

COOLDOWN_DAYS = 90  # 재노출 냉각(일). 점수 상수가 아니라 노출 정책이므로 rank.py 아닌 여기가 소유.
LOG_PATH = "today_log.jsonl"


def _append(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def pending_pick(events: list[dict]) -> dict | None:
    """마지막 pick에 대응하는 mark가 없으면 그 pick 이벤트를 반환(재제시 대상)."""
    for e in reversed(events):  # mark는 항상 자기 pick 뒤에만 쌓인다(append-only)
        if e.get("event") == "mark":
            return None
        if e.get("event") == "pick":
            return e
    return None


def pick_today(conn, now: dt.date, events: list[dict]) -> dict | None:
    """오늘의 top-1 — 잊힌(활성도 < LOW_ACTIVITY) 문서 중 가장 잊힌 것. now는 테스트 주입용."""
    docs = [{"doc_path": p, "title": t, "timestamp": ts} for p, t, ts in conn.execute(
        "SELECT doc_path, MAX(title), MAX(timestamp) FROM chunks GROUP BY doc_path")]
    counts = newer_counts(docs)
    cooled = {e["doc_path"] for e in events if e.get("event") == "pick"
              and (now - dt.date.fromisoformat(e["ts"])).days < COOLDOWN_DAYS}
    cands = []
    for d in docs:
        ts = dt.date.fromisoformat(d["timestamp"]) if d["timestamp"] else None
        act = min(activity(ts, now), volume_activity(counts[d["doc_path"]]))
        if act < LOW_ACTIVITY and d["doc_path"] not in cooled:
            cands.append((1.0 - act, d))
    cands.sort(key=lambda c: (-c[0], hashlib.sha1(f"{now}{c[1]['doc_path']}".encode()).hexdigest()))
    return cands[0][1] if cands else None


def representative_chunk(conn, doc_path: str, model: str = DEFAULT_MODEL) -> str:
    """청크 임베딩(단위벡터) 평균에 코사인 최근접인 대표 청크의 원문 — 요약 아님."""
    try:
        rows = conn.execute(
            f"SELECT c.text, v.embedding FROM chunks c JOIN vec_{model.replace('-', '_')} v "
            "ON v.rowid = c.id WHERE c.doc_path = ?", (doc_path,)).fetchall()
        vecs = np.stack([np.frombuffer(b, dtype=np.float32) for _, b in rows])
        center = vecs.mean(axis=0)
        return rows[int(np.argmax(vecs @ (center / np.linalg.norm(center))))][0]  # 단위벡터라 내적=코사인
    except Exception:  # 벡터 부재·테이블 없음 — 발췌가 없는 것보단 첫 청크가 낫다
        return conn.execute("SELECT text FROM chunks WHERE doc_path = ? ORDER BY chunk_index "
                            "LIMIT 1", (doc_path,)).fetchone()[0]


def cmd_prepare(args) -> None:
    conn = connect(args.db)
    events = read_log(args.log) if os.path.exists(args.log) else []
    now = dt.date.today()
    doc = pending_pick(events)
    if doc is None:
        doc = pick_today(conn, now, events)
        if doc is None:
            sys.exit("후보 없음 — 모든 문서가 활성이거나 냉각 중이다. 오늘은 쉼.")
        _append(args.log, {"ts": now.isoformat(), "event": "pick",
                           "doc_path": doc["doc_path"], "title": doc["title"]})
    ts = conn.execute("SELECT MAX(timestamp) FROM chunks WHERE doc_path = ?",
                      (doc["doc_path"],)).fetchone()[0]
    print(f"=== 오늘의 잊힌 나: {doc['title']} ===\n작성일: {ts or '미상'} / 경로: {doc['doc_path']}\n\n"
          + representative_chunk(conn, doc["doc_path"], args.model)
          + "\n\n=== 지시 ===\n이 발췌를 원문 그대로 보여줘 — 수정·요약 금지, 한 줄 해설만 허용.\n"
            "반응 기록: uv run python -m sies.today mark --read [--good|--bad] 또는 --skip")


def cmd_mark(args) -> None:
    pend = pending_pick(read_log(args.log) if os.path.exists(args.log) else [])
    if pend is None:
        sys.exit("pending 없음 — prepare를 먼저 실행하라")
    _append(args.log, {"ts": dt.date.today().isoformat(), "event": "mark",
                       "doc_path": pend["doc_path"], "status": "read" if args.read else "skip",
                       "judge": "good" if args.good else ("bad" if args.bad else None)})
    print(f"기록됨: {pend['title']} → {'read' if args.read else 'skip'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="오늘의 잊힌 나")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prepare", help="오늘의 문서 선택 + 발췌 출력")
    p1.add_argument("--db", default="sies.db")
    p1.add_argument("--model", default=DEFAULT_MODEL)
    p2 = sub.add_parser("mark", help="반응 기록(확정) — 여기서만 냉각이 개시된다")
    for flag in ("--read", "--skip", "--good", "--bad"):
        p2.add_argument(flag, action="store_true")
    for p, f in ((p1, cmd_prepare), (p2, cmd_mark)):
        p.add_argument("--log", default=LOG_PATH)
        p.set_defaults(func=f)
    args = ap.parse_args()
    if args.cmd == "mark" and (args.read == args.skip or (args.good and args.bad)):
        ap.error("--read/--skip 중 정확히 하나, --good/--bad는 동시 지정 불가")
    args.func(args)


if __name__ == "__main__":
    main()
