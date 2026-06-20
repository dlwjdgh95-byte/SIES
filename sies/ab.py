"""A/B 하니스 — 베이스라인 vs 역전, 블라인드 판정 + JSONL 로그.

킬 테스트: 한 달 로그에서 역전의 '적중률'이 베이스라인을 이기는가(판정자는 나).
편향을 줄이려 두 방식의 후보를 *섞어 출처를 가린 채* 판정하고, 결과를 로그에 쌓는다.

사용:
    uv run python -m sies.ab "관성에 대하여" --judge      # 판정까지
    uv run python -m sies.ab "관성에 대하여"              # 결과만 로그(판정 보류)
집계:
    uv run python -m sies.stats
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
from pathlib import Path

from .embed import DEFAULT_MODEL, MODELS, get_embedder
from .rank import DEFAULT_HALF_LIFE, rank_baseline, rank_inverted
from .retrieve import candidate_pool
from .store import connect

DEFAULT_LOG = "search_log.jsonl"


def candidate_key(c: dict) -> str:
    return str(c["id"])


def _brief(scored, inverted: bool, activity: float | None = None) -> dict:
    c = scored.candidate
    d = {
        "id": c["id"],
        "title": c["title"],
        "chunk_index": c["chunk_index"],
        "timestamp": c["timestamp"],
        "similarity": round(scored.similarity, 4),
    }
    # 활성도는 method 무관 '잊힘도' — 적중 분석용으로 베이스라인 후보에도 심는다.
    act = scored.activity if inverted else activity
    if act is not None and act == act:  # NaN 제외
        d["activity"] = round(act, 4)
    if inverted:
        d["activity_time"] = round(scored.activity_time, 4)
        d["activity_vol"] = round(scored.activity_vol, 4)
        d["volume"] = scored.volume
        d["score"] = round(scored.score, 4)
        d["band_weight"] = round(scored.band_weight, 4)
    return d


def make_record(query, model, half_life, k, baseline, inverted) -> dict:
    """판정 전 세션 레코드(verdicts 비어 있음)."""
    # 역전은 전체 풀을 점수화하므로 id→활성도 맵으로 베이스라인 후보에도 활성도를 채운다.
    activity_by_id = {s.candidate["id"]: s.activity for s in inverted}
    return {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "model": model,
        "half_life": half_life,
        "k": k,
        "baseline": [_brief(s, False, activity_by_id.get(s.candidate["id"])) for s in baseline[:k]],
        "inversion": [_brief(s, True) for s in inverted[:k]],
        "verdicts": {},
    }


def hit_rate(record: dict, method: str) -> float | None:
    """method('baseline'|'inversion') 후보 중 hit 비율. 판정된 게 없으면 None."""
    ids = [str(c["id"]) for c in record[method]]
    verdicts = record["verdicts"]
    judged = [verdicts[i] for i in ids if verdicts.get(i) in ("hit", "miss")]
    if not judged:
        return None
    return judged.count("hit") / len(judged)


def _union_candidates(record: dict) -> list[dict]:
    """두 방식 후보를 id로 합집합(중복 제거)."""
    seen, union = set(), []
    for c in record["baseline"] + record["inversion"]:
        if str(c["id"]) not in seen:
            seen.add(str(c["id"]))
            union.append(c)
    return union


_SENT_END = re.compile(r"[.!?…]['\"”’)\]]?\s")


def _preview(text: str, n: int = 110) -> str:
    """공백 정규화 후 n자 근처에서 *문장 경계*로 자른다(중간 절단 방지)."""
    t = " ".join(text.split())
    if len(t) <= n:
        return t
    ends = [m.end() for m in _SENT_END.finditer(t + " ")]
    before = [e for e in ends if e <= n]
    if before:                       # n 이내 마지막 문장 끝
        return t[: before[-1]].rstrip()
    if ends:                         # 첫 문장이 n보다 길면 그 문장까지 통째로
        return t[: ends[0]].rstrip()
    return t[:n].rsplit(" ", 1)[0] + "…"  # 문장부호가 없으면 단어 경계


def judge_interactive(record: dict, text_by_id: dict[int, str]) -> None:
    """후보를 섞어 출처를 가린 채 적중/헛것 판정. 결과를 record['verdicts']에 기록."""
    union = _union_candidates(record)
    random.shuffle(union)
    print("\n── 블라인드 판정 ── (h=적중, m=헛것, s=건너뜀, q=중단)\n")
    for c in union:
        cid = str(c["id"])
        print(f"[{c['title']}] {c['timestamp']}")
        print(f"  {_preview(text_by_id.get(c['id'], ''))}")
        ans = input("  적중? [h/m/s/q] ").strip().lower()
        if ans == "q":
            break
        record["verdicts"][cid] = {"h": "hit", "m": "miss"}.get(ans, "skip")
        print()


def append_log(record: dict, path: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _print_side_by_side(record: dict) -> None:
    print(f'질의: "{record["query"]}"  [{record["model"]}]\n')
    print("── 베이스라인(순수 유사도) ──")
    for i, c in enumerate(record["baseline"], 1):
        print(f"{i:>2}. 유사도 {c['similarity']:.3f} [{c['title']}] {c['timestamp']}")
    print("\n── 역전(관련성 × (1−활성도) × 밴드패스) ──")
    for i, c in enumerate(record["inversion"], 1):
        print(f"{i:>2}. 점수 {c['score']:.3f} (유사도 {c['similarity']:.3f} · 활성도 {c['activity']:.2f}) "
              f"[{c['title']}] {c['timestamp']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="SIES A/B 하니스 (베이스라인 vs 역전)")
    ap.add_argument("query")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--db", default="sies.db")
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--half-life", type=float, default=DEFAULT_HALF_LIFE)
    ap.add_argument("--judge", action="store_true", help="블라인드 적중/헛것 판정")
    args = ap.parse_args()

    emb = get_embedder(args.model).load()
    qv = emb.encode([args.query], is_query=True)[0]
    conn = connect(args.db)
    pool = candidate_pool(conn, args.model, qv)
    conn.close()

    baseline = rank_baseline(pool)
    inverted = rank_inverted(pool, dt.date.today(), args.half_life)
    record = make_record(args.query, args.model, args.half_life, args.k, baseline, inverted)

    _print_side_by_side(record)

    if args.judge:
        text_by_id = {s.candidate["id"]: s.candidate["text"] for s in baseline}
        judge_interactive(record, text_by_id)
        br, ir = hit_rate(record, "baseline"), hit_rate(record, "inversion")
        print(f"\n이번 세션 적중률 — 베이스라인 {br}  vs  역전 {ir}")

    append_log(record, args.log)
    print(f"\n로그 추가됨 → {args.log}  (총 {sum(1 for _ in open(args.log, encoding='utf-8'))}세션)")


if __name__ == "__main__":
    main()
