"""CLI 공용 소품 — 미리보기·퍼센트 포맷·JSONL 로그 읽기.

search/ab/bench/stats/replay가 각자 들고 있던 사본을 한 곳으로 모았다.
"""
from __future__ import annotations

import json
import re

_SENT_END = re.compile(r"[.!?…]['\"”’)\]]?\s")


def preview(text: str, n: int = 110) -> str:
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


def fmt_pct(x) -> str:
    return f"{x:.1%}" if isinstance(x, float) else "—"


def read_log(path: str) -> list[dict]:
    """JSONL 세션 로그 → dict 리스트. 없으면 FileNotFoundError 그대로."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]
