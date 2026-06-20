"""인덱싱 텍스트 정제 — PDF 띄어쓰기 교정 + 앞머리 마커 제거.

검색 결정엔 안 닿는 *인덱싱 전처리*다(원칙 2와 무관).
- 띄어쓰기: PDF 추출이 단어 중간에 끼워 넣은 공백만 제거. Kiwi 재교정 결과와
  교집합을 취해 '제거'만 채택하고 '분할'은 버린다 → 고유명사(강명국 등) 보존.
- 마커: 청크 맨 앞의 'Q3, Q4', '8.', '2)' 같은 열거/질문 번호 제거.
"""
from __future__ import annotations

import re
from functools import lru_cache

# 앞머리 열거/질문 마커: "Q3, Q4 ", "Q5. ", "8. ", "2) " 등(한 번만)
_MARKER = re.compile(r"^\s*(?:Q\s*\d+\s*[.,)]?\s*)+|^\s*\d+\s*[.)]\s*")


def strip_leading_marker(text: str) -> str:
    """청크 맨 앞의 열거/질문 번호를 한 번 제거."""
    return _MARKER.sub("", text, count=1)


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def _space_pos(s: str) -> tuple[str, set[int]]:
    """공백 제거 문자열 + '각 문자 뒤에 공백이 오는' 인덱스 집합."""
    chars: list[str] = []
    pos: set[int] = set()
    idx = -1
    for ch in s:
        if ch == " ":
            if idx >= 0:
                pos.add(idx)
        else:
            chars.append(ch)
            idx += 1
    return "".join(chars), pos


def _fix_line(line: str) -> str:
    """한 줄의 '잘못 끼인 공백'만 제거(추가 안 함) — 고유명사 보존."""
    if " " not in line:
        return line
    fixed = _kiwi().space(line, reset_whitespace=True)
    a, ap = _space_pos(line)
    b, bp = _space_pos(fixed)
    if a != b:               # 글자가 달라지면(이론상 X) 안전하게 원문 유지
        return line
    keep = ap & bp           # 원문에도 있고 Kiwi도 남긴 공백만 = 제거만 채택
    out: list[str] = []
    for i, ch in enumerate(a):
        out.append(ch)
        if i in keep:
            out.append(" ")
    return "".join(out)


def fix_spacing(text: str) -> str:
    """줄 단위로 PDF 추출 띄어쓰기 오류를 교정(문단/줄 구조 보존)."""
    return "\n".join(_fix_line(ln) for ln in text.split("\n"))
