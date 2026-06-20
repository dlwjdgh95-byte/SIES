"""코퍼스 로딩 — corpus/ 아래 글을 읽어 Document로 정규화한다.

지원 형식: .md/.txt(직접), .pdf(pypdf), .hwp(pyhwp), .docx(python-docx). 발굴된 옛 글은 PDF·HWP가 많다.

Notion export 형식을 처리한다:
    # 제목
    회차: 31
    주제/키워드: 냄새
    작성자: 이정호
    모임일자: 2026년 3월 29일

    <본문 문단들...>

타임스탬프 우선순위:
  (1) 내용 메타 날짜('key: value' 헤더) → (2) 본문 머리의 날짜 헤더(시 등) →
  (3) 파일명의 날짜 → (4) 파일 수정일자.
날짜는 한국어(2026년 3월 29일)·ISO(2026-03-29)·영어(May 30. 2026) 형식을 인식한다.
(PDF·HWP 발굴 글은 보통 날짜가 없어 수정일자 = 진짜 옛 날짜로 떨어진다.)
"""
from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

TEXT_SUFFIXES = (".md", ".markdown", ".txt")
BINARY_SUFFIXES = (".pdf", ".hwp", ".docx")

# 무시할 파일들: Notion 해시 잔재가 아니라 진짜 쓰레기/구조 파일
SKIP_NAMES = {".gitkeep", "README.md"}
SKIP_SUFFIXES = ("Zone.Identifier",)  # Windows 다운로드 ADS 잔재

# 내용 메타에서 날짜를 담는 키 후보
DATE_KEYS = ("모임일자", "작성일자", "작성일", "날짜", "date", "Date")

# 한국어/숫자/영어 날짜 패턴
_KO_DATE = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_ISO_DATE = re.compile(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})")
# 영어 월 이름: "May 30. 2026", "Jun 2 2026", "September 3rd, 2019"
_EN_MONTHS = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"], start=1)
}
_EN_DATE = re.compile(
    r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?\.?,?\s+(\d{4})\b"
)

# 본문 머리에서 날짜 헤더를 찾을 때 훑는 줄 수(제목/작성자/날짜가 사는 영역)
_HEAD_LINES = 5


@dataclass
class Document:
    path: Path
    source: str               # 상위 폴더명 (brunch/essays/poems/...)
    title: str
    body: str                 # 메타 헤더를 걷어낸 본문
    meta: dict[str, str] = field(default_factory=dict)
    timestamp: dt.date | None = None
    timestamp_origin: str = "mtime"  # content | filename | mtime


# 한글(.hwp) 추출 시 끼는 구조 마커 — 단독 줄이면 잡음이므로 제거
_HWP_MARKERS = re.compile(r"^\s*<(표|그림|개체|미주|각주)>\s*$")


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    from .normalize import fix_spacing

    reader = PdfReader(str(path))
    raw = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return fix_spacing(raw)  # PDF 추출 띄어쓰기 오류 교정(이 포맷에서만 발생)


def _extract_hwp(path: Path) -> str:
    """pyhwp의 hwp5txt로 텍스트 추출. venv의 콘솔 스크립트를 직접 호출한다."""
    hwp5txt = os.path.join(os.path.dirname(sys.executable), "hwp5txt")
    if not os.path.exists(hwp5txt):
        hwp5txt = "hwp5txt"  # PATH에 있길 기대
    out = subprocess.run(
        [hwp5txt, str(path)], capture_output=True, text=True, check=True
    )
    lines = [ln for ln in out.stdout.splitlines() if not _HWP_MARKERS.match(ln)]
    return "\n".join(lines)


def _extract_docx(path: Path) -> str:
    """python-docx로 문단 텍스트 추출. 빈 문단은 빈 줄로 남겨 문단 청킹을 돕는다."""
    from docx import Document as _Docx

    return "\n".join(p.text for p in _Docx(str(path)).paragraphs)


def _read_raw(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".hwp":
        return _extract_hwp(path)
    if suffix == ".docx":
        return _extract_docx(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_date(text: str) -> dt.date | None:
    m = _KO_DATE.search(text) or _ISO_DATE.search(text)
    if m:
        y, mo, d = (int(g) for g in m.groups())
    else:
        em = _EN_DATE.search(text)
        if not em:
            return None
        mo = _EN_MONTHS.get(em.group(1).lower()[:3])
        if not mo:  # 월 이름이 아닌 단어(예: "Section 12 2026") → 무시
            return None
        d, y = int(em.group(2)), int(em.group(3))
    try:
        return dt.date(y, mo, d)
    except ValueError:
        return None


def _split_notion(raw: str) -> tuple[str, dict[str, str], str]:
    """(title, meta, body)로 분해한다. Notion export 관용 형식 가정."""
    lines = raw.splitlines()
    i = 0
    title = ""
    # 1) 첫 H1을 제목으로
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].lstrip().startswith("# "):
        title = lines[i].lstrip()[2:].strip()
        i += 1

    # 2) 빈 줄 건너뛰고, 이어지는 'key: value' 블록을 메타로
    while i < len(lines) and not lines[i].strip():
        i += 1
    meta: dict[str, str] = {}
    meta_re = re.compile(r"^([^:#\n]{1,30}):\s*(.+)$")
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            # 메타 블록이 시작됐다면 빈 줄에서 끝
            if meta:
                i += 1
                break
            i += 1
            continue
        m = meta_re.match(line.strip())
        if m:
            meta[m.group(1).strip()] = m.group(2).strip()
            i += 1
        else:
            break

    body = "\n".join(lines[i:]).strip()
    return title, meta, body


def _resolve_timestamp(
    path: Path, meta: dict[str, str], body: str = ""
) -> tuple[dt.date, str]:
    # (1) 내용 메타의 날짜 (Notion 'key: value' 헤더)
    for key in DATE_KEYS:
        if key in meta:
            d = _parse_date(meta[key])
            if d:
                return d, "content"
    # (2) 본문 머리의 날짜 헤더 (시처럼 '제목/작성자/날짜' 줄을 가진 글)
    head = "\n".join([ln for ln in body.splitlines() if ln.strip()][:_HEAD_LINES])
    d = _parse_date(head)
    if d:
        return d, "content"
    # (3) 파일명의 날짜
    d = _parse_date(path.stem)
    if d:
        return d, "filename"
    # (4) 파일 수정일자
    return dt.date.fromtimestamp(path.stat().st_mtime), "mtime"


def _clean_title(title: str, path: Path) -> str:
    if title:
        return title
    # H1이 없으면 파일명에서 Notion 해시(끝의 32자리 hex)를 떼고 사용
    stem = re.sub(r"\s+[0-9a-f]{32}$", "", path.stem)
    return stem.strip()


def load_document(path: Path) -> Document:
    raw = _read_raw(path)
    title, meta, body = _split_notion(raw)
    title = _clean_title(title, path)
    if not body:
        body = raw.strip()
    ts, origin = _resolve_timestamp(path, meta, body)
    return Document(
        path=path,
        source=path.parent.name,
        title=title,
        body=body,
        meta=meta,
        timestamp=ts,
        timestamp_origin=origin,
    )


def iter_corpus(root: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in SKIP_NAMES:
            continue
        if path.name.endswith(SKIP_SUFFIXES):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES + BINARY_SUFFIXES:
            continue
        docs.append(load_document(path))
    return docs


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("corpus")
    docs = iter_corpus(root)
    print(f"{len(docs)}개 문서\n")
    for d in docs:
        print(f"[{d.source}] {d.title}")
        print(f"    날짜: {d.timestamp} ({d.timestamp_origin})  본문 {len(d.body)}자")
        if d.meta:
            print(f"    메타: {d.meta}")
