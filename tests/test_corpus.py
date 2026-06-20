"""corpus.py — Notion 파싱, 타임스탬프 결정, 정리/스킵 규칙."""
import datetime as dt

from sies.corpus import (
    Document,
    _clean_title,
    _parse_date,
    _split_notion,
    iter_corpus,
    load_document,
)

NOTION_DOC = """# 메주 냄새

회차: 31
주제/키워드: 냄새
작성자: 이정호
모임일자: 2026년 3월 29일

첫 문단. 외할머니 이야기로 시작한다.

둘째 문단. 메주 냄새가 온 방을 잠식했다.
"""


# ── _parse_date ──────────────────────────────────────────────
def test_parse_korean_date():
    assert _parse_date("모임일자: 2026년 3월 29일") == dt.date(2026, 3, 29)


def test_parse_iso_date():
    assert _parse_date("2019-03-01-제목") == dt.date(2019, 3, 1)
    assert _parse_date("2019.03.01") == dt.date(2019, 3, 1)


def test_parse_english_date():
    assert _parse_date("May 30. 2026") == dt.date(2026, 5, 30)
    assert _parse_date("Jun 2 2026") == dt.date(2026, 6, 2)
    assert _parse_date("September 3rd, 2019") == dt.date(2019, 9, 3)
    assert _parse_date("December 25 2020") == dt.date(2020, 12, 25)


def test_parse_english_date_ignores_non_month_words():
    # 월 이름이 아닌 단어 + 숫자는 날짜가 아니다
    assert _parse_date("Section 12 2026") is None
    assert _parse_date("chapter 3 2019") is None


def test_parse_date_none_when_absent():
    assert _parse_date("날짜 없는 텍스트") is None


def test_parse_invalid_date_returns_none():
    # 13월은 존재하지 않음 → None (크래시 아님)
    assert _parse_date("2026년 13월 40일") is None


# ── _split_notion ────────────────────────────────────────────
def test_split_notion_separates_title_meta_body():
    title, meta, body = _split_notion(NOTION_DOC)
    assert title == "메주 냄새"
    assert meta["회차"] == "31"
    assert meta["주제/키워드"] == "냄새"
    assert meta["모임일자"] == "2026년 3월 29일"
    # 본문은 메타 헤더를 포함하지 않는다
    assert "회차" not in body
    assert body.startswith("첫 문단")
    assert "둘째 문단" in body


def test_split_notion_no_meta():
    title, meta, body = _split_notion("# 제목만\n\n본문 한 줄.")
    assert title == "제목만"
    assert meta == {}
    assert body == "본문 한 줄."


def test_split_notion_no_h1():
    title, meta, body = _split_notion("머리말 없이 그냥 본문.")
    assert title == ""
    assert "그냥 본문" in body


# ── _clean_title ─────────────────────────────────────────────
def test_clean_title_strips_notion_hash(tmp_path):
    p = tmp_path / "메주 냄새 32f0ddfea9618061b203fe26349937cd.md"
    p.write_text("x")
    assert _clean_title("", p) == "메주 냄새"


def test_clean_title_prefers_h1(tmp_path):
    p = tmp_path / "whatever 32f0ddfea9618061b203fe26349937cd.md"
    p.write_text("x")
    assert _clean_title("진짜 제목", p) == "진짜 제목"


# ── load_document: 타임스탬프 결정 분기 ───────────────────────
def test_timestamp_from_content(tmp_path):
    p = tmp_path / "essay 32f0ddfea9618061b203fe26349937cd.md"
    p.write_text(NOTION_DOC, encoding="utf-8")
    doc = load_document(p)
    assert doc.timestamp == dt.date(2026, 3, 29)
    assert doc.timestamp_origin == "content"


def test_timestamp_from_filename(tmp_path):
    p = tmp_path / "2019-03-01-옛글.md"
    p.write_text("# 옛글\n\n본문.", encoding="utf-8")
    doc = load_document(p)
    assert doc.timestamp == dt.date(2019, 3, 1)
    assert doc.timestamp_origin == "filename"


def test_timestamp_falls_back_to_mtime(tmp_path):
    p = tmp_path / "no-date.md"
    p.write_text("# 무제\n\n본문.", encoding="utf-8")
    doc = load_document(p)
    assert doc.timestamp_origin == "mtime"
    assert doc.timestamp is not None


def test_timestamp_from_body_head_english_date(tmp_path):
    # 시처럼 제목/작성자/날짜 헤더가 본문 머리에 있는 경우
    p = tmp_path / "릴케.txt"
    p.write_text("릴케\nby 돌멩이\nJun 2. 2026\n\n사랑이란 두 고독이…", encoding="utf-8")
    doc = load_document(p)
    assert doc.timestamp == dt.date(2026, 6, 2)
    assert doc.timestamp_origin == "content"


def test_body_date_not_scanned_deep(tmp_path):
    # 머리(_HEAD_LINES) 한참 뒤의 산문 속 날짜는 무시 → mtime로
    body_lines = ["줄 하나"] * 10 + ["2019년 3월 1일에 있었던 일"]
    p = tmp_path / "essay.md"
    p.write_text("# 제목\n\n" + "\n".join(body_lines), encoding="utf-8")
    doc = load_document(p)
    assert doc.timestamp_origin == "mtime"


def test_content_date_wins_over_filename(tmp_path):
    # 파일명에도 날짜가 있지만 내용의 모임일자가 우선
    p = tmp_path / "2019-01-01-제목.md"
    p.write_text(NOTION_DOC, encoding="utf-8")
    doc = load_document(p)
    assert doc.timestamp == dt.date(2026, 3, 29)
    assert doc.timestamp_origin == "content"


# ── iter_corpus: 스킵 규칙 ────────────────────────────────────
def test_iter_corpus_skips_junk(tmp_path):
    (tmp_path / "essays").mkdir()
    (tmp_path / "essays" / "글.md").write_text("# 글\n\n본문.", encoding="utf-8")
    (tmp_path / "essays" / ".gitkeep").write_text("")
    (tmp_path / "essays" / "README.md").write_text("# 안내")
    (tmp_path / "essays" / "글.md:Zone.Identifier").write_text("[ZoneTransfer]")
    (tmp_path / "essays" / "그림.png").write_text("binary")

    docs = iter_corpus(tmp_path)
    titles = [d.title for d in docs]
    assert titles == ["글"]  # .gitkeep, README, Zone.Identifier, png 모두 제외


def test_iter_corpus_records_source_folder(tmp_path):
    (tmp_path / "poems").mkdir()
    (tmp_path / "poems" / "시.md").write_text("# 시\n\n한 줄.", encoding="utf-8")
    docs = iter_corpus(tmp_path)
    assert docs[0].source == "poems"


def test_load_document_returns_dataclass(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("# X\n\n본문.", encoding="utf-8")
    assert isinstance(load_document(p), Document)
