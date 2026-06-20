"""corpus.py 추출 경로 — PDF/HWP 분기, 한글 마커 제거, 수집 통합.

실제 PDF·HWP는 개인 글이라 픽스처로 커밋하지 않는다.
대신 추출 함수를 monkeypatch 해서 라우팅·필터 로직만 검증한다.
"""
import datetime as dt
import types

import sies.corpus as corpus
from sies.corpus import (
    BINARY_SUFFIXES,
    TEXT_SUFFIXES,
    _HWP_MARKERS,
    _read_raw,
    iter_corpus,
    load_document,
)


# ── 한글 구조 마커 정규식 ──────────────────────────────────────
def test_hwp_markers_match_standalone():
    for m in ("<표>", "  <그림> ", "<개체>", "<미주>", "<각주>"):
        assert _HWP_MARKERS.match(m)


def test_hwp_markers_ignore_inline_text():
    # 본문 중 꺾쇠 표현은 마커가 아니다
    assert not _HWP_MARKERS.match("나는 <표>를 보았다")
    assert not _HWP_MARKERS.match("표를 그렸다")


# ── _read_raw 분기 ────────────────────────────────────────────
def test_read_raw_routes_pdf(monkeypatch, tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF-fake")
    monkeypatch.setattr(corpus, "_extract_pdf", lambda path: "PDF 본문")
    assert _read_raw(p) == "PDF 본문"


def test_read_raw_routes_hwp(monkeypatch, tmp_path):
    p = tmp_path / "x.hwp"
    p.write_bytes(b"\xd0\xcf\x11\xe0")  # OLE 매직
    monkeypatch.setattr(corpus, "_extract_hwp", lambda path: "HWP 본문")
    assert _read_raw(p) == "HWP 본문"


def test_read_raw_reads_text_directly(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("# 제목\n\n본문.", encoding="utf-8")
    assert "본문." in _read_raw(p)


# ── _extract_hwp: 마커 제거 (subprocess stub) ─────────────────
def test_extract_hwp_strips_markers(monkeypatch, tmp_path):
    p = tmp_path / "doc.hwp"
    p.write_bytes(b"\xd0\xcf\x11\xe0")
    fake_stdout = "<표>\n\n첫 문단입니다.\n<그림>\n둘째 문단입니다.\n"

    def fake_run(cmd, **kw):
        return types.SimpleNamespace(stdout=fake_stdout, returncode=0)

    monkeypatch.setattr(corpus.subprocess, "run", fake_run)
    out = corpus._extract_hwp(p)
    assert "<표>" not in out
    assert "<그림>" not in out
    assert "첫 문단입니다." in out
    assert "둘째 문단입니다." in out


# ── iter_corpus: 새 형식 수용 ────────────────────────────────
def test_supported_suffix_sets():
    assert ".pdf" in BINARY_SUFFIXES and ".hwp" in BINARY_SUFFIXES
    assert ".md" in TEXT_SUFFIXES and ".txt" in TEXT_SUFFIXES


def test_iter_corpus_includes_pdf_and_hwp(monkeypatch, tmp_path):
    (tmp_path / "essays").mkdir()
    (tmp_path / "essays" / "옛글.pdf").write_bytes(b"%PDF-fake")
    (tmp_path / "essays" / "한글.hwp").write_bytes(b"\xd0\xcf\x11\xe0")
    (tmp_path / "essays" / "노트.md").write_text("# 노트\n\n본문.", encoding="utf-8")

    monkeypatch.setattr(corpus, "_extract_pdf", lambda path: "피디에프 본문입니다.")
    monkeypatch.setattr(corpus, "_extract_hwp", lambda path: "한글 파일 본문입니다.")

    docs = iter_corpus(tmp_path)
    titles = sorted(d.title for d in docs)
    assert titles == ["노트", "옛글", "한글"]


def test_binary_doc_falls_back_to_mtime(monkeypatch, tmp_path):
    # 발굴 글: 내용·파일명에 날짜 없음 → mtime 사용
    p = tmp_path / "긁적긁적 정호.pdf"
    p.write_bytes(b"%PDF-fake")
    monkeypatch.setattr(corpus, "_extract_pdf", lambda path: "날짜 없는 본문입니다.")
    doc = load_document(p)
    assert doc.timestamp_origin == "mtime"
    assert isinstance(doc.timestamp, dt.date)
    assert doc.title == "긁적긁적 정호"
