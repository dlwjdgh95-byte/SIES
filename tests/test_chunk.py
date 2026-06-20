"""chunk.py — 문단 분할과 짧은 문단 병합."""
import datetime as dt
from pathlib import Path

from sies.chunk import MIN_CHARS, Chunk, chunk_document, split_paragraphs
from sies.corpus import Document


# MIN_CHARS(40)를 확실히 넘는 문단 — 한국어는 압축적이라 길이를 넉넉히 잡는다
LONG_A = "이것은 충분히 긴 첫 번째 문단입니다. 마흔 글자를 확실히 넘기려고 문장을 길게 늘여 적습니다."
LONG_B = "이것은 충분히 긴 두 번째 문단입니다. 역시 마흔 글자를 넘도록 내용을 충분히 채워 넣었습니다."
SHORT = "짧음."  # MIN_CHARS 미만


def test_fixtures_span_min_chars():
    # 픽스처 전제 검증: LONG은 임계 초과, SHORT는 미만
    assert len(LONG_A) > MIN_CHARS and len(LONG_B) > MIN_CHARS
    assert len(SHORT) < MIN_CHARS


def test_split_basic_paragraphs():
    paras = split_paragraphs(f"{LONG_A}\n\n{LONG_B}")
    assert len(paras) == 2


def test_split_handles_whitespace_only_lines():
    # 공백만 있는 줄도 문단 구분자로 취급
    paras = split_paragraphs(f"{LONG_A}\n   \n{LONG_B}")
    assert len(paras) == 2


def test_short_paragraph_merges_into_previous():
    paras = split_paragraphs(f"{LONG_A}\n\n{SHORT}")
    assert len(paras) == 1
    assert SHORT in paras[0]


def test_leading_short_paragraph_stays_standalone():
    # 앞에 병합할 문단이 없으면 짧아도 독립 유지
    paras = split_paragraphs(f"{SHORT}\n\n{LONG_A}")
    assert paras[0] == SHORT
    assert len(paras) == 2


def test_empty_body_yields_no_paragraphs():
    assert split_paragraphs("") == []
    assert split_paragraphs("   \n  \n") == []


def _doc(body, ts=dt.date(2026, 3, 29)):
    return Document(
        path=Path("corpus/essays/x.md"),
        source="essays",
        title="X",
        body=body,
        timestamp=ts,
    )


def test_chunk_document_carries_metadata():
    chunks = chunk_document(_doc("문단 하나 충분히 길어요 길어요 길어요 길어요."))
    assert len(chunks) == 1
    c = chunks[0]
    assert isinstance(c, Chunk)
    assert c.title == "X"
    assert c.source == "essays"
    assert c.timestamp == "2026-03-29"
    assert c.chunk_index == 0


def test_chunk_indices_are_sequential():
    body = "\n\n".join(
        f"이것은 {i}번째 문단입니다. 마흔 글자를 넘기려고 문장을 충분히 길게 늘여 적습니다."
        for i in range(4)
    )
    chunks = chunk_document(_doc(body))
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]


def test_chunk_handles_missing_timestamp():
    chunks = chunk_document(_doc("문단 하나 충분히 길어요 길어요 길어요 길어요.", ts=None))
    assert chunks[0].timestamp == ""
