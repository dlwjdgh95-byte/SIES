"""story/cards 순수함수 단위테스트 — API 호출 없음."""
import json

from people.cards import render_card, render_index
from people.story import STORY_SCHEMA, build_prompt, write_prompt_pack

SAMPLE = {
    "qid": "Q313711",
    "name_en": "Canaan Banana",
    "name_ko": "케이넌 바나나",
    "occupation": "religious",
    "peak_month": "2016-03",
    "peak_views": 308_106,
    "recent_monthly_views": 5655.08,
    "wiki_en": "https://en.wikipedia.org/wiki/Canaan_Banana",
    "wiki_ko": "https://ko.wikipedia.org/wiki/카난_바나나",
    "summary_lang": "ko",
    "summary": "짐바브웨의 초대 대통령이었다.",
    "description": "짐바브웨의 정치인",
    "awards": [{"label": "노벨 문학상", "month": "2015-10"}],
    "death_month": "2003-11",
    "peak_reason": {"type": "unknown", "detail": None},
    "image_name": "Foo.jpg",
    "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Foo.jpg?width=512",
}

STORY = {
    "headline": "권력을 내려놓은 첫 대통령",
    "peak_story": "정점 서사 본문.",
    "modern_inspiration": "영감 본문.",
    "tags": ["역사", "아프리카"],
}


# ── story ────────────────────────────────────────────────────
def test_build_prompt_contains_facts():
    p = build_prompt(SAMPLE)
    assert "케이넌 바나나" in p
    assert "짐바브웨의 초대 대통령" in p
    assert "2016-03" in p and "308,106" in p
    assert "노벨 문학상(2015-10)" in p


def test_build_prompt_missing_summary_guard():
    e = {**SAMPLE, "summary": None}
    assert "발췌 없음" in build_prompt(e)


def test_schema_is_strict_object():
    # structured outputs 요건: additionalProperties=false + required 전체 명시
    assert STORY_SCHEMA["additionalProperties"] is False
    assert set(STORY_SCHEMA["required"]) == set(STORY_SCHEMA["properties"].keys())


def test_write_prompt_pack(tmp_path):
    write_prompt_pack([SAMPLE], tmp_path)
    assert (tmp_path / "Q313711.txt").exists()
    assert (tmp_path / "_SYSTEM.txt").exists()
    json.loads((tmp_path / "_SCHEMA.json").read_text())  # 유효한 JSON


# ── cards ────────────────────────────────────────────────────
def test_render_card_with_story():
    md = render_card(SAMPLE, STORY)
    assert md.startswith("# 케이넌 바나나 (Canaan Banana)")
    assert "권력을 내려놓은 첫 대통령" in md
    assert "정점 서사 본문." in md
    assert "`역사` `아프리카`" in md
    assert "서사 미생성" not in md


def test_render_card_without_story_placeholder():
    md = render_card(SAMPLE, None)
    assert "서사 미생성" in md
    assert "위키 발췌" in md            # 서사 없을 땐 발췌로 대체
    assert "Q313711.txt" in md          # 프롬프트 팩 안내


def test_render_index_marks_story_presence():
    idx = render_index([SAMPLE], {"Q313711": STORY})
    assert "✅" in idx
    idx2 = render_index([SAMPLE], {})
    assert "✅" not in idx2
