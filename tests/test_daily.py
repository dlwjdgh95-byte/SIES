"""daily/threads 순수함수 단위테스트 — 네트워크 없음."""
import pytest

from people.daily import load_state, pick_next, save_state
from people.threads import validate_posts


# ── validate_posts ───────────────────────────────────────────
def test_validate_posts_ok():
    posts = ["훅이에요 (1/4)", "본문 (2/4)", "개념 (3/4)", "액션 (4/4)"]
    assert validate_posts(posts) == posts


def test_validate_posts_too_long():
    posts = ["a" * 501, "b", "c", "d"]
    with pytest.raises(ValueError, match="1번 포스트 501자"):
        validate_posts(posts)


def test_validate_posts_count():
    with pytest.raises(ValueError, match="개수"):
        validate_posts(["하나", "둘"])
    with pytest.raises(ValueError, match="개수"):
        validate_posts([f"p{i}" for i in range(9)])


def test_validate_posts_empty_post():
    with pytest.raises(ValueError, match="비어"):
        validate_posts(["훅", "  ", "셋", "넷"])


def test_validate_posts_not_strings():
    with pytest.raises(ValueError, match="문자열 리스트"):
        validate_posts(["훅", 123, "셋", "넷"])


# ── pick_next ────────────────────────────────────────────────
Q = ["Q1", "Q2", "Q3"]


def test_pick_next_first_unposted():
    state = {"pending": None, "posted": {"Q1": {}}}
    assert pick_next(state, Q) == "Q2"


def test_pick_next_pending_wins():
    state = {"pending": "Q3", "posted": {}}
    assert pick_next(state, Q) == "Q3"


def test_pick_next_posted_pending_ignored():
    # pending이 이미 발행됐으면(레이스) 무시하고 다음 미발행으로.
    state = {"pending": "Q1", "posted": {"Q1": {}}}
    assert pick_next(state, Q) == "Q2"


def test_pick_next_exhausted():
    state = {"pending": None, "posted": {q: {} for q in Q}}
    assert pick_next(state, Q) is None


# ── state 라운드트립 ─────────────────────────────────────────
def test_state_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    assert load_state(p) == {"pending": None, "posted": {}}
    s = {"pending": "Q1", "posted": {"Q0": {"date": "2026-07-06", "post_ids": ["1"]}}}
    save_state(s, p)
    assert load_state(p) == s
