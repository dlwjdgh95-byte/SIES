"""normalize.py — 앞머리 마커 제거 + PDF 띄어쓰기 교정."""
from sies.normalize import fix_spacing, strip_leading_marker


def test_strip_question_markers():
    assert strip_leading_marker("Q3, Q4 자본주의사회에서") == "자본주의사회에서"
    assert strip_leading_marker("Q5. 가치관 다른것") == "가치관 다른것"


def test_strip_numeric_markers():
    assert strip_leading_marker("8. 가치관차이") == "가치관차이"
    assert strip_leading_marker("3. 에피쿠로스의 재평가") == "에피쿠로스의 재평가"
    assert strip_leading_marker("2) 무언가를") == "무언가를"


def test_keeps_non_markers():
    # 불릿(-)·연도·일반 문장은 안 건드린다.
    assert strip_leading_marker("- 내가 지켜야할") == "- 내가 지켜야할"
    assert strip_leading_marker("2023년 겨울에") == "2023년 겨울에"
    assert strip_leading_marker("나는 사랑받고 싶어 한다") == "나는 사랑받고 싶어 한다"


def test_fix_spacing_removes_intra_word_spaces():
    out = fix_spacing("올바른 방 식으로 최 선을 다하는")
    assert "방식으로" in out and "최선을" in out


def test_fix_spacing_preserves_proper_nouns():
    # Kiwi가 분할하려는 고유명사/합성어는 보존(제거만 채택).
    src = "강명국 중사의 민트색 레이를 타고 성남 15비행단으로 향했다."
    out = fix_spacing(src)
    assert "강명국" in out and "민트색" in out and "15비행단" in out


def test_fix_spacing_preserves_newlines():
    out = fix_spacing("첫 줄입니다.\n\n둘째 문단.")
    assert "\n\n" in out
