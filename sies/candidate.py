"""검색 후보의 입력 계약 — 제품의 뇌(rank.py)가 소비하는 *유일한* 형태.

이 계약은 의도적으로 **모달리티 무관(modality-free)** 이다. 랭커는 글의 내용
(text)도, 그것이 텍스트인지 이미지인지 비디오인지도 보지 않는다. 오직:
  - distance: 정규화(코사인 공간) 임베딩의 벡터 거리 — *어떤* 모달리티든 무방
  - timestamp: ISO 날짜(없으면 중립 활성도)
  - 문서 정체성(doc_path / title): 볼륨(몰아쓰기) 축 계산용
만 본다. text / media_ref / source / payload 등 그 밖의 키는 들고 다녀도 되지만
점수 계산에는 절대 닿지 않는다(원칙 2: 검색 결정은 결정론적 산수뿐).

그래서 미래에 이미지·비디오 모달리티를 추가해도(예: 텍스트질의→이미지를 CLIP계
공유공간에서 따로 임베딩) 각 파이프라인이 이 Candidate만 내놓으면 rank.py는
손대지 않고 그대로 동작한다. 설계 메모: docs/multimodal-contract.md.
"""
from __future__ import annotations

from typing import NotRequired, TypedDict


class Candidate(TypedDict):
    """rank.py의 입력 계약. store.search()가 돌려주는 dict가 이미 이를 만족한다.

    추가 키(text, id, chunk_index, 미래의 media_ref 등)는 허용되며 랭커가 무시한다.
    distance는 반드시 *정규화된* 임베딩(코사인 공간)에서 와야 한다 —
    cosine_from_l2가 d²=2(1−cos) 관계를 가정하기 때문.
    """
    distance: float                 # 벡터 L2 거리(코사인 공간). 필수.
    timestamp: NotRequired[str]     # ISO 날짜. ""·누락 → 활성도 중립(0.5)
    doc_path: NotRequired[str]      # 문서 정체성(볼륨 축)
    title: NotRequired[str]         # doc_path 없을 때 정체성 폴백
