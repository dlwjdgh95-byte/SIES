"""공용 검색 헬퍼 — 질의 임베딩 + 전체 후보 풀 조회.

역전 재순위의 밴드패스는 *전체 유사도 분포*에 대한 퍼센타일이므로,
상위 k가 아니라 모든 청크를 후보로 가져온다(개인용 규모라 가능).
"""
from __future__ import annotations

import numpy as np

from .store import count_chunks, search


def candidate_pool(conn, alias: str, query_vec: np.ndarray) -> list[dict]:
    n = count_chunks(conn)
    if n == 0:
        return []
    return search(conn, alias, query_vec, k=n)
