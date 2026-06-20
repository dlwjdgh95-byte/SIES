"""store.py — sqlite-vec 저장/검색 왕복. 가짜 벡터로 모델 없이 검증."""
import numpy as np
import pytest

from sies.chunk import Chunk
from sies.store import (
    connect,
    ensure_vec_table,
    init_schema,
    search,
    store_embeddings,
    upsert_chunks,
)

ALIAS = "kure"
DIM = 4


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    init_schema(c)
    ensure_vec_table(c, ALIAS, DIM)
    yield c
    c.close()


def _chunk(i, text):
    return Chunk(
        doc_path=f"corpus/essays/doc{i}.md",
        source="essays",
        title=f"제목{i}",
        timestamp="2026-03-29",
        chunk_index=0,
        text=text,
    )


def _unit(vec):
    v = np.array(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_roundtrip_returns_nearest_first(conn):
    chunks = [_chunk(0, "사과"), _chunk(1, "바나나"), _chunk(2, "자동차")]
    vecs = np.stack([_unit([1, 0, 0, 0]), _unit([0, 1, 0, 0]), _unit([0, 0, 1, 0])])
    ids = upsert_chunks(conn, chunks)
    store_embeddings(conn, ALIAS, ids, vecs)

    # [1,0,0,0]에 가장 가까운 건 chunk0("사과")
    res = search(conn, ALIAS, _unit([0.9, 0.1, 0, 0]), k=3)
    assert len(res) == 3
    assert res[0]["title"] == "제목0"
    assert res[0]["text"] == "사과"
    # distance는 오름차순
    dists = [r["distance"] for r in res]
    assert dists == sorted(dists)


def test_k_limits_results(conn):
    chunks = [_chunk(i, f"t{i}") for i in range(5)]
    vecs = np.stack([_unit([1, i, 0, 0]) for i in range(5)])
    ids = upsert_chunks(conn, chunks)
    store_embeddings(conn, ALIAS, ids, vecs)
    assert len(search(conn, ALIAS, _unit([1, 0, 0, 0]), k=2)) == 2


def test_upsert_is_idempotent(conn):
    """같은 (doc_path, chunk_index) 재인덱싱 시 중복 행이 생기지 않는다."""
    c = _chunk(0, "원본")
    ids1 = upsert_chunks(conn, [c])
    c.text = "수정됨"
    ids2 = upsert_chunks(conn, [c])
    assert ids1 == ids2  # 같은 행 id 재사용
    row = conn.execute("SELECT text FROM chunks WHERE id=?", (ids1[0],)).fetchone()
    assert row[0] == "수정됨"
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1


def test_reembedding_replaces_vector(conn):
    """같은 청크를 다시 임베딩하면 벡터 테이블에도 중복이 없다."""
    c = _chunk(0, "x")
    ids = upsert_chunks(conn, [c])
    store_embeddings(conn, ALIAS, ids, np.stack([_unit([1, 0, 0, 0])]))
    store_embeddings(conn, ALIAS, ids, np.stack([_unit([0, 1, 0, 0])]))
    cnt = conn.execute(f"SELECT COUNT(*) FROM vec_{ALIAS}").fetchone()[0]
    assert cnt == 1


def test_search_empty_table_returns_empty(conn):
    assert search(conn, ALIAS, _unit([1, 0, 0, 0]), k=5) == []
