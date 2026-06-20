"""sqlite-vec 벡터 저장소.

모델별로 임베딩 테이블을 분리(차원이 다르고, Phase 0에서 모델을 비교하므로).
chunks 테이블은 모델 무관 메타데이터를 보관한다.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import sqlite_vec

from .chunk import Chunk


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _vec_table(alias: str) -> str:
    return f"vec_{alias.replace('-', '_')}"


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            doc_path    TEXT NOT NULL,
            source      TEXT NOT NULL,
            title       TEXT NOT NULL,
            timestamp   TEXT,
            chunk_index INTEGER NOT NULL,
            text        TEXT NOT NULL,
            UNIQUE(doc_path, chunk_index)
        )
        """
    )
    conn.commit()


def ensure_vec_table(conn: sqlite3.Connection, alias: str, dim: int) -> None:
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {_vec_table(alias)} "
        f"USING vec0(embedding float[{dim}])"
    )
    conn.commit()


def upsert_chunks(conn: sqlite3.Connection, chunks: list[Chunk]) -> list[int]:
    """청크 메타를 저장하고 rowid 목록을 반환(임베딩 정렬용)."""
    ids = []
    for c in chunks:
        cur = conn.execute(
            """
            INSERT INTO chunks (doc_path, source, title, timestamp, chunk_index, text)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(doc_path, chunk_index) DO UPDATE SET
                title=excluded.title, timestamp=excluded.timestamp, text=excluded.text
            RETURNING id
            """,
            (c.doc_path, c.source, c.title, c.timestamp, c.chunk_index, c.text),
        )
        ids.append(cur.fetchone()[0])
    conn.commit()
    return ids


def store_embeddings(
    conn: sqlite3.Connection, alias: str, chunk_ids: list[int], vecs: np.ndarray
) -> None:
    table = _vec_table(alias)
    for cid, vec in zip(chunk_ids, vecs):
        conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (cid,))
        conn.execute(
            f"INSERT INTO {table} (rowid, embedding) VALUES (?, ?)",
            (cid, vec.tobytes()),
        )
    conn.commit()


def count_chunks(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def search(
    conn: sqlite3.Connection, alias: str, query_vec: np.ndarray, k: int = 10
) -> list[dict]:
    """벡터 KNN — 순수 유사도(베이스라인). distance 작을수록 가깝다."""
    table = _vec_table(alias)
    rows = conn.execute(
        f"""
        SELECT c.id, c.title, c.source, c.timestamp, c.chunk_index, c.text, v.distance
        FROM {table} v
        JOIN chunks c ON c.id = v.rowid
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (query_vec.astype(np.float32).tobytes(), k),
    ).fetchall()
    cols = ["id", "title", "source", "timestamp", "chunk_index", "text", "distance"]
    return [dict(zip(cols, r)) for r in rows]
