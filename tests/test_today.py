"""today.py — 오늘의 잊힌 나. 가짜 4차원 단위벡터로 임베딩 모델 없이 검증."""
import argparse
import datetime as dt
import json

import numpy as np
import pytest

from sies.chunk import Chunk
from sies.store import connect, ensure_vec_table, init_schema, store_embeddings, upsert_chunks
from sies.today import cmd_mark, cmd_prepare, pick_today

ALIAS = "kure"
NOW = dt.date(2026, 7, 7)


def _unit(v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


@pytest.fixture
def db(tmp_path):
    """문서 4편 — doc0(2018)·doc1(2019)은 잊힘, doc2는 날짜 없음(중립 0.5), recent는 오늘(활성)."""
    path = tmp_path / "t.db"
    c = connect(path)
    init_schema(c)
    ensure_vec_table(c, ALIAS, 4)
    metas = [("corpus/doc0.md", "제목0", "2018-01-01", "본문0"),
             ("corpus/doc1.md", "제목1", "2019-06-01", "본문1"),
             ("corpus/doc2.md", "제목2", None, "본문2"),
             ("corpus/recent.md", "최근", NOW.isoformat(), "최근 글")]
    chunks = [Chunk(doc_path=p, source="essays", title=t, timestamp=ts, chunk_index=0, text=x)
              for p, t, ts, x in metas]
    ids = upsert_chunks(c, chunks)
    store_embeddings(c, ALIAS, ids, np.stack([_unit([1, i, 0, 0]) for i in range(4)]))
    c.close()
    return path


def _prep_args(db, log):
    return argparse.Namespace(db=str(db), model=ALIAS, log=str(log))


def _mark_args(log, read=True, skip=False, good=False, bad=False):
    return argparse.Namespace(log=str(log), read=read, skip=skip, good=good, bad=bad)


def test_deterministic_same_now(db):
    c = connect(db)  # 같은 now+DB면 항상 같은 문서 — 가장 오래된(가장 잊힌) doc0
    a, b = pick_today(c, NOW, []), pick_today(c, NOW, [])
    assert a["doc_path"] == b["doc_path"] == "corpus/doc0.md"


def test_active_and_undated_docs_excluded(db):
    """오늘 쓴 글(활성도~1)과 날짜 없는 글(0.5)은 LOW_ACTIVITY 문턱을 못 넘는다."""
    c = connect(db)
    picked = {pick_today(c, NOW, ev)["doc_path"] for ev in
              ([], [{"ts": NOW.isoformat(), "event": "pick", "doc_path": "corpus/doc0.md"}])}
    assert picked == {"corpus/doc0.md", "corpus/doc1.md"}


def test_cooldown_excludes_marked_pick(db):
    c = connect(db)
    events = [{"ts": NOW.isoformat(), "event": "pick", "doc_path": "corpus/doc0.md", "title": "제목0"},
              {"ts": NOW.isoformat(), "event": "mark", "doc_path": "corpus/doc0.md",
               "status": "read", "judge": None}]
    assert pick_today(c, NOW, events)["doc_path"] == "corpus/doc1.md"


def test_pending_represented_without_new_log(db, tmp_path, capsys):
    """mark 없이 prepare 두 번 → 같은 문서 재제시, pick 로그는 1건만."""
    log = tmp_path / "log.jsonl"
    cmd_prepare(_prep_args(db, log))
    first = capsys.readouterr().out
    cmd_prepare(_prep_args(db, log))
    second = capsys.readouterr().out
    assert "제목0" in first and "제목0" in second and "본문0" in second
    assert [json.loads(ln)["event"] for ln in log.read_text().splitlines()] == ["pick"]


def test_mark_without_pending_exits(tmp_path):
    with pytest.raises(SystemExit):
        cmd_mark(_mark_args(tmp_path / "log.jsonl"))


def test_mark_then_next_prepare_picks_new_doc(db, tmp_path, capsys):
    """mark로 확정(냉각 개시)하면 다음 prepare는 다른 문서를 고른다."""
    log = tmp_path / "log.jsonl"
    cmd_prepare(_prep_args(db, log))
    cmd_mark(_mark_args(log, good=True))
    capsys.readouterr()
    cmd_prepare(_prep_args(db, log))
    assert "제목1" in capsys.readouterr().out
    ev = [json.loads(ln) for ln in log.read_text().splitlines()]
    assert [e["event"] for e in ev] == ["pick", "mark", "pick"]
    assert ev[1]["status"] == "read" and ev[1]["judge"] == "good"
