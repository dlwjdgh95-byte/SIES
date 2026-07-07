"""today.py — 오늘의 잊힌 나. 가짜 4차원 단위벡터로 임베딩 모델 없이 검증."""
import argparse
import datetime as dt
import json
import sys

import numpy as np
import pytest

from sies.chunk import Chunk
from sies.store import connect, ensure_vec_table, init_schema, store_embeddings, upsert_chunks
from sies.today import cmd_mark, cmd_prepare, main, pick_today

ALIAS = "kure"
NOW = dt.date(2026, 7, 7)


def _unit(v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


@pytest.fixture
def db(tmp_path):
    """문서 5편 — doc0(2018)·doc1(2019)만 잊힘. doc2(날짜 없음)·dirty(비ISO ts)는 중립 0.5, recent는 활성."""
    path = tmp_path / "t.db"
    c = connect(path)
    init_schema(c)
    ensure_vec_table(c, ALIAS, 4)
    metas = [("corpus/doc0.md", "제목0", "2018-01-01", "본문0"),
             ("corpus/doc1.md", "제목1", "2019-06-01", "본문1"),
             ("corpus/doc2.md", "제목2", None, "본문2"),
             ("corpus/dirty.md", "지저분", "2018-01-01T09:30:00", "본문d"),
             ("corpus/recent.md", "최근", NOW.isoformat(), "최근 글")]
    chunks = [Chunk(doc_path=p, source="essays", title=t, timestamp=ts, chunk_index=0, text=x)
              for p, t, ts, x in metas]
    ids = upsert_chunks(c, chunks)
    store_embeddings(c, ALIAS, ids, np.stack([_unit([1, i, 0, 0]) for i in range(5)]))
    c.close()
    return path


def _prep_args(db, log):
    return argparse.Namespace(db=str(db), model=ALIAS, log=str(log))


def _mark_args(log, read=True, skip=False, good=False, bad=False):
    return argparse.Namespace(log=str(log), read=read, skip=skip, good=good, bad=bad)


def _mark(doc_path, ts=NOW, status="read"):
    return {"ts": ts.isoformat(), "event": "mark", "doc_path": doc_path, "status": status, "judge": None}


def test_deterministic_same_now(db):
    c = connect(db)  # 같은 now+DB면 항상 같은 문서 — 가장 오래된(가장 잊힌) doc0
    a, b = pick_today(c, NOW, []), pick_today(c, NOW, [])
    assert a["doc_path"] == b["doc_path"] == "corpus/doc0.md"


def test_active_and_undated_docs_excluded(db):
    """잊힌 doc0·doc1을 냉각시키면 활성(recent)·중립(doc2·dirty)만 남아 후보가 없어야 한다."""
    c = connect(db)
    events = [_mark("corpus/doc0.md"), _mark("corpus/doc1.md")]
    assert pick_today(c, NOW, events) is None


def test_cooldown_excludes_marked_pick(db):
    c = connect(db)
    events = [{"ts": NOW.isoformat(), "event": "pick", "doc_path": "corpus/doc0.md", "title": "제목0"},
              _mark("corpus/doc0.md")]
    assert pick_today(c, NOW, events)["doc_path"] == "corpus/doc1.md"


def test_cooldown_starts_at_mark_ts_not_pick_ts(db):
    """91일+ 방치한 pending을 오늘 mark → 냉각 기점은 mark ts라 즉시 재선정되지 않는다."""
    c = connect(db)
    events = [{"ts": (NOW - dt.timedelta(days=200)).isoformat(), "event": "pick",
               "doc_path": "corpus/doc0.md", "title": "제목0"},
              _mark("corpus/doc0.md")]
    assert pick_today(c, NOW, events)["doc_path"] == "corpus/doc1.md"


def test_corrupt_log_events_ignored(db):
    """ts 결측·비ISO인 mark 이벤트는 냉각 판단에서 조용히 무시된다(로그 오염 내성)."""
    c = connect(db)
    events = [{"event": "mark", "status": "read"},  # ts·doc_path 자체가 없음
              {"ts": "엉망", "event": "mark", "doc_path": "corpus/doc1.md", "status": "read"}]
    assert pick_today(c, NOW, events)["doc_path"] == "corpus/doc0.md"


def test_prepare_survives_non_iso_db_timestamp(db, tmp_path, capsys):
    cmd_prepare(_prep_args(db, tmp_path / "log.jsonl"))  # dirty.md(비ISO ts)가 있어도 죽지 않는다
    assert "제목0" in capsys.readouterr().out


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


def test_mark_skip_with_judge_rejected(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv",  # --skip+--good 거부 — 미열람 판정이 good율을 오염시키지 않게
                        ["today", "mark", "--skip", "--good", "--log", str(tmp_path / "l.jsonl")])
    with pytest.raises(SystemExit):
        main()


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
