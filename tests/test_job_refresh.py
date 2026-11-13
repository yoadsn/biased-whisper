import json
import os
import time

from bwhisper import jobs


def write_job(root, jid, status="running", done=0, total=10):
    d = root / jid
    d.mkdir(exist_ok=True)
    path = d / "job.json"
    path.write_text(json.dumps({"id": jid, "type": "demo", "config": {}, "status": status,
                                "progress": {"done": done, "total": total}, "message": "", "summary": None,
                                "error": None, "created": time.time(), "started": None, "finished": None}))
    t = time.time() + done  # distinct mtimes regardless of filesystem timestamp resolution
    os.utime(path, (t, t))
    return d


def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(jobs, "ensure_dirs", lambda: None)
    return jobs.Manager()


def test_runs_started_by_another_process_appear_and_update(tmp_path, monkeypatch):
    m = manager(tmp_path, monkeypatch)
    assert m.list() == []
    write_job(tmp_path, "cli-1", done=1)  # CLI process starts a run after the server is up
    assert [j["id"] for j in m.list()] == ["cli-1"]
    assert m.get("cli-1").progress["done"] == 1
    write_job(tmp_path, "cli-1", done=7)
    assert m.get("cli-1").progress["done"] == 7
    write_job(tmp_path, "cli-1", status="done", done=10)
    assert m.list()[0]["status"] == "done"


def test_active_runs_at_startup_are_interrupted_but_a_live_cli_can_overwrite(tmp_path, monkeypatch):
    write_job(tmp_path, "old", done=3)
    m = manager(tmp_path, monkeypatch)
    assert m.get("old").status == "interrupted"
    write_job(tmp_path, "old", done=9)  # the CLI process kept running and saved again
    assert m.get("old").status == "running" and m.get("old").progress["done"] == 9


def test_own_runs_are_never_overwritten_from_disk(tmp_path, monkeypatch):
    m = manager(tmp_path, monkeypatch)
    monkeypatch.setitem(jobs.REGISTRY, "demo", {"kind": "io", "defaults": {}, "fn": lambda ctx, cfg: {}})
    j = m.new_job("demo", {})
    j.status = "running"
    write_job(tmp_path, j.id, status="failed")
    assert m.get(j.id).status == "running"


def test_ui_cancel_reaches_a_run_owned_by_another_process(tmp_path, monkeypatch):
    m = manager(tmp_path, monkeypatch)
    d = write_job(tmp_path, "cli-2", done=2)
    assert m.cancel("cli-2")
    assert (d / "cancel.flag").exists()
    ctx = jobs.Ctx(m.get("cli-2"))
    assert ctx.should_stop()  # the owning process sees the marker on its next check
    assert not m.cancel("missing")
