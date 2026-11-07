import weakref

import pytest

from bwhisper import jobs, model


@pytest.mark.parametrize("outcome", ["success", "cancel", "failure"])
def test_cleanup_runs_after_tensor_frames_unwind(tmp_path, monkeypatch, outcome):
    monkeypatch.setattr(jobs, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(jobs, "ensure_dirs", lambda: None)
    refs, cleaned = [], []
    class Resource:
        pass
    def run(ctx, cfg):
        resource = Resource()
        refs.append(weakref.ref(resource))
        if outcome == "cancel":
            raise jobs.Cancelled()
        if outcome == "failure":
            raise RuntimeError("intentional failure")
        return {"ok": True}
    def cleanup():
        assert refs[0]() is None  # not inside the live job's finally block
        cleaned.append(True)
    monkeypatch.setattr(model, "unload_all", cleanup)
    monkeypatch.setitem(jobs.REGISTRY, "cleanup_test", {"kind": "gpu", "defaults": {"release_model_after_run": True}, "fn": run})
    manager = jobs.Manager()
    try:
        job = manager.run_sync("cleanup_test")
        assert cleaned == [True]
        assert job.status == {"success": "done", "cancel": "cancelled", "failure": "failed"}[outcome]
    finally:
        for pool in manager.pools.values():
            pool.shutdown()


@pytest.mark.parametrize("kind,release", [("gpu", False), ("io", True)])
def test_cleanup_does_not_affect_unrelated_jobs(tmp_path, monkeypatch, kind, release):
    monkeypatch.setattr(jobs, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(jobs, "ensure_dirs", lambda: None)
    monkeypatch.setattr(model, "unload_all", lambda: pytest.fail("unexpected model cleanup"))
    monkeypatch.setitem(jobs.REGISTRY, "keep_test", {
        "kind": kind, "defaults": {"release_model_after_run": release}, "fn": lambda ctx, cfg: {"ok": True},
    })
    manager = jobs.Manager()
    try:
        assert manager.run_sync("keep_test").status == "done"
    finally:
        for pool in manager.pools.values():
            pool.shutdown()


def test_target_job_types_default_to_releasing_models():
    from bwhisper.budget import BUDGET_DEFAULTS
    from bwhisper.config import PHASE3_DEFAULTS
    from bwhisper.phase3_device import BENCHMARK_DEFAULTS

    assert all(cfg["release_model_after_run"] for cfg in
               (BUDGET_DEFAULTS, PHASE3_DEFAULTS, BENCHMARK_DEFAULTS))
