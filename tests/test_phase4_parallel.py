import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import pytest
import torch

from bwhisper import phase4_parallel as P
from bwhisper.phase4 import DEFAULTS
from bwhisper.phase4_worker import global_batches


def test_parallel_config_preserves_global_batch():
    cfg = copy.deepcopy(DEFAULTS)
    for world in (1, 2, 4, 8):
        cfg["devices"] = [f"cuda:{i}" for i in range(world)]
        P.validate_parallel(cfg, 8)
    cfg["devices"] = ["cuda:0", "cuda:0"]
    with pytest.raises(ValueError):
        P.validate_parallel(cfg, 8)
    cfg["devices"] = ["cuda:0", "cuda:1", "cuda:2"]
    with pytest.raises(ValueError, match="multiple"):
        P.validate_parallel(cfg, 8)


def test_direction_config_requires_explicit_cached_source():
    from bwhisper.phase4 import validate
    cfg = copy.deepcopy(DEFAULTS)
    cfg["target_style"] = "digits"
    with pytest.raises(ValueError, match="teacher_source_run"):
        validate(cfg)
    cfg["teacher_source_run"] = "phase4-source"
    validate(cfg)
    cfg["parallel"] = False
    with pytest.raises(ValueError, match="parallel"):
        validate(cfg)
    cfg["target_style"] = "unknown"
    with pytest.raises(ValueError, match="target_style"):
        validate(cfg)


@pytest.mark.parametrize("style", ["cardinals", "digits"])
def test_phase4_run_id_contains_direction(tmp_path, monkeypatch, style):
    from bwhisper import jobs
    monkeypatch.setattr(jobs, "RUNS_DIR", tmp_path)
    manager = jobs.Manager.__new__(jobs.Manager)
    manager.jobs = {}
    job = manager.new_job("phase4", {"target_style": style})
    assert job.id.startswith(f"phase4-{style}-")
    assert json.loads((job.dir / "job.json").read_text())["config"]["target_style"] == style


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_global_batches_cover_once_with_equal_rank_slots(world):
    ids = list(range(11))
    ranks = [list(global_batches(ids, 8, rank, world)) for rank in range(world)]
    observed = [i for batches in ranks for slots, _ in batches for i in slots if i is not None]
    assert sorted(observed) == ids
    assert all(len(batches) == 2 for batches in ranks)
    assert all(len(slots) == 8/world for batches in ranks for slots, _ in batches)
    assert all(batches[-1][1] == 3 for batches in ranks)


def test_cache_checksums_and_audio_identity(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"first")
    sample = {"id": 1, "source": "test", "meta": {"recording": "one"}, "audio_path": str(audio)}
    key = P.sample_identity(sample)
    row = {"fingerprint": "teacher", "sample_key": key, "target": None, "reason": "degenerate"}
    entry = tmp_path / "entry.json"
    assert P.cache_entry(entry, "teacher", key) is None
    P.save_cache(entry, row)
    assert P.cache_entry(entry, "teacher", key)["reason"] == "degenerate"
    changed = json.loads(entry.read_text())
    changed["reason"] = "accepted"
    entry.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="corrupt"):
        P.cache_entry(entry, "teacher", key)
    audio.write_bytes(b"changed")
    assert P.sample_identity(sample) != key


def test_cache_lock_refuses_competing_preparation(tmp_path):
    with P.cache_lock(tmp_path):
        with pytest.raises(ValueError, match="already"):
            with P.cache_lock(tmp_path):
                pass


def test_checkpoint_identity_detects_weight_and_tokenizer_changes(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "tokenizer.json").write_text('{"token":1}')
    identity = P.checkpoint_identity(tmp_path)
    (tmp_path / "tokenizer.json").write_text('{"token":2}')
    assert P.checkpoint_identity(tmp_path) != identity


def test_task_queue_claims_each_sample_once(tmp_path):
    path = tmp_path / "tasks.db"
    P.task_queue(path, [{"id": i} for i in range(10)])
    one = sqlite3.connect(path)
    two = sqlite3.connect(path)
    try:
        ids = [P.claim_task(one if i%2 else two)["id"] for i in range(10)]
        assert ids == list(range(10))
        assert P.claim_task(one) is None
    finally:
        one.close()
        two.close()


@pytest.mark.parametrize("fail", [False, True])
def test_local_queue_journals_and_snapshot_cleanup(tmp_path, fail):
    path = tmp_path / "tasks.db"
    with pytest.raises(RuntimeError) if fail else nullcontext():
        with P.local_task_queue(path, [{"id": 1}, {"id": 2}]):
            local = path.resolve()
            assert path.is_symlink()
            assert local.parent.parent == Path("/tmp/opencode")
            with sqlite3.connect(path) as db:
                assert P.claim_task(db)["id"] == 1
                db.execute("BEGIN IMMEDIATE")
                db.execute("UPDATE tasks SET claimed=1 WHERE id=2")
                assert local.with_name("tasks.db-journal").exists()
                assert not path.with_name("tasks.db-journal").exists()
                db.rollback()
            if fail:
                raise RuntimeError("worker failure fixture")
    assert not path.is_symlink()
    assert not local.exists()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT id,claimed FROM tasks ORDER BY id").fetchall() == [(1, 1), (2, 0)]


def test_local_queue_eight_processes_claim_each_sample_once(tmp_path):
    path = tmp_path / "tasks.db"
    script = """
import json, sqlite3, sys, time
from bwhisper.phase4_parallel import claim_task
claimed = []
with sqlite3.connect(sys.argv[1], timeout=10) as db:
    while (sample := claim_task(db)) is not None:
        claimed.append(sample['id'])
        time.sleep(.001)
print(json.dumps(claimed))
"""
    with P.local_task_queue(path, [{"id": i} for i in range(800)]):
        children = [subprocess.Popen([sys.executable, "-c", script, str(path)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(8)]
        try:
            ids = []
            for child in children:
                stdout, stderr = child.communicate(timeout=30)
                assert child.returncode == 0, stderr
                ids.extend(json.loads(stdout))
            assert sorted(ids) == list(range(800))
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait()


def test_failed_claim_rolls_back_transaction(tmp_path):
    path = tmp_path / "tasks.db"
    P.task_queue(path, [{"id": 1}])
    with sqlite3.connect(path) as db:
        db.execute("CREATE TRIGGER fail_claim BEFORE UPDATE ON tasks BEGIN SELECT RAISE(ABORT, 'fixture'); END")
        with pytest.raises(sqlite3.IntegrityError, match="fixture"):
            P.claim_task(db)
        assert not db.in_transaction
    with sqlite3.connect(path, timeout=.1) as db:
        db.execute("DROP TRIGGER fail_claim")
        assert P.claim_task(db)["id"] == 1


class Context:
    def __init__(self):
        self.stop = False
    def should_stop(self):
        return self.stop


def test_process_failure_aborts_peers(tmp_path):
    commands = [[sys.executable, "-c", "raise RuntimeError('intentional fixture failure')"],
                [sys.executable, "-c", "import threading; threading.Event().wait(120)"]]
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="intentional fixture failure"):
        P.supervise(Context(), commands, tmp_path, 30, lambda: [])
    assert time.monotonic() - started < 20


def test_process_cancellation_is_bounded(tmp_path):
    from bwhisper.jobs import Cancelled
    ctx = Context()
    ctx.stop = True
    command = [sys.executable, "-c", "import threading; threading.Event().wait(120)"]
    with pytest.raises(Cancelled):
        P.supervise(ctx, [command], tmp_path, 30, lambda: [], cancel_grace=.1)
    assert (tmp_path / "STOP").exists()


def test_stalled_worker_times_out(tmp_path):
    command = [sys.executable, "-c", "import threading; threading.Event().wait(120)"]
    with pytest.raises(TimeoutError, match="stalled"):
        P.supervise(Context(), [command], tmp_path, 1,
                    lambda: [{"rank": 0, "stage": "training", "progress_at": time.time()-2}])


def test_terminal_worker_cannot_hang_during_shutdown(tmp_path):
    command = [sys.executable, "-c", "import threading; threading.Event().wait(120)"]
    with pytest.raises(TimeoutError, match="shutdown"):
        P.supervise(Context(), [command], tmp_path, .1,
                    lambda: [{"rank": 0, "stage": "done", "progress_at": time.time()}])


def test_teacher_filter_output_is_same_serial_or_worker(monkeypatch):
    from types import SimpleNamespace
    from bwhisper import phase4_worker as W
    from bwhisper.phase4 import training_target

    cfg = copy.deepcopy(DEFAULTS)
    L = SimpleNamespace(device="cpu", eos=31, sot_seq=[1, 3],
                        model=SimpleNamespace(config=SimpleNamespace(max_target_positions=32)))
    monkeypatch.setattr(W, "load_audio", lambda *a: None)
    for row in [
        {"verdict": "no_digits", "baseline_text": "עשרים אנשים", "baseline_tokens": [4, 5], "baseline_ended": True},
        {"verdict": "cardinal", "baseline_text": "20 אנשים", "suppressed_text": "עשרים אנשים", "y_cardinal": [4, 5],
         "baseline_ended": True, "suppressed_ended": True, "diff": {"rest_match": 1}},
        {"verdict": "degenerate", "baseline_text": "20 אנשים", "baseline_ended": True},
        {"verdict": "no_digits", "baseline_text": "עשרים אנשים", "baseline_tokens": [4], "baseline_ended": False},
    ]:
        monkeypatch.setattr(W, "analyse_sample", lambda *a: row)
        actual = W.generate_one(L, {"id": 1, "audio_path": "unused"}, cfg, None)
        assert (actual["target"], actual["reason"]) == training_target(row, cfg)


def test_resume_rng_includes_numpy_and_is_weights_only_serializable(tmp_path):
    import numpy as np
    import random
    from bwhisper.phase4_worker import rng_state, restore_rng

    saved = rng_state("cpu")
    path = tmp_path / "rng.pt"
    torch.save(saved, path)
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng(torch.load(path, weights_only=True), "cpu")
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)


def test_gpu_reservation_detects_other_process(tmp_path, monkeypatch):
    from bwhisper import config
    from bwhisper.jobs import gpu_reservation

    monkeypatch.setattr(config, "DATA", tmp_path)
    env = {**os.environ, "BW_DATA": str(tmp_path)}
    script = "from bwhisper.jobs import gpu_reservation\nwith gpu_reservation('phase4', {'parallel':True,'devices':['cuda:0']}): pass"
    with gpu_reservation("phase4", {"parallel": True, "devices": ["cuda:0"]}):
        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "reserved by another" in result.stderr


@pytest.mark.parametrize("all_rejected", [False, True])
def test_parent_reuses_cached_targets_and_keeps_one_writer(tmp_path, monkeypatch, all_rejected):
    from bwhisper import phase4, model
    from bwhisper.jobs import Ctx, Job

    monkeypatch.setattr(P, "DATA", tmp_path / "data")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(model, "unload_all", lambda: None)
    checkpoint = tmp_path / "base"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{}')
    samples = {}
    for sid in range(1, 4):
        audio = tmp_path / f"{sid}.wav"
        audio.write_bytes(bytes([sid]))
        samples[sid] = {"id": sid, "source": "fixture", "meta": {"recording": str(sid)}, "audio_path": str(audio)}
    manifest = {"train_ids": [1, 2], "eval_ids": [3], "cohort_run": "setup-fixture", "cohort_sha256": "fixture"}
    monkeypatch.setattr(phase4, "select_cohorts", lambda cfg: (samples, copy.deepcopy(manifest)))
    monkeypatch.setattr(P, "teacher_identity", lambda *a: {"teacher": "fixture"})
    calls = []
    def run(ctx, commands, directory, timeout, on_update, **kwargs):
        request = json.loads((ctx.dir / "request.json").read_text())
        calls.append(directory.name)
        if directory.name == "teacher":
            with sqlite3.connect(directory / "tasks.db") as db:
                while sample := P.claim_task(db):
                    key = request["sample_keys"][str(sample["id"])]
                    row = {"fingerprint": request["manifest"]["teacher_fingerprint"], "sample_key": key,
                           "sample_id": sample["id"], "target": {"kind": "preservation", "tokens": [4]} if sample["id"] == 1 and not all_rejected else None,
                           "reason": "accepted" if sample["id"] == 1 else "degenerate"}
                    P.save_cache(Path(request["cache"]) / f"{sample['id']}-{key}.json", row)
        else:
            metric = {"epoch": 1, "step": 1, "loss": .5, "learning_rate": .001, "seen": 1,
                      "accepted_presentations": 1, "trainable_parameters": 10,
                      "student_clips_per_second": 1, "updates_per_second": 1}
            (directory / "metrics.jsonl").write_text(json.dumps(metric) + '\n')
            P.atomic_json(directory / "result.json", {"updates": 1})
        on_update()
    monkeypatch.setattr(P, "supervise", run)
    for index in range(2):
        cfg = {**copy.deepcopy(DEFAULTS), "model": str(checkpoint), "devices": ["cuda:0", "cuda:1"]}
        directory = tmp_path / f"run-{index}"
        directory.mkdir()
        if all_rejected:
            with pytest.raises(ValueError, match="no accepted"):
                P.run_parallel(Ctx(Job(f"run-{index}", "phase4", cfg, directory)), cfg)
            assert calls == ["teacher"]
            return
        result = P.run_parallel(Ctx(Job(f"run-{index}", "phase4", cfg, directory)), cfg)
        assert result["updates"] == 1
        status = json.loads((directory / "monitor.json").read_text())
        assert status["stage"] == "complete"
        assert status["cache_hits"] == (0 if index == 0 else 2)
        assert len((directory / "metrics.jsonl").read_text().splitlines()) == 1
    assert calls == ["teacher", "distributed", "distributed"]


@pytest.fixture
def digit_cache(tmp_path, monkeypatch):
    from bwhisper import phase4, model
    from bwhisper.config import load_settings
    import transformers
    data = tmp_path / "data"
    monkeypatch.setattr(P, "DATA", data)
    monkeypatch.setattr(phase4, "RUNS_DIR", data / "runs")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(model, "unload_all", lambda: None)
    checkpoint = tmp_path / "base"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{}')
    samples = {}
    for sid in range(1, 6):
        audio = tmp_path / f"{sid}.wav"
        audio.write_bytes(bytes([sid]))
        samples[sid] = {"id": sid, "source": "fixture", "meta": {"recording": str(sid)}, "audio_path": str(audio)}
    manifest = {"train_ids": [1, 2, 3, 4], "eval_ids": [5], "cohort_run": "setup", "cohort_sha256": "fixture", "seed": 42}
    monkeypatch.setattr(phase4, "select_cohorts", lambda cfg: (samples, copy.deepcopy(manifest)))
    cfg = {**copy.deepcopy(DEFAULTS), "model": str(checkpoint), "devices": ["cuda:0", "cuda:1"],
           "target_style": "digits", "teacher_source_run": "phase4-source"}
    identity = {"model": P.checkpoint_identity(checkpoint), "settings": {k: load_settings()[k] for k in ("language", "timestamps")}}
    fingerprint = P.digest(identity)
    cache = data / "phase4_cache" / fingerprint
    cache.mkdir(parents=True)
    P.atomic_json(cache / "identity.json", identity)
    source = data / "runs" / cfg["teacher_source_run"]
    source.mkdir(parents=True)
    P.atomic_json(source / "job.json", {"type": "phase4", "status": "done", "config": {**cfg, "target_style": "cardinals"}})
    keys = {str(sid): P.sample_identity(sample) for sid, sample in samples.items()}
    P.atomic_json(source / "request.json", {"manifest": {**manifest, "teacher_fingerprint": fingerprint}, "sample_keys": keys})
    for sid, kind in [(1, "suppressed_cardinal"), (2, "natural_cardinal"), (3, "preservation"), (4, None)]:
        P.save_cache(cache / f"{sid}-{keys[str(sid)]}.json", {"sample_id": sid, "fingerprint": fingerprint,
            "sample_key": keys[str(sid)], "baseline": "20 people" if sid == 1 else "people",
            "target": {"kind": kind, "text": "people", "tokens": [4]} if kind else None,
            "reason": "accepted" if kind else "rest_match", "verdict": "cardinal" if sid == 1 else "no_digits"})
    class Tokenizer:
        eos_token_id = 100
        def encode(self, text, add_special_tokens):
            assert not add_special_tokens
            return [20, 21]
        def decode(self, tokens, skip_special_tokens):
            return "20 people"
    monkeypatch.setattr(transformers.WhisperTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    return cfg, source, samples, cache


def test_digits_derive_separate_cache_without_teacher_inference(digit_cache, tmp_path, monkeypatch):
    from bwhisper.jobs import Ctx, Job
    cfg, source, samples, old_cache = digit_cache
    original = {p.name: p.read_bytes() for p in old_cache.iterdir()}
    calls = []
    def supervise(ctx, commands, directory, timeout, on_update, **kwargs):
        assert directory.name == "distributed"
        request = json.loads((ctx.dir / "request.json").read_text())
        assert [r["target"]["kind"] for r in request["accepted"]] == ["natural_digits", "preservation"]
        assert request["accepted"][0]["target"]["tokens"] == [20, 21]
        assert Path(request["cache"]) != old_cache
        calls.append(directory.name)
        P.atomic_json(directory / "result.json", {"updates": 1, "target_style": "digits"})
        on_update()
    monkeypatch.setattr(P, "supervise", supervise)
    for i in range(2):
        directory = tmp_path / f"digit-run-{i}"
        directory.mkdir()
        result = P.run_parallel(Ctx(Job(directory.name, "phase4", cfg, directory)), cfg)
        assert result["target_style"] == "digits"
        assert json.loads((directory / "manifest.json").read_text())["target_style"] == "digits"
    assert calls == ["distributed", "distributed"]
    assert {p.name: p.read_bytes() for p in old_cache.iterdir()} == original


@pytest.mark.parametrize("damage", ["audio", "model", "filters", "identity", "direction"])
def test_digits_refuse_incompatible_source(digit_cache, tmp_path, damage):
    from bwhisper.jobs import Ctx, Job
    cfg, source, samples, cache = digit_cache
    if damage == "audio":
        Path(samples[1]["audio_path"]).write_bytes(b"changed")
    elif damage == "model":
        (Path(cfg["model"]) / "config.json").write_text('{"changed":true}')
    elif damage == "filters":
        cfg["min_rest_match"] = .5
    elif damage == "identity":
        P.atomic_json(cache / "identity.json", {"bad": True})
    else:
        job = json.loads((source / "job.json").read_text())
        job["config"]["target_style"] = "digits"
        P.atomic_json(source / "job.json", job)
    directory = tmp_path / "bad-digit-run"
    directory.mkdir()
    with pytest.raises(ValueError):
        P.run_parallel(Ctx(Job(directory.name, "phase4", cfg, directory)), cfg)


def probe(directory, world, extra=()):
    subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(world),
                    "tests/phase4_probe.py", str(directory), *extra],
                   check=True, capture_output=True, text=True, timeout=120,
                   env={**os.environ, "OMP_NUM_THREADS": "1"})


def adapter_tensors(directory):
    from safetensors.torch import load_file
    name = json.loads((directory / "resume.json").read_text())["path"]
    return load_file(str(directory / name / "adapter.safetensors"))


def test_ddp_update_parity_and_resume(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    probe(one, 1, ["--epochs", "2"])
    probe(two, 2, ["--epochs", "2"])
    for key, value in adapter_tensors(one).items():
        torch.testing.assert_close(value, adapter_tensors(two)[key], rtol=2e-4, atol=2e-5)
    stopped, resumed = tmp_path / "stopped", tmp_path / "resumed"
    probe(stopped, 2, ["--epochs", "2", "--stop-after-epoch"])
    probe(resumed, 2, ["--epochs", "2", "--resume", str(stopped)])
    for key, value in adapter_tensors(two).items():
        torch.testing.assert_close(value, adapter_tensors(resumed)[key], rtol=0, atol=0)
