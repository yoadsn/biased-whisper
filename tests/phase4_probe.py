"""Synthetic multi-process validation/benchmark. Never reads corpus audio."""
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers import WhisperConfig, WhisperForConditionalGeneration

from bwhisper.phase4 import DEFAULTS
from bwhisper.phase4_parallel import atomic_json
from bwhisper.phase4_worker import Telemetry, train_engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--stop-after-epoch", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--encoder", type=int, default=0)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.set_num_threads(1)
    device = f"cuda:{rank}" if args.cuda else "cpu"
    if args.cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if args.cuda else "gloo", timeout=timedelta(seconds=120),
                            **({"device_id": torch.device(device)} if args.cuda else {}))
    run = Path(args.directory)
    if rank == 0:
        (run / "distributed").mkdir(parents=True, exist_ok=True)
    dist.barrier()
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(rank=2, alpha=4., dropout=0., bf16=args.bf16, epochs=args.epochs,
               learning_rate=.01, weight_decay=0., warmup_steps=0, max_grad_norm=1000.,
               encoder_top_layers=args.encoder, checkpoint_every_steps=1)
    torch.manual_seed(123)
    model = WhisperForConditionalGeneration(WhisperConfig(
        vocab_size=32, num_mel_bins=8, d_model=16, encoder_layers=2, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
        decoder_ffn_dim=32, max_source_positions=16, max_target_positions=32,
        pad_token_id=0, bos_token_id=1, eos_token_id=31, decoder_start_token_id=1,
        suppress_tokens=[], begin_suppress_tokens=[], dropout=0., attention_dropout=0., activation_dropout=0.,
    )).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    L = SimpleNamespace(model=model, device=device, name="synthetic-whisper", sot_seq=[1, 3], eos=31, vocab=32)
    ids = list(range(1, 12))  # global tail of three: exercises zero-masked slots on 4/8 ranks
    request = {"cfg": cfg, "manifest": {"accepted_ids": ids, "eval_ids": [], "training_fingerprint": f"synthetic-{world}"},
               "accepted": [{"sample_id": i, "target": {"tokens": [4 + i % 3, 9, 10]}} for i in ids]}
    if args.resume:
        previous = Path(args.resume)
        request["resume_checkpoint"] = str(previous / json.loads((previous / "resume.json").read_text())["path"])
    telemetry = Telemetry(run / "distributed", rank, device)
    def evaluate(L, adapter, epoch, step):
        if args.stop_after_epoch and epoch == 1:
            (run / "distributed/STOP").touch()
        return {"epoch": epoch, "step": step, "adapter": {"n": 0}, "baseline": {"n": 0}}
    started = time.monotonic()
    try:
        train_engine(L, request, run, rank, world, telemetry,
                     feature_fn=lambda sid: torch.full((1, 8, 32), float(sid)/11, device=device), evaluate_fn=evaluate)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        if rank == 0:
            atomic_json(run / "probe.json", {"world_size": world, "cuda": args.cuda,
                        "bf16": args.bf16, "encoder": args.encoder, "engine_seconds": time.monotonic() - started})
    finally:
        telemetry.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
