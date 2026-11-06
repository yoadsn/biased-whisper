import numpy as np
import pytest
import torch

from bwhisper import model as M
from bwhisper.phase2 import point_objective
from bwhisper.phase2n import Recorder, attribute_point, bh_fdr, component_layout


@pytest.fixture(scope="module")
def L():
    try:
        return M.load_model("yoad/whisper-tiny", device="cpu")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"model unavailable: {e}")


def test_bh_fdr():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert bh_fdr(p, 0.05).sum() == 2
    assert bh_fdr(np.ones(5), 0.05).sum() == 0


def test_grad_x_act_predicts_ablation(L):
    """First-order attribution of an MLP neuron ~ real change in the objective when it is zeroed."""
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(16000 * 3) * 0.05).astype(np.float32)
    tok = lambda s: L.tokenizer.encode(s, add_special_tokens=False)  # noqa: E731
    prefix = list(L.sot_seq) + tok(" היום")
    digit = int(L.digit_ids[0])
    pt = {"prefix_ids": prefix, "digit_token": {"id": digit}, "cardinal_token": {"id": tok(" עשרים")[0]}}
    lay = component_layout(L, ["decoder", "encoder"])
    mc = L.model.config
    hd = {"decoder": mc.d_model // mc.decoder_attention_heads, "encoder": mc.d_model // mc.encoder_attention_heads}
    rec = Recorder(L, lay)
    full, _, _, d0 = attribute_point(L, rec, lay, audio, pt, "mass", hd)
    rec.remove()
    preds, reals = [], []
    mlp = [s for s in lay if s["kind"] == "mlp"]
    for s in mlp[:4]:
        seg = full[s["offset"] : s["offset"] + s["n"]]
        for j in np.argsort(-np.abs(seg))[:4]:
            layer = (L.model.model.decoder if s["tower"] == "decoder" else L.model.model.encoder).layers[s["layer"]]
            h = layer.fc2.register_forward_pre_hook(lambda m, i, j=int(j): (i[0].index_fill(-1, torch.tensor([j]), 0.0),))
            with torch.inference_mode():
                d1 = float(point_objective(L, M.encode(L, audio), pt, "mass"))
            h.remove()
            preds.append(-float(seg[j]))
            reals.append(d1 - d0)
    assert np.corrcoef(preds, reals)[0, 1] > 0.8
