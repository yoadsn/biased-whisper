import numpy as np
import pytest
import torch

from bwhisper import model as M
from bwhisper.phase1 import StyleTokens, first_divergence, is_hyphen_text, is_prefix_text, reconvergence
from bwhisper.phase2 import point_objective


@pytest.fixture(scope="module")
def L():
    try:
        return M.load_model("yoad/whisper-tiny", device="cpu")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"model unavailable: {e}")


def test_text_predicates():
    assert is_hyphen_text("-") and is_hyphen_text("־") and is_hyphen_text(" ב-")
    assert not is_hyphen_text("ב") and not is_hyphen_text("")
    assert is_prefix_text(" ב") and is_prefix_text(" מה") and is_prefix_text("ו")
    assert not is_prefix_text(" בא") and not is_prefix_text(" 20")


def test_divergence_region():
    a = [1, 2, 3, 9, 9, 4, 5, 6, 7]
    b = [1, 2, 3, 8, 8, 8, 4, 5, 6, 7]
    d = first_divergence(a, b)
    assert d == 3
    assert reconvergence(a, b, d) == (5, 6)
    assert reconvergence([1, 2], [1, 3], 1) == (2, 2)


def test_span_start_hyphen_prefix(L):
    st = StyleTokens(L)
    toks = L.tokenizer.encode(" היום ב-2025 אנחנו", add_special_tokens=False)
    d_pos = next(i for i, t in enumerate(toks) if bool(L.digit_mask[t]))
    s = st.span_start(toks, d_pos, L.decode)
    assert L.decode(toks[s:d_pos], skip_special=False).strip() == "ב-"
    ban = st.step_ban()
    assert ban(toks[: s + 1])  # right after the prefix token, hyphens are banned
    assert not ban(toks[:s])
    pct = L.tokenizer.encode(" 80%", add_special_tokens=False)
    assert pct[-1] in st.symbols


def test_seq_logprobs_matches_stepwise(L):
    audio = np.zeros(16000 * 2, dtype=np.float32)
    enc = M.encode(L, audio)
    ids = list(L.sot_seq) + L.tokenizer.encode(" היום ב-2025", add_special_tokens=False)
    lp = M.seq_logprobs(L, enc, ids)
    for j in range(len(L.sot_seq) - 1, len(ids) - 1):
        step = torch.log_softmax(M.prefix_logits(L, enc, ids[: j + 1]).float(), -1)[ids[j + 1]]
        assert torch.allclose(lp[j], step, atol=1e-4)


def test_seq_contrast_objective(L):
    audio = np.zeros(16000 * 2, dtype=np.float32)
    enc = M.encode(L, audio)
    tok = lambda s: L.tokenizer.encode(s, add_special_tokens=False)  # noqa: E731
    y_d, y_c = tok(" היום ב-2025 אנחנו"), tok(" היום באלפיים עשרים וחמש אנחנו")
    d = first_divergence(y_d, y_c)
    ed, ec = reconvergence(y_d, y_c, d)
    pt = {"y_digit": y_d, "y_cardinal": y_c, "div_digit": [d, ed], "div_cardinal": [d, ec]}
    n0 = len(L.sot_seq)
    lp_d = M.seq_logprobs(L, enc, list(L.sot_seq) + y_d)[n0 + d - 1 : n0 + ed - 1].sum()
    lp_c = M.seq_logprobs(L, enc, list(L.sot_seq) + y_c)[n0 + d - 1 : n0 + ec - 1].sum()
    assert torch.allclose(point_objective(L, enc, pt, "auto"), lp_d - lp_c, atol=1e-4)


def test_merging_prefixes_and_maqaf(L):
    st = StyleTokens(L)
    tid = lambda s: L.tokenizer.encode(s, add_special_tokens=False)[0]  # noqa: E731
    assert tid(" ב") in st.merging and tid(" ל") in st.merging
    assert tid(" ו") not in st.merging  # ' ותשעה' -> [' ו', 'ת', ...]
    maqaf = L.tokenizer.encode(" ב־2025", add_special_tokens=False)[1]
    assert maqaf in st.hyphen and maqaf in st.d6
    toks = L.tokenizer.encode(" היום ב-2025", add_special_tokens=False)
    d_pos = next(i for i, t in enumerate(toks) if bool(L.digit_mask[t]))
    s = st.span_start(toks, d_pos, L.decode)
    ban = st.step_ban(prefix_len=s, span_tok=toks[s])
    assert toks[s] in ban(toks[:s])
