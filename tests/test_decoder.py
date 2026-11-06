import numpy as np
import pytest
import torch

from bwhisper import model as M
from bwhisper.hebrew_numbers import auto_verdict, classify_word, count_mentions


@pytest.fixture(scope="module")
def L():
    try:
        return M.load_model("yoad/whisper-tiny", device="cpu")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"model unavailable: {e}")


def test_digit_tokens_found(L):
    assert len(L.digit_ids) > 100
    for i in L.digit_ids[:50].tolist():
        assert any(c.isdigit() for c in L.decode([i], skip_special=False))
    assert not L.digit_mask[L.tokenizer.convert_tokens_to_ids("<|he|>")]


def test_greedy_matches_generate(L):
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(16000 * 3) * 0.01).astype(np.float32)
    feats = M.features(L, audio)
    with torch.inference_mode():
        ref = L.model.generate(feats, language="he", task="transcribe", return_timestamps=False, max_new_tokens=20, do_sample=False)
    ref_text = L.tokenizer.decode(ref[0], skip_special_tokens=True).strip()
    mine, r = M.transcribe(L, audio, max_new=20)
    assert mine == ref_text


def test_rewind_matches_full_decode(L):
    rng = np.random.default_rng(1)
    audio = (rng.standard_normal(16000 * 3) * 0.01).astype(np.float32)
    enc = M.encode(L, audio)
    full = M.greedy(L, enc, list(L.sot_seq), max_new=12)
    k = min(4, len(full.tokens))
    rest = M.greedy(L, enc, list(L.sot_seq) + full.tokens[:k], max_new=12 - k)
    assert full.tokens[:k] + rest.tokens == full.tokens[: k + len(rest.tokens)]


def test_prefix_logits_matches_cached(L):
    audio = np.zeros(16000 * 2, dtype=np.float32)
    enc = M.encode(L, audio)
    ids = list(L.sot_seq) + [L.tokenizer.encode(" היום", add_special_tokens=False)[0]]
    a = M.prefix_logits(L, enc, ids)
    b, _ = M.decoder_logits(L, enc, torch.tensor([ids]))
    assert torch.allclose(a, b, atol=1e-4)


def test_number_lexicon():
    assert classify_word("עשרה") == "cardinal"
    assert classify_word("ובשלושה") == "cardinal"
    assert classify_word("10") == "digit"
    assert classify_word("בנובמבר") is None
    assert classify_word("הראשון") == "ordinal"
    assert count_mentions("היום 10 בנובמבר 2019")["digit_mentions"] == 2
    assert count_mentions("עשרים ושלוש אחוז")["word_mentions"] == 1
    v, d = auto_verdict("היום 10 בנובמבר", "היום עשרה בנובמבר")
    assert v == "cardinal" and d["rest_match"] == 1.0
    assert auto_verdict("היום 10 בנובמבר", "היום 10 בנובמבר")[0] == "still_digits"
    assert auto_verdict("היום 10 בנובמבר", "היום בנובמבר")[0] == "no_number_words"
