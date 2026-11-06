import subprocess
from pathlib import Path

import numpy as np
import pytest

from bwhisper.audio import (
    crop_to_wav,
    decode_mini_ogg_to_wav,
    load_audio,
    ogg_page_time_bounds,
    parse_ogg_pages,
    parse_opus_header,
    select_ogg_pages_for_range,
)
from bwhisper.corpus import explore, store


@pytest.fixture(scope="module")
def sample_opus(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("ogg")
    out = d / "sine.opus"
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
        "-c:a", "libopus", "-b:a", "32k", str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def _best_alignment(a: np.ndarray, b: np.ndarray, max_lag: int = 160) -> tuple[float, float, int]:
    """Return (max_corr, mad_at_max_corr, best_lag_samples) across lags in [-max_lag, +max_lag]."""
    best_corr = -1.0
    best_mad = 1e9
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[lag:], b[: len(a) - lag]
        else:
            x, y = a[: len(a) + lag], b[-lag:]
        m = min(len(x), len(y))
        if m < 100:
            continue
        cc = float(np.corrcoef(x[:m], y[:m])[0, 1])
        if cc > best_corr:
            best_corr = cc
            best_mad = float(np.mean(np.abs(x[:m] - y[:m])))
            best_lag = lag
    return best_corr, best_mad, best_lag


def test_ogg_page_parser_and_selection(sample_opus: Path, tmp_path: Path):
    data = sample_opus.read_bytes()
    pages = parse_ogg_pages(data)
    assert len(pages) >= 15
    # Exact contiguous page coverage of the entire file
    assert pages[0].offset == 0
    assert sum(p.length for p in pages) == len(data)
    for i in range(1, len(pages)):
        assert pages[i].offset == pages[i - 1].offset + pages[i - 1].length
        assert pages[i].seq == pages[i - 1].seq + 1
        assert pages[i].granule >= pages[i - 1].granule

    hdr_bytes, pre_skip, serial = parse_opus_header(data[:8192])
    assert pre_skip > 0
    assert len(hdr_bytes) > 40

    audio_pages = [p for p in pages if p.granule > 0]
    for i in range(1, len(audio_pages)):
        assert audio_pages[i].granule > audio_pages[i - 1].granule

    # Select pages covering [6.2, 11.7]
    t_start, t_end = 6.2, 11.7
    sel = select_ogg_pages_for_range(pages, pre_skip, t_start, t_end, is_file_start=True, is_file_end=True)
    assert sel is not None
    sel_pages, t0 = sel
    bounds = ogg_page_time_bounds(sel_pages, pre_skip)
    assert bounds is not None
    assert t0 <= t_start - 0.1
    assert bounds[1] >= t_end

    # Decode mini-Ogg and compare against local crop_to_wav
    wav_mini = tmp_path / "mini.wav"
    wav_ref = tmp_path / "ref.wav"
    dur_mini = decode_mini_ogg_to_wav(hdr_bytes, sel_pages, t0, t_start, t_end, wav_mini)
    dur_ref = crop_to_wav(sample_opus, wav_ref, t_start, t_end)
    assert abs(dur_mini - (t_end - t_start)) < 0.05
    assert abs(dur_mini - dur_ref) < 0.05

    a_mini = load_audio(wav_mini)
    a_ref = load_audio(wav_ref)
    corr, mad, _ = _best_alignment(a_mini, a_ref)
    assert mad < 1e-3
    assert corr > 0.999


def test_live_range_vs_reference_five_segments(tmp_path: Path):
    cfg = explore.load_cfg()
    if not cfg.get("cookie"):
        pytest.skip("no explore cookie configured")
    auth = explore.check_auth()
    if not auth.get("ok"):
        pytest.skip(f"explore auth unavailable: {auth.get('reason')}")

    # Pick 5 existing explore segments stored by the reference crop_url_to_wav path
    rows = [
        r for r in store.query(source="explore", limit=20)
        if r.get("start") is not None and r.get("end") is not None and Path(r["audio_path"]).exists()
    ][:5]
    if len(rows) < 5:
        pytest.skip("fewer than 5 reference explore rows in store")

    base = "https://explore.ivrit.ai"
    with explore.client(cfg) as c:
        for i, r in enumerate(rows):
            uuid = r["meta"]["uuid"]
            aurl = f"{base}/audio/{uuid}.opus"
            c_start = max(0.0, float(r["start"]) - 0.25)
            c_end = float(r["end"]) + 0.25
            dst = tmp_path / f"range_{i}.wav"
            seg = {"key": r["source_key"], "c_start": c_start, "c_end": c_end, "dst": dst}
            cropped, b_read = explore.crop_episode_segments_range(c, aurl, None, [seg])
            assert len(cropped) == 1
            _, dur_range = cropped[0]
            dur_ref = float(r["duration"])
            assert abs(dur_range - dur_ref) <= 0.1

            a_range = load_audio(dst)
            a_ref = load_audio(r["audio_path"])
            corr, mad, _ = _best_alignment(a_range, a_ref)
            assert corr > 0.99, f"low cross-correlation peak {corr} on {r['source_key']}"
            assert mad < 0.01, f"high MAD {mad} on {r['source_key']}"
