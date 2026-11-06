from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE


def load_audio(path: str | Path, start: float | None = None, end: float | None = None) -> np.ndarray:
    """Decode (a time range of) any ffmpeg-readable file to 16 kHz mono float32."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def crop_to_wav(src: str | Path, dst: str | Path, start: float | None, end: float | None) -> float:
    """Crop [start, end] of src to a 16 kHz mono wav at dst. Returns duration."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.decode('utf-8', 'replace')[:200]}")
    return wav_duration(dst)


def wav_duration(path: str | Path) -> float:
    import wave

    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def crop_url_to_wav(
    url: str,
    dst: str | Path,
    start: float | None,
    end: float | None,
    headers: dict[str, str] | None = None,
) -> float:
    """Crop [start, end] of a remote audio URL via HTTP Range requests to a 16 kHz mono wav.
    Never includes headers or command line in error messages."""
    import re

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-multiple_requests", "1"]
    if headers:
        hdr_str = "".join(f"{k}: {v}\r\n" for k, v in headers.items() if v)
        if hdr_str:
            cmd += ["-headers", hdr_str]
    if start is not None:
        cmd += ["-ss", f"{max(0.0, float(start)):.3f}"]
    if end is not None:
        cmd += ["-to", f"{float(end):.3f}"]
    cmd += ["-i", str(url), "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=90)
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg timed out after 90s") from None
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", "replace")
        err = re.sub(r"(?i)cookie[^\r\n]*", "<redacted>", err)
        raise RuntimeError(f"ffmpeg failed ({r.returncode}): {err.strip()[:200]}")
    return wav_duration(dst)


class OggPage:
    __slots__ = ("offset", "length", "header_type", "granule", "serial", "seq", "raw")

    def __init__(
        self,
        offset: int,
        length: int,
        header_type: int,
        granule: int,
        serial: int,
        seq: int,
        raw: bytes,
    ):
        self.offset = offset
        self.length = length
        self.header_type = header_type
        self.granule = granule
        self.serial = serial
        self.seq = seq
        self.raw = raw


def parse_ogg_pages(buf: bytes, expected_serial: int | None = None) -> list[OggPage]:
    """Parse complete Ogg pages from an arbitrary byte buffer."""
    import struct

    pages: list[OggPage] = []
    n = len(buf)
    pos = 0
    while pos <= n - 27:
        idx = buf.find(b"OggS", pos)
        if idx < 0 or idx > n - 27:
            break
        ver, htype, gran, serial, seq, _crc, nseg = struct.unpack_from("<BBqIIIB", buf, idx + 4)
        if ver != 0 or (htype & ~0x07) != 0:
            pos = idx + 4
            continue
        if expected_serial is not None and serial != expected_serial:
            pos = idx + 4
            continue
        hdr_end = idx + 27 + nseg
        if hdr_end > n:
            break
        body_len = sum(buf[idx + 27 : hdr_end])
        page_end = hdr_end + body_len
        if page_end > n:
            break
        if expected_serial is None and page_end + 4 <= n and buf[page_end : page_end + 4] != b"OggS":
            pos = idx + 4
            continue
        pages.append(
            OggPage(
                offset=idx,
                length=page_end - idx,
                header_type=htype,
                granule=gran,
                serial=serial,
                seq=seq,
                raw=buf[idx:page_end],
            )
        )
        pos = page_end
    return pages


def parse_opus_header(buf: bytes) -> tuple[bytes, int, int]:
    """Extract (header_pages_bytes, pre_skip, serial) from the start of an Ogg Opus stream."""
    import struct

    pages = parse_ogg_pages(buf)
    if not pages:
        raise ValueError("No Ogg pages found in header buffer")
    bos = pages[0]
    nseg = bos.raw[26]
    payload = bos.raw[27 + nseg :]
    if not payload.startswith(b"OpusHead") or len(payload) < 19:
        raise ValueError("First Ogg page is not an OpusHead header")
    pre_skip = struct.unpack_from("<H", payload, 10)[0]
    serial = bos.serial
    hdr_pages = []
    for p in pages:
        if p.serial == serial and p.granule == 0:
            hdr_pages.append(p.raw)
        elif p.serial == serial and p.granule > 0:
            break
    if len(hdr_pages) < 2:
        raise ValueError("Incomplete Ogg Opus headers (missing OpusTags page)")
    return b"".join(hdr_pages), int(pre_skip), int(serial)


def ogg_page_time_bounds(
    pages: list[OggPage], pre_skip: int
) -> tuple[float, float] | None:
    """Return (first_audio_page_end_sec, last_audio_page_end_sec) for parsed pages."""
    audio = [p for p in pages if p.granule > 0]
    if not audio:
        return None
    return (
        (audio[0].granule - pre_skip) / 48000.0,
        (audio[-1].granule - pre_skip) / 48000.0,
    )


def select_ogg_pages_for_range(
    pages: list[OggPage],
    pre_skip: int,
    start_sec: float,
    end_sec: float,
    is_file_start: bool = False,
    is_file_end: bool = False,
    preroll: float = 0.1,
) -> tuple[list[OggPage], float] | None:
    """Select consecutive audio pages covering [start_sec - preroll, end_sec] and return
    (selected_pages, t0_sec) where t0_sec is the exact stream time of the first sample
    decoded by ffmpeg when `header_bytes + selected_pages` is decoded.
    Returns None if `pages` does not cover the requested range."""
    audio = [p for p in pages if p.granule > 0]
    if not audio:
        return None
    want_start = max(0.0, float(start_sec) - preroll)
    want_end = float(end_sec)
    first_end = (audio[0].granule - pre_skip) / 48000.0
    last_end = (audio[-1].granule - pre_skip) / 48000.0

    if not is_file_start and want_start > 0.0 and first_end > want_start:
        return None
    if not is_file_end and last_end < want_end:
        return None

    # Find anchor page ending <= want_start
    if is_file_start and first_end > want_start:
        k = 0
    else:
        anchor = 0
        for i, p in enumerate(audio):
            if (p.granule - pre_skip) / 48000.0 <= want_start:
                anchor = i
            else:
                break
        k = anchor + 1 if first_end <= want_start else 0

    min_k = 0 if is_file_start else 1
    while k > min_k and (audio[k].header_type & 0x01) != 0:
        k -= 1
    if (audio[k].header_type & 0x01) != 0 and k > 0:
        k -= 1

    m = len(audio)
    for i in range(k, len(audio)):
        if (audio[i].granule - pre_skip) / 48000.0 >= want_end:
            # Include one extra page if available for resampler lookahead
            m = min(len(audio), i + 2)
            break

    g_before = audio[k - 1].granule if k > 0 else 0
    t0 = ((g_before - pre_skip) / 48000.0) if k > 0 else 0.0
    return audio[k:m], t0


def write_pcm16_wav(dst: str | Path, pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Write a 1-D int16 numpy array to a mono WAV file and return its duration in seconds."""
    import wave

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    pcm16 = np.ascontiguousarray(pcm, dtype=np.int16)
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    return len(pcm16) / float(sample_rate)


def decode_mini_ogg_to_pcm16(
    header_bytes: bytes,
    pages: list[OggPage],
    t0: float,
    start_sec: float,
    end_sec: float,
) -> np.ndarray:
    """Decode `header_bytes + pages` via an in-memory ffmpeg pipe and slice to [start_sec, end_sec]."""
    mini = header_bytes + b"".join(p.raw for p in pages)
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin",
        "-i", "pipe:0",
        "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
    ]
    r = subprocess.run(cmd, input=mini, capture_output=True, timeout=30)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", "replace")[:200]
        raise RuntimeError(f"ffmpeg mini-ogg decode failed: {err}")
    arr = np.frombuffer(r.stdout, dtype=np.int16)
    s_idx = max(0, int(round((start_sec - t0) * SAMPLE_RATE)))
    want_len = max(1, int(round((end_sec - start_sec) * SAMPLE_RATE)))
    return arr[s_idx : s_idx + want_len].copy()


def decode_mini_ogg_to_wav(
    header_bytes: bytes,
    pages: list[OggPage],
    t0: float,
    start_sec: float,
    end_sec: float,
    dst: str | Path,
) -> float:
    """Decode a mini-Ogg slice and write it to a 16 kHz mono WAV at `dst`."""
    pcm = decode_mini_ogg_to_pcm16(header_bytes, pages, t0, start_sec, end_sec)
    return write_pcm16_wav(dst, pcm, SAMPLE_RATE)


