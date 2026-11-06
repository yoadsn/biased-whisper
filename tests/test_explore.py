import pytest


def test_episode_cap_persists_across_search_seeds(tmp_path, monkeypatch):
    import httpx
    from bwhisper.corpus import explore

    episode = {"uuid": "same-episode", "duration_sec": 600,
               "hits": [{"segment_idx": i * 3, "start_sec": i * 10,
                         "end_sec": i * 10 + 4, "text": f"מספר {i}"} for i in range(10)]}
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"episodes": [episode], "has_more": False}))
    monkeypatch.setattr(explore, "client", lambda cfg: httpx.Client(transport=transport))
    monkeypatch.setattr(explore, "load_cfg", lambda: {**explore.DEFAULT_CFG, "headers": {}})
    monkeypatch.setattr(explore, "load_settings", lambda: {"explore_base_url": "https://example.test"})
    existing = set()
    monkeypatch.setattr(explore.store, "exists", lambda source, key: key in existing)
    def add(source, key, *args, **kwargs):
        existing.add(key)
        return len(existing)
    monkeypatch.setattr(explore.store, "add_sample", add)
    monkeypatch.setattr(explore.store, "audio_path_for", lambda source, key: tmp_path / (key + ".wav"))
    monkeypatch.setattr(explore, "crop_episode_segments_range",
                        lambda client, url, duration, segs: ([(s, 4.0) for s in segs], 10))
    counts = {"same-episode": 1}  # includes pre-existing corpus, not just this fetch
    first = explore.fetch_episodes_batch(seed=1, n=10, per_episode=3, episode_counts=counts, log=lambda msg: None)
    second = explore.fetch_episodes_batch(seed=2, n=10, per_episode=3, episode_counts=counts, log=lambda msg: None)
    assert first["added"] == 2 and second["added"] == 0
    assert counts == {"same-episode": 3}

from bwhisper.audio import crop_url_to_wav
from bwhisper.corpus.explore import (
    dedupe_episode_hits,
    select_episode_segments,
    select_page_segments,
)


def test_select_page_segments_dedupe_duration_and_non_adjacent():
    fake_page = {
        "query": r"\d",
        "mode": "regex",
        "page": 1,
        "per_page": 40,
        "seed": 123,
        "has_more": True,
        "total_hits": 100,
        "episodes": [
            {
                "episode_idx": 1,
                "uuid": "ep-aaa",
                "source": "כאן 11",
                "title": "חדשות הערב",
                "date": "2024-11-01",
                "duration_sec": 3600.0,
                "hits": [
                    # Repeated hits on the same segment (one per digit char in "2024")
                    {"segment_idx": 10, "start_sec": 100.0, "end_sec": 106.5, "text": "בשנת 2024 היו בחירות"},
                    {"segment_idx": 10, "start_sec": 100.0, "end_sec": 106.5, "text": "בשנת 2024 היו בחירות"},
                    {"segment_idx": 10, "start_sec": 100.0, "end_sec": 106.5, "text": "בשנת 2024 היו בחירות"},
                    # Adjacent segment 11 -> should be skipped in favor of non-adjacent segment 14
                    {"segment_idx": 11, "start_sec": 106.5, "end_sec": 112.0, "text": "עם 80% הצבעה"},
                    # Too short (< 1.5s) -> skipped
                    {"segment_idx": 13, "start_sec": 120.0, "end_sec": 120.8, "text": "5"},
                    # Too long (> 28.0s) -> skipped
                    {"segment_idx": 14, "start_sec": 130.0, "end_sec": 165.0, "text": "100 שקלים"},
                    # Valid non-adjacent segment 15
                    {"segment_idx": 15, "start_sec": 170.0, "end_sec": 178.0, "text": "תקציב של 50 מיליון"},
                    # Third valid segment 20 -> beyond per_episode=2
                    {"segment_idx": 20, "start_sec": 210.0, "end_sec": 216.0, "text": "עוד 30 עובדים"},
                ],
            },
            {
                "episode_idx": 2,
                "uuid": "ep-bbb",
                "source": "גלצ",
                "title": "יומן הבוקר",
                "date": "2024-11-02",
                "duration_sec": 1800.0,
                "hits": [
                    # Only two adjacent segments available -> fallback still picks both for per_episode=2
                    {"segment_idx": 4, "start_sec": 20.0, "end_sec": 25.0, "text": "כביש 6 פתוח"},
                    {"segment_idx": 5, "start_sec": 25.0, "end_sec": 30.0, "text": "במרחק 12 קילומטר"},
                ],
            },
        ],
    }

    dedup1 = dedupe_episode_hits(fake_page["episodes"][0], min_dur=1.5, max_dur=28.0)
    assert [s["segment_idx"] for s in dedup1] == [10, 11, 15, 20]

    ep1_sel = select_episode_segments(fake_page["episodes"][0], per_episode=2)
    assert [s["segment_idx"] for s in ep1_sel] == [10, 15]
    assert [s["key"] for s in ep1_sel] == ["ep-aaa:10", "ep-aaa:15"]

    all_sel = select_page_segments(fake_page, per_episode=2)
    assert [s["key"] for s in all_sel] == ["ep-aaa:10", "ep-aaa:15", "ep-bbb:4", "ep-bbb:5"]


def test_crop_url_to_wav_error_does_not_leak_cookie(tmp_path):
    dst = tmp_path / "out.wav"
    secret_cookie = "super_secret_session_value_12345"
    with pytest.raises(RuntimeError) as exc_info:
        crop_url_to_wav(
            "http://127.0.0.1:1/nonexistent.opus",
            dst,
            start=1.0,
            end=3.0,
            headers={
                "Cookie": f"session={secret_cookie}",
                "User-Agent": "test-agent",
                "Referer": "https://explore.ivrit.ai/",
            },
        )
    msg = str(exc_info.value)
    assert "Cookie" not in msg
    assert "cookie" not in msg.lower()
    assert secret_cookie not in msg


def test_new_explore_helpers_never_leak_secrets(tmp_path):
    import httpx
    from bwhisper.corpus.explore import (
        crop_episode_segments_full,
        crop_episode_segments_range,
        public_cfg,
    )

    secret_cookie = "top_secret_google_session_token_999"
    secret_auth = "Bearer secret_bearer_xyz"
    pub = public_cfg({"cookie": secret_cookie, "headers": {"Authorization": secret_auth, "Accept": "application/json"}})
    assert secret_cookie not in str(pub)
    assert secret_auth not in str(pub)

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="service unavailable")

    transport = httpx.MockTransport(mock_handler)
    seg = {"key": "ep:1", "c_start": 1.0, "c_end": 3.0, "dst": tmp_path / "x.wav"}
    with httpx.Client(transport=transport, headers={"Cookie": secret_cookie, "Authorization": secret_auth}) as c:
        for fn in (
            lambda: crop_episode_segments_range(c, "https://explore.ivrit.ai/audio/ep.opus", 100.0, [seg]),
            lambda: crop_episode_segments_full(c, "https://explore.ivrit.ai/audio/ep.opus", [seg]),
        ):
            with pytest.raises(Exception) as exc_info:
                fn()
            err_str = str(exc_info.value)
            assert secret_cookie not in err_str
            assert secret_auth not in err_str
            assert "Cookie" not in err_str
