"""Live Zipformer decode keeps an encoder cache and does not rewrite earlier audio."""

import torch

from decoding.zipformer_greedy import greedy_search_batch, greedy_search_continue, initial_context
from decoding.zipformer_stream import (
    EMBED_CHUNK,
    FEATURE_CHUNK,
    FEATURE_SHIFT,
    ZipformerStream,
    _bandlimit,
    stable_16k,
)
from models.features import SAMPLE_RATE, log_mel


def _tone(seconds: float, freq: float, amp: float, dc: float, sr: int) -> torch.Tensor:
    n = int(seconds * sr)
    t = torch.arange(n, dtype=torch.float32) / sr
    return amp * torch.sin(2 * torch.pi * freq * t) + dc


def test_feature_prefix_is_stable_when_audio_grows():
    sr = 48000
    torch.manual_seed(0)
    native = torch.randn(sr * 2) * 0.1
    half = sr
    early = stable_16k(native[:half], sr, final=False)
    later = stable_16k(native, sr, final=False)
    done = stable_16k(native, sr, final=True)
    assert early.numel() > 0
    assert torch.equal(early, later[: early.numel()])
    assert torch.equal(later, done[: later.numel()])
    early_mel = log_mel(_bandlimit(early), dither=0.0)
    later_mel = log_mel(_bandlimit(later), dither=0.0)
    # fbank itself moves finished frames by ~1e-6 when the clip grows.
    # The stream keeps the rows it already fed to the encoder.
    assert torch.allclose(early_mel, later_mel[: early_mel.size(0)], atol=1e-4, rtol=1e-4)


def test_short_audio_is_buffered_and_retries_do_not_double_count():
    sess = ZipformerStream(None, None, torch.device("cpu"))
    try:
        sess.add(torch.zeros(10), SAMPLE_RATE, offset=4)
        raise AssertionError("gap should fail")
    except ValueError as exc:
        assert "gap" in str(exc)
    n = int(0.4 * SAMPLE_RATE)
    first = _tone(0.4, 220, 0.1, 0.2, SAMPLE_RATE)
    assert sess.add(first, SAMPLE_RATE, offset=0) == n
    assert sess.encoder_out is None
    assert sess.add(first, SAMPLE_RATE, offset=0) == n
    extra = _tone(0.05, 180, 0.1, 0.0, SAMPLE_RATE)
    assert sess.add(torch.cat([first, extra]), SAMPLE_RATE, offset=0) == n + extra.numel()


def test_cached_chunks_match_one_shot_and_do_not_rewrite():
    pytest = __import__("pytest")
    pytest.importorskip("k2")
    from models.zipformer_icefall import build_zipformer
    from train_zipformer import set_batch_count
    from utils import ROOT, load_yaml

    cfg = load_yaml(ROOT / "configs/zipformer_s_streaming_pool.yaml")
    torch.manual_seed(0)
    model = build_zipformer(cfg, 40).eval()
    set_batch_count(model, 20000.0)
    device = torch.device("cpu")

    encoder_out = torch.randn(1, 9, max(cfg["model"]["encoder_dim"]))
    lens = torch.tensor([encoder_out.size(1)])
    with torch.inference_mode():
        full = greedy_search_continue(model, encoder_out, initial_context(model))
        ctx = initial_context(model)
        ids = []
        for part in (encoder_out[:, :4], encoder_out[:, 4:]):
            step, ctx = greedy_search_continue(model, part, ctx)
            ids.extend(step)
        assert ids == full[0]
        assert torch.equal(ctx, full[1])
        assert greedy_search_batch(model, encoder_out, lens)[0] == full[0]

        # DC on the first half would move every sample if the frontend
        # subtracted the mean of the whole clip.
        head = _tone(1.3, 220, 0.12, 0.25, SAMPLE_RATE)
        tail = _tone(1.3, 180, 0.12, 0.0, SAMPLE_RATE)
        whole = torch.cat([head, tail])
        grew = ZipformerStream(model, None, device)
        assert grew.add(head[: int(0.4 * SAMPLE_RATE)], SAMPLE_RATE, offset=0) == int(0.4 * SAMPLE_RATE)
        assert grew.encoder_out is None
        grew.add(head, SAMPLE_RATE, offset=0)
        assert grew.encoder_out is not None
        assert grew.feat_done > 0 and grew.feat_done % FEATURE_SHIFT == 0
        assert grew.encoder_out.size(1) == (grew.feat_done // FEATURE_SHIFT) * ((EMBED_CHUNK + 1) // 2)
        partial = grew.encoder_out.clone()
        partial_tokens = list(grew.tokens)
        grew.add(whole, SAMPLE_RATE, offset=0)
        assert torch.equal(grew.encoder_out[:, : partial.size(1)], partial)
        assert grew.tokens[: len(partial_tokens)] == partial_tokens
        # A retried body that adds nothing does not run another chunk.
        frames = grew.encoder_out.shape[1]
        grew.add(whole, SAMPLE_RATE, offset=0)
        assert grew.encoder_out.shape[1] == frames

        once = ZipformerStream(model, None, device)
        once.add(whole, SAMPLE_RATE, offset=0)
        assert torch.equal(once.encoder_out, grew.encoder_out)
        assert once.tokens == grew.tokens

        grew.add(whole, SAMPLE_RATE, offset=0, final=True)
        once.add(whole, SAMPLE_RATE, offset=0, final=True)
        assert grew.encoder_out.size(1) > partial.size(1)
        assert torch.equal(grew.encoder_out[:, : partial.size(1)], partial)
        assert torch.equal(once.encoder_out, grew.encoder_out)
        assert grew.flushed
        frozen = grew.encoder_out.clone()
        grew.add(torch.cat([whole, torch.zeros(SAMPLE_RATE)]), SAMPLE_RATE, offset=0, final=True)
        assert torch.equal(grew.encoder_out, frozen)
        assert grew.feat_done >= FEATURE_CHUNK
