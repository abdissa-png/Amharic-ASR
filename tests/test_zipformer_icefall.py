"""The streaming Zipformer pipeline matches the icefall recipe's wiring."""

from __future__ import annotations

import pytest
import torch

from models.zipformer_icefall import (
    encoder_frames,
    labels_ok,
    loss_scales,
)
from utils import ROOT, load_yaml


def test_rnnt_decode_runs_on_a_subset_of_epochs():
    from train_zipformer import _min_recorded, rnnt_decode_epoch

    assert [e for e in range(1, 31) if rnnt_decode_epoch(e, 30, 5)] == [1, 5, 10, 15, 20, 25, 30]
    assert all(rnnt_decode_epoch(e, 30, 1) for e in range(1, 31))
    history = [
        {"dev_ctc_wer": 0.9},
        {"dev_ctc_wer": None},
        {"dev_ctc_wer": float("nan")},
        {"dev_ctc_wer": 0.4},
    ]
    assert _min_recorded(history, "dev_ctc_wer") == 0.4


def test_waveform_augment_starts_at_its_epoch(tmp_path):
    from data.dataset import AsrDataset
    from train_zipformer import _rebuild, _start_waveform_augment
    from utils import write_jsonl

    path = tmp_path / "train.jsonl"
    write_jsonl(path, [{"clip_id": str(i), "duration_s": 2.0} for i in range(6)])
    cfg = {"seed": 0, "train": {"num_workers": 0, "batch_seconds": 4}}
    ds = AsrDataset(path, None, train=True, max_duration=40.0)
    loader = _rebuild(torch.utils.data.DataLoader(ds), cfg, True)
    loader.batch_sampler.seed = 7
    augment = object()

    assert _start_waveform_augment(loader, augment, 1, 2, cfg) is loader
    assert ds.augment is None
    on = _start_waveform_augment(loader, augment, 2, 2, cfg)
    assert on is not loader and on.dataset.augment is augment
    assert on.batch_sampler.seed == 7
    assert _start_waveform_augment(on, augment, 3, 2, cfg) is on
    assert _start_waveform_augment(loader, None, 5, 2, cfg) is loader


def _whiten_grad(device: str, limit: float):
    load_icefall_modules()
    from scaling import Whiten, WhiteningPenaltyFunction

    w = Whiten(num_groups=1, whitening_limit=limit, prob=(0.025, 0.25), grad_scale=0.1)
    torch.manual_seed(0)
    base = torch.randn(400, 1, device=device)
    x = (base * torch.ones(1, 16, device=device) + 0.01 * torch.randn(400, 16, device=device))
    x.requires_grad_(True)
    g = torch.randn(400, 16, device=device)
    WhiteningPenaltyFunction.apply(x, w).backward(g)
    return w, x.grad, g


def load_icefall_modules():
    from models.zipformer_icefall import load_icefall

    load_icefall()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_whiten_penalty_selected_without_a_host_branch(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    # Rank-1 features are far from white: metric ~16 against a limit of 2.
    w, grad, g = _whiten_grad(device, limit=2.0)
    assert not torch.allclose(grad, g)
    w2, grad2, g2 = _whiten_grad(device, limit=1000.0)
    assert torch.equal(grad2, g2)
    if device == "cuda":
        torch.cuda.synchronize()
    w.snapshot_schedule()
    w2.snapshot_schedule()
    assert w.prob == w.max_prob and w._apply_prob_py == w.max_prob
    assert w2.prob == w2.min_prob and w2._apply_prob_py == w2.min_prob


def test_fill_scalar_writes_only_on_change():
    load_icefall_modules()
    from scaling import fill_scalar

    m = torch.nn.Module()
    m.register_buffer("_v", torch.zeros(()), persistent=False)
    fill_scalar(m, "_v", 0.5)
    assert float(m._v) == 0.5
    m._v.zero_()
    fill_scalar(m, "_v", 0.5)
    assert float(m._v) == 0.0
    fill_scalar(m, "_v", 0.25)
    assert float(m._v) == 0.25


def test_set_batch_count_reaches_every_scheduled_module():
    pytest.importorskip("k2")
    from models.zipformer_icefall import build_zipformer
    from train_zipformer import set_batch_count

    cfg = load_yaml(ROOT / "configs/zipformer_s_streaming_pool.yaml")
    model = build_zipformer(cfg, 50)
    set_batch_count(model, 3.0)
    set_batch_count(model, 7.0)
    counted = [m for m in model.modules() if hasattr(m, "batch_count")]
    assert counted and all(m.batch_count == 7.0 for m in counted)
    named = dict(model.named_modules())
    assert all(m.name == n for n, m in named.items() if hasattr(m, "name") and n)


def test_encoder_frames_match_icefall_filter():
    # zipformer/train.py: T = ((num_frames - 7) // 2 + 1) // 2
    assert encoder_frames(400) == ((400 - 7) // 2 + 1) // 2
    assert labels_ok(400, encoder_frames(400) - 2)
    assert not labels_ok(400, encoder_frames(400) - 1)


def test_loss_scales_match_compute_loss():
    tcfg = {"warm_step": 2000, "simple_loss_scale": 0.5, "ctc_loss_scale": 0.2}
    assert loss_scales(0, tcfg) == (1.0, 0.1, 0.2)
    assert loss_scales(2000, tcfg) == (0.5, 1.0, 0.2)
    mid = loss_scales(1000, tcfg)
    assert mid[0] == pytest.approx(0.75)
    assert mid[1] == pytest.approx(0.55)
    assert loss_scales(0, {**tcfg, "warm_step": 0}) == (0.5, 1.0, 0.2)


def test_streaming_configs_are_zipformer_s():
    et = load_yaml(ROOT / "configs/zipformer_s_streaming_et.yaml")
    pool = load_yaml(ROOT / "configs/zipformer_s_streaming_pool.yaml")
    assert et["model"]["num_encoder_layers"] == [2, 2, 2, 2, 2, 2]
    assert et["model"]["encoder_dim"] == [192, 256, 256, 256, 256, 256]
    assert et["model"]["downsampling_factor"] == [1, 2, 4, 8, 4, 2]
    assert et["model"]["cnn_module_kernel"] == [31, 31, 15, 15, 15, 31]
    assert et["model"]["causal"] is True
    assert 64 not in et["model"]["left_context_frames"]
    assert et["train"]["base_lr"] == 0.045
    assert et["train"]["ctc_loss_scale"] == 0.2
    assert et["paths"]["metadata_dir"] == "metadata"
    assert pool["paths"]["metadata_dir"] == "metadata/pool_clean"
    assert pool["model"]["encoder_dim"] == et["model"]["encoder_dim"]
    assert pool["train"]["warm_step"] == 0
    assert pool["paths"]["exp_dir"] != et["paths"]["exp_dir"]

    align = load_yaml(ROOT / "configs/zipformer_s_streaming_et_align.yaml")
    assert pool["paths"]["init_from"] == align["paths"]["exp_dir"] + "/best.pt"
    for key in ("ctc_loss_scale", "prune_range", "lm_scale", "am_scale", "simple_loss_scale"):
        assert pool["train"][key] == align["train"][key], key
    assert pool["model"] == align["model"]
    assert pool["train"]["augment"]["gain_db"] is False
    assert pool["train"]["augment_start_epoch"] == 2
    assert pool["train"]["rnnt_decode_every"] == align["train"]["rnnt_decode_every"]
    for cfg in (et, pool, align):
        assert cfg["train"]["cudnn"] is False

    assert align["model"]["encoder_dim"] == et["model"]["encoder_dim"]
    assert align["model"]["causal"] is True
    assert align["train"]["warm_step"] == 5000
    assert align["train"]["ctc_loss_scale"] == 0.5
    assert align["train"]["prune_range"] == 15
    assert align["train"]["lm_scale"] == 0.0
    assert align["train"]["specaug_start_epoch"] == 9
    assert align["train"]["rnnt_decode_every"] == 5
    assert align["paths"]["exp_dir"] != et["paths"]["exp_dir"]
    assert "init_from" not in align["paths"]

    scratch = load_yaml(ROOT / "configs/zipformer_s_streaming_pool_scratch.yaml")
    assert scratch["model"]["encoder_dim"] == et["model"]["encoder_dim"]
    assert scratch["train"]["warm_step"] == et["train"]["warm_step"]
    assert scratch["train"]["ctc_loss_scale"] == et["train"]["ctc_loss_scale"]
    assert scratch["train"]["lm_scale"] == et["train"]["lm_scale"]
    assert scratch["train"]["prune_range"] == et["train"]["prune_range"]
    assert scratch["train"]["early_stopping_patience"] == scratch["train"]["epochs"]
    assert scratch["paths"]["metadata_dir"] == "metadata/pool_clean"
    assert "init_from" not in scratch["paths"]
    assert scratch["paths"]["exp_dir"] not in (
        et["paths"]["exp_dir"],
        pool["paths"]["exp_dir"],
        align["paths"]["exp_dir"],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="k2 pruned RNN-T runs on CUDA")
def test_zipformer_s_forward_and_param_count():
    k2 = pytest.importorskip("k2")
    pytest.importorskip("lhotse")
    from models.zipformer_icefall import build_zipformer, combine_loss, ragged_labels

    cfg = load_yaml(ROOT / "configs/zipformer_s_streaming_et.yaml")
    # One chunk size so the step does not depend on the random draw.
    cfg["model"]["chunk_size"] = [16]
    cfg["model"]["left_context_frames"] = [128]
    vocab = 320
    model = build_zipformer(cfg, vocab)
    n = sum(p.numel() for p in model.parameters())
    # Zipformer-S transducer is 23.3M at BPE 500. A 320-way character head
    # is in the same band; a missing stack would land far below it.
    assert 18_000_000 < n < 40_000_000, n

    device = torch.device("cuda")
    model.to(device).train()
    torch.manual_seed(0)
    b, t, f = 2, 160, 80
    feats = torch.randn(b, t, f, device=device)
    lens = torch.tensor([160, 120], device=device)
    ids = [torch.randint(1, vocab, (12,), dtype=torch.int32) for _ in range(b)]
    target = torch.nn.utils.rnn.pad_sequence(
        [row.to(torch.long) for row in ids], batch_first=True
    ).to(device)
    target_lens = torch.tensor([12, 12], device=device)
    assert labels_ok(120, 12)
    y = ragged_labels(target, target_lens)
    assert y.dim0 == 2
    simple, pruned, ctc, _, _ = model(
        x=feats,
        x_lens=lens,
        y=y,
        prune_range=5,
        am_scale=0.0,
        lm_scale=0.25,
    )
    loss = combine_loss(
        simple,
        pruned,
        ctc,
        batch_idx=1,
        tcfg=cfg["train"],
    )
    assert torch.isfinite(loss), float(loss)
    loss.backward()
    grad = next(p.grad for p in model.parameters() if p.grad is not None)
    assert torch.isfinite(grad).all()

    model.eval()
    with torch.no_grad():
        base, base_pruned, _, _, _ = model(
            x=feats, x_lens=lens, y=y, prune_range=5, am_scale=0.0, lm_scale=0.25, delay_penalty=0.0
        )
        again, _, _, _, _ = model(
            x=feats, x_lens=lens, y=y, prune_range=5, am_scale=0.0, lm_scale=0.25, delay_penalty=0.0
        )
        penalized, penalized_pruned, _, _, _ = model(
            x=feats, x_lens=lens, y=y, prune_range=5, am_scale=0.0, lm_scale=0.25, delay_penalty=0.05
        )
    assert torch.allclose(base, again)
    assert torch.isfinite(penalized) and torch.isfinite(penalized_pruned)
    assert not (torch.allclose(base, penalized) and torch.allclose(base_pruned, penalized_pruned))

    from decoding.zipformer_greedy import greedy_search_batch

    with torch.no_grad():
        enc, enc_lens = model.forward_encoder(feats, lens)
        hyps = greedy_search_batch(model, enc, enc_lens)
    assert len(hyps) == b
    assert all(isinstance(tok, int) for hyp in hyps for tok in hyp)
