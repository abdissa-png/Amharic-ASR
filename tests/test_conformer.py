import torch

from models.conformer import ConformerEncoder, count_parameters, zip_schedule
from models.ctc import build_model
from train import _amp_dtype, transfer_depthwise, transfer_mha_to_relpos
from utils import ROOT, load_yaml


def _cfg(d_model, layers, heads, ffn, kernel=31, **extra):
    model = {
        "d_model": d_model,
        "num_layers": layers,
        "num_heads": heads,
        "ffn_dim": ffn,
        "conv_kernel_size": kernel,
        "dropout": 0.1,
        "input_dim": 80,
    }
    model.update(extra)
    return {"model": model}


def test_bf16_without_tensor_cores_uses_fp16():
    enabled, dtype = _amp_dtype("bf16", cuda=True, bf16_tensor_cores=False)
    assert enabled and dtype == torch.float16
    enabled, dtype = _amp_dtype("bf16", cuda=True, bf16_tensor_cores=True)
    assert enabled and dtype == torch.bfloat16
    enabled, dtype = _amp_dtype("fp16", cuda=True, bf16_tensor_cores=True)
    assert enabled and dtype == torch.float16
    enabled, dtype = _amp_dtype("bf16", cuda=False, bf16_tensor_cores=False)
    assert not enabled and dtype == torch.float32


def test_20m_param_band():
    cfg = load_yaml(ROOT / "configs/20m_char.yaml")
    model = build_model(cfg, vocab_size=400)
    n = count_parameters(model)
    assert 12_000_000 < n < 28_000_000, n


def test_50m_param_band():
    cfg = load_yaml(ROOT / "configs/50m_char.yaml")
    model = build_model(cfg, vocab_size=400)
    n = count_parameters(model)
    assert 35_000_000 < n < 65_000_000, n


def test_50m_pool_config_builds():
    cfg = load_yaml(ROOT / "configs/50m_char_12layer_pool.yaml")
    model = build_model(cfg, vocab_size=320)
    n = count_parameters(model)
    assert 40_000_000 < n < 60_000_000, n
    assert cfg["model"]["pos_encoding"] == "rel"
    assert cfg["paths"]["metadata_dir"] == "metadata/pool"
    assert "init_from" not in cfg["paths"]


def test_default_checkpoint_keys():
    """Existing 20 M runs store MHA in_proj / conv.dw / two stride-2 convs."""
    enc = ConformerEncoder(d_model=64, num_layers=2, num_heads=2, ffn_dim=128, conv_kernel_size=7)
    keys = enc.state_dict()
    assert "subsample.conv.0.weight" in keys
    assert "subsample.conv.2.weight" in keys
    assert "layers.0.attn.attn.in_proj_weight" in keys
    assert "layers.0.conv.dw.weight" in keys
    assert "layers.0.conv.bn.weight" in keys


def test_forward_ctc_shapes():
    cfg = _cfg(64, 2, 2, 128, kernel=7)
    model = build_model(cfg, vocab_size=20)
    feats = torch.randn(2, 80, 80)
    lengths = torch.tensor([80, 60])
    logits, logit_lens = model(feats, lengths)
    assert logits.shape[0] == 2
    assert logits.shape[-1] == 20
    assert (logit_lens > 0).all()
    targets = torch.randint(1, 20, (2, 5))
    tlen = torch.tensor([5, 4])
    loss = model.ctc_loss(logits, logit_lens, targets, tlen)
    assert torch.isfinite(loss)


def _run_variant(extra, t=80):
    cfg = _cfg(64, 4, 2, 128, kernel=extra.pop("kernel", 7), **extra)
    model = build_model(cfg, vocab_size=20)
    feats = torch.randn(2, t, 80)
    lengths = torch.tensor([t, t - 16])
    logits, logit_lens = model(feats, lengths)
    assert logits.shape[-1] == 20
    assert logits.shape[1] == int(logit_lens[0])
    assert (logit_lens > 0).all()
    targets = torch.randint(1, 20, (2, 4))
    nll, _, _ = model.loss(feats, lengths, targets, torch.tensor([4, 3]), inter_weight=0.3, blank=0)
    assert torch.isfinite(nll)
    return model, logits, logit_lens


def test_relpos_forward():
    _run_variant({"pos_encoding": "rel"})


def test_multiconv_forward():
    model, _, _ = _run_variant({"kernel": [5, 7, 9]})
    assert isinstance(model.encoder.layers[0].conv.dw, torch.nn.ModuleList)


def test_lnconv_forward():
    _run_variant({"conv_norm": "layer_norm"})


def test_zipformer_restores_time():
    baseline = build_model(_cfg(64, 4, 2, 128, kernel=7), vocab_size=20)
    zip_model = build_model(_cfg(64, 4, 2, 128, kernel=7, zipformer=True), vocab_size=20)
    feats = torch.randn(2, 80, 80)
    lengths = torch.tensor([80, 64])
    _, t_base = baseline(feats, lengths)
    _, t_zip = zip_model(feats, lengths)
    assert torch.equal(t_base, t_zip)
    assert zip_schedule(12) == [(2, 1), (2, 2), (4, 4), (2, 2), (2, 1)]


def test_2x_subsample_keeps_20ms_grid():
    model, logits, logit_lens = _run_variant(
        {"subsample_factor": 2, "kernel": 15, "conv_norm": "layer_norm", "pos_encoding": "rel", "att_context_size": 32},
        t=80,
    )
    assert model.encoder.time_ratio == 2
    # One stride-2 conv: (80 - 1) // 2 + 1 = 40 encoder frames.
    assert int(logit_lens[0]) == 40
    assert logits.size(1) == 40


def test_20ms_k61_config_builds():
    cfg = load_yaml(ROOT / "configs/20m_char_12layer_20ms_k61.yaml")
    model = build_model(cfg, vocab_size=320)
    n = count_parameters(model)
    assert 12_000_000 < n < 28_000_000, n
    assert model.encoder.time_ratio == 2
    assert model.encoder.layers[0].conv.kernels == (61,)
    assert model.encoder.layers[0].conv.conv_norm == "layer_norm"
    assert model.encoder.layers[0].attn.att_context_size == 128
    assert cfg["train"]["epochs"] == 18
    assert cfg["paths"]["exp_dir"] == "exp/20m_char_12layer_20ms_k61"
    assert cfg["paths"]["init_from"] == "exp/20m_char_12layer_pool/best.pt"
    assert cfg["paths"]["exp_dir"] != cfg["paths"]["init_from"].rsplit("/", 1)[0]
    _assert_full_transfer("configs/20m_char_12layer_pool.yaml", cfg)


def test_20ms_config_builds():
    cfg = load_yaml(ROOT / "configs/20m_char_12layer_20ms.yaml")
    model = build_model(cfg, vocab_size=320)
    n = count_parameters(model)
    assert 12_000_000 < n < 28_000_000, n
    assert model.encoder.time_ratio == 2
    assert model.encoder.layers[0].conv.kernels == (15,)
    assert model.encoder.layers[0].conv.conv_norm == "layer_norm"
    assert model.encoder.layers[0].attn.att_context_size == 128
    assert cfg["train"]["epochs"] == 18
    assert cfg["train"]["max_duration"] == 40
    assert cfg["paths"]["metadata_dir"] == "metadata/pool_clean"
    assert cfg["train"]["short_crop"]["prob"] == 0.15
    assert cfg["paths"]["init_from"] == "exp/20m_char_12layer_pool/best.pt"
    assert cfg["paths"]["exp_dir"] != cfg["paths"]["init_from"].rsplit("/", 1)[0]
    _assert_full_transfer("configs/20m_char_12layer_pool.yaml", cfg)


def test_pool_short_config_is_a_fine_tune():
    cfg = load_yaml(ROOT / "configs/20m_char_12layer_pool_short.yaml")
    pool = load_yaml(ROOT / "configs/20m_char_12layer_pool.yaml")
    assert cfg["model"] == pool["model"]
    assert cfg["train"]["epochs"] == 6
    assert cfg["train"]["lr"] == 1.5e-4
    assert cfg["train"]["short_crop"]["prob"] == 0.15
    assert cfg["paths"]["metadata_dir"] == "metadata/pool_clean"
    assert cfg["paths"]["init_from"] == "exp/20m_char_12layer_pool/best.pt"
    assert cfg["paths"]["exp_dir"] != "exp/20m_char_12layer_pool"
    _assert_full_transfer("configs/20m_char_12layer_pool.yaml", cfg)


def test_short_clip_output_does_not_depend_on_batch_padding():
    for variant in (
        {"kernel": [9, 15, 31], "pos_encoding": "rel"},
        {"subsample_factor": 2, "kernel": 15, "conv_norm": "layer_norm", "pos_encoding": "rel", "att_context_size": 8},
    ):
        cfg = _cfg(64, 4, 4, 128, **variant)
        model = build_model(cfg, vocab_size=20).eval()
        torch.manual_seed(0)
        short, long = torch.randn(30, 80), torch.randn(300, 80)
        with torch.no_grad():
            alone, n = model(short[None], torch.tensor([30]))
            batch = torch.zeros(2, 300, 80)
            batch[0, :30], batch[1] = short, long
            mixed, _ = model(batch, torch.tensor([30, 300]))
        k = int(n[0])
        assert torch.allclose(alone[0, :k], mixed[0, :k], atol=1e-4), variant


def _assert_full_transfer(donor_config: str, cfg: dict) -> None:
    src = build_model(load_yaml(ROOT / donor_config), vocab_size=320).state_dict()
    cur = build_model(cfg, vocab_size=320).state_dict()
    loaded = {k for k, v in src.items() if k in cur and cur[k].shape == v.shape}
    dw_copied, _ = transfer_depthwise(src, cur)
    assert [k for k in cur if k not in loaded and k not in dw_copied] == []


def test_50m_20ms_config_builds():
    cfg = load_yaml(ROOT / "configs/50m_char_12layer_20ms.yaml")
    model = build_model(cfg, vocab_size=320)
    n = count_parameters(model)
    assert 40_000_000 < n < 60_000_000, n
    assert model.encoder.time_ratio == 2
    assert model.encoder.d_model == 384
    assert model.encoder.layers[0].conv.kernels == (15,)
    assert model.encoder.layers[0].conv.conv_norm == "layer_norm"
    assert cfg["train"]["epochs"] == 100
    assert cfg["train"]["batch_seconds"] == 80
    assert cfg["train"]["cudnn"] is False
    assert cfg["train"]["max_duration"] == 40
    assert cfg["paths"]["init_from"] == "exp/50m_char_12layer_pool/best.pt"
    _assert_full_transfer("configs/50m_char_12layer_pool.yaml", cfg)


def test_fast_8x_subsample():
    model, logits, logit_lens = _run_variant(
        {"subsample_factor": 8, "subsample": "dw_striding", "kernel": 9},
        t=160,
    )
    assert model.encoder.time_ratio == 8
    # 160 Mel frames → 20 encoder frames after three stride-2 convs.
    assert int(logit_lens[0]) == logits.size(1)
    assert logits.size(1) < 40


def test_local_attention_forward():
    _run_variant({"att_context_size": 8, "pos_encoding": "rel"})


def _blocked_vs_full(t: int, c: int, pad_from: int | None):
    from models.conformer import RelPositionalEncoding, RelSelfAttention

    torch.manual_seed(t * 31 + c)
    layer = RelSelfAttention(32, 4, dropout=0.0, att_context_size=c)
    with torch.no_grad():
        layer.u.normal_()
        layer.v.normal_()
    layer.eval()
    x = torch.randn(2, t, 32)
    pos = RelPositionalEncoding(32)(x)
    mask = torch.zeros(2, t, dtype=torch.bool)
    if pad_from is not None:
        mask[1, pad_from:] = True
    keep = ~mask
    outs, grads = [], []
    for blocked in (False, True):
        z = x.detach().clone().requires_grad_(True)
        zn = layer.norm(z)
        q, k, v = [layer._shape(p, 2, t) for p in layer.qkv(zn).chunk(3, dim=-1)]
        out = (
            layer._attend_blocked(q, k, v, pos, mask, c)
            if blocked
            else layer._attend_full(q, k, v, pos, mask)
        )
        out[keep].sum().backward()
        outs.append(out.detach()[keep])
        grads.append(z.grad)
    assert torch.allclose(outs[0], outs[1], atol=1e-5, rtol=1e-4), (t, c)
    assert torch.allclose(grads[0], grads[1], atol=1e-4, rtol=1e-4), (t, c)


def test_blocked_attention_matches_full_mask():
    _blocked_vs_full(37, 3, 28)
    _blocked_vs_full(150, 3, 100)
    _blocked_vs_full(200, 32, 131)
    _blocked_vs_full(193, 32, None)
    _blocked_vs_full(260, 70, 250)


def test_blocked_attention_used_past_block_length():
    from models.conformer import RelPositionalEncoding, RelSelfAttention

    layer = RelSelfAttention(32, 4, dropout=0.0, att_context_size=32)
    calls = []
    layer._attend_blocked = lambda *a: calls.append("blocked") or torch.zeros(1, a[0].size(2), 32)
    layer._attend_full = lambda *a: calls.append("full") or torch.zeros(1, a[0].size(2), 32)
    for t in (100, 129):
        x = torch.randn(1, t, 32)
        layer(x, None, RelPositionalEncoding(32)(x))
    assert calls == ["full", "blocked"]


def test_variant_configs_build():
    for name in (
        "ablation_4layer_relpos.yaml",
        "ablation_4layer_multiconv.yaml",
        "ablation_4layer_zip.yaml",
        "ablation_4layer_lnconv.yaml",
        "20m_bpe_fast.yaml",
        "20m_char_relpos.yaml",
        "20m_char_relpos_multiconv.yaml",
        "20m_char_12layer_pool.yaml",
        "20m_char_12layer_pool_short.yaml",
        "20m_char_12layer_20ms.yaml",
        "20m_char_12layer_20ms_k61.yaml",
    ):
        cfg = load_yaml(ROOT / "configs" / name)
        model = build_model(cfg, vocab_size=400)
        n = count_parameters(model)
        assert 6_000_000 < n < 28_000_000, (name, n)


def test_transfer_depthwise_center_crops_smaller_kernels():
    src = build_model(_cfg(64, 2, 2, 128, kernel=31), vocab_size=20).state_dict()
    dest = build_model(_cfg(64, 2, 2, 128, kernel=[9, 15, 31], pos_encoding="rel"), vocab_size=20)
    cur = dest.state_dict()
    copied, _ = transfer_depthwise(src, cur)
    dest.load_state_dict(cur)
    assert any(k.endswith("conv.dw.2.weight") for k in copied)
    donor = src["encoder.layers.0.conv.dw.weight"]
    got31 = dest.state_dict()["encoder.layers.0.conv.dw.2.weight"]
    got9 = dest.state_dict()["encoder.layers.0.conv.dw.0.weight"]
    assert torch.equal(got31, donor)
    assert torch.equal(got9, donor[..., 11:20])


def test_transfer_depthwise_stretches_40ms_branches_onto_61_taps():
    src = build_model(_cfg(16, 1, 2, 32, kernel=[9, 15, 31]), vocab_size=20).state_dict()
    dest = build_model(_cfg(16, 1, 2, 32, kernel=61), vocab_size=20)
    cur = dest.state_dict()
    stem = "encoder.layers.0.conv.dw."
    for i, value in enumerate((1.0, 3.0, 5.0)):
        src[f"{stem}{i}.weight"].fill_(value)
    copied, used = transfer_depthwise(src, cur)
    assert f"{stem}weight" in copied
    assert [f"{stem}{i}.weight" for i in range(3)] == [k for k in used if k.startswith(stem) and k.endswith(".weight")]
    got = cur[f"{stem}weight"]
    # Widest branch is 31 taps. Averaged onto 31, then stretched to 61.
    # Outer taps are only that branch: 5/3, with midpoints still 5/3.
    assert torch.allclose(got[..., 0], torch.full_like(got[..., 0], 5.0 / 3.0))
    assert torch.allclose(got[..., 1], torch.full_like(got[..., 1], 5.0 / 3.0))
    # Centre tap (index 30) is the mean of all three donor centres.
    assert torch.allclose(got[..., 30], torch.full_like(got[..., 30], (1.0 + 3.0 + 5.0) / 3.0))


def test_transfer_depthwise_folds_branches_into_one_kernel():
    from models.conformer import ConvModule

    multi = ConvModule(16, [9, 15], dropout=0.0).eval()
    single = ConvModule(16, 15, dropout=0.0).eval()
    src = {f"x.conv.{k}": v for k, v in multi.state_dict().items()}
    cur = {f"x.conv.{k}": v.clone() for k, v in single.state_dict().items()}
    for k, v in src.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v
    copied, used = transfer_depthwise(src, cur)
    assert "x.conv.dw.weight" in copied and "x.conv.dw.1.weight" in used
    single.load_state_dict({k[len("x.conv.") :]: v for k, v in cur.items()})
    x = torch.randn(2, 40, 16)
    assert torch.allclose(single(x), multi(x), atol=1e-5)


def test_transfer_mha_to_relpos_copies_qkv_and_out():
    src = build_model(_cfg(64, 2, 2, 128, kernel=7), vocab_size=20).state_dict()
    dest = build_model(_cfg(64, 2, 2, 128, kernel=7, pos_encoding="rel"), vocab_size=20)
    cur = dest.state_dict()
    pos_before = cur["encoder.layers.0.attn.pos_proj.weight"].clone()
    u_before = cur["encoder.layers.0.attn.u"].clone()
    copied, used = transfer_mha_to_relpos(src, cur)
    dest.load_state_dict(cur)
    assert "encoder.layers.0.attn.qkv.weight" in copied
    assert "encoder.layers.0.attn.attn.in_proj_weight" in used
    got = dest.state_dict()
    assert torch.equal(got["encoder.layers.0.attn.qkv.weight"], src["encoder.layers.0.attn.attn.in_proj_weight"])
    assert torch.equal(got["encoder.layers.0.attn.qkv.bias"], src["encoder.layers.0.attn.attn.in_proj_bias"])
    assert torch.equal(got["encoder.layers.0.attn.out.weight"], src["encoder.layers.0.attn.attn.out_proj.weight"])
    assert torch.equal(got["encoder.layers.0.attn.out.bias"], src["encoder.layers.0.attn.attn.out_proj.bias"])
    assert torch.equal(got["encoder.layers.0.attn.pos_proj.weight"], pos_before)
    assert torch.equal(got["encoder.layers.0.attn.u"], u_before)


def test_init_from_relpos_multiconv_remaps_all_mismatches():
    src = build_model(_cfg(64, 2, 2, 128, kernel=31), vocab_size=20).state_dict()
    dest = build_model(_cfg(64, 2, 2, 128, kernel=[9, 15, 31], pos_encoding="rel"), vocab_size=20)
    cur = dest.state_dict()
    skipped = [k for k, v in src.items() if k not in cur or cur[k].shape != v.shape]
    _, dw_used = transfer_depthwise(src, cur)
    _, attn_used = transfer_mha_to_relpos(src, cur)
    leftover = [k for k in skipped if k not in set(dw_used) | set(attn_used)]
    assert leftover == []
