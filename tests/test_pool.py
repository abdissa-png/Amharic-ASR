import json
from pathlib import Path

from data.prepare_leyu import assign_prompts
from data.prepare_pool import prepare
from data.prepare_pool_leyu import prepare as prepare_pool_leyu


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_prepare_pool_stamps_corpus_and_skips_missing(tmp_path: Path):
    root = tmp_path
    et_train = [
        {"clip_id": "et_1", "audio": "data/train/a.flac", "text": "ሰላም", "duration_s": 1.0, "region": "addis"},
        {"clip_id": "et_2", "audio": "data/train/b.flac", "text": "እንደምን", "duration_s": 1.0, "region": ""},
    ]
    et_dev = [{"clip_id": "et_d", "audio": "data/dev/a.flac", "text": "ሰላም", "duration_s": 1.0, "region": "addis"}]
    et_test = [{"clip_id": "et_t", "audio": "data/test/a.flac", "text": "ሰላም", "duration_s": 1.0, "region": "addis"}]
    _write(root / "metadata" / "train.jsonl", et_train)
    _write(root / "metadata" / "dev.jsonl", et_dev)
    _write(root / "metadata" / "test.jsonl", et_test)

    fl_train = [{"clip_id": "fl_1", "audio": "data/fleurs_am_et/train/x.flac", "text": "hello", "duration_s": 2.0}]
    fl_dev = [{"clip_id": "fl_d", "audio": "data/fleurs_am_et/dev/x.flac", "text": "hello", "duration_s": 1.0}]
    fl_test = [{"clip_id": "fl_t", "audio": "data/fleurs_am_et/test/x.flac", "text": "hello", "duration_s": 1.0}]
    _write(root / "metadata" / "fleurs_am_et" / "train.jsonl", fl_train)
    _write(root / "metadata" / "fleurs_am_et" / "dev.jsonl", fl_dev)
    _write(root / "metadata" / "fleurs_am_et" / "test.jsonl", fl_test)

    stats = prepare(out_dir=root, sources=("dataset_et", "fleurs", "alffa"), dev_policy="et")
    assert "alffa" in stats["skipped"]
    assert stats["sources"] == ["dataset_et", "fleurs"]
    train = [json.loads(l) for l in (root / "metadata" / "pool" / "train.jsonl").read_text().splitlines() if l]
    assert {r["corpus"] for r in train} == {"dataset_et", "fleurs"}
    assert {r["clip_id"] for r in train} == {"et_1", "et_2", "fl_1"}
    # Default region stamps empty fleurs rows as "".
    assert any(r["clip_id"] == "et_1" and r["region"] == "addis" for r in train)

    dev = [json.loads(l) for l in (root / "metadata" / "pool" / "dev.jsonl").read_text().splitlines() if l]
    assert [r["clip_id"] for r in dev] == ["et_d"]
    pooled = [json.loads(l) for l in (root / "metadata" / "pool" / "dev_pooled.jsonl").read_text().splitlines() if l]
    assert {r["clip_id"] for r in pooled} == {"et_d", "fl_d"}


def test_leyu_prompts_do_not_cross_splits():
    prompts = [f"prompt_{i}" for i in range(100)]
    assignment = assign_prompts(prompts, dev_frac=0.05, test_frac=0.05, seed=0)
    assert set(assignment) == set(prompts)
    assert set(assignment.values()) == {"train", "dev", "test"}
    assert sum(v == "dev" for v in assignment.values()) == 5
    assert sum(v == "test" for v in assignment.values()) == 5


def test_pool_leyu_keeps_et_dev_and_adds_leyu_train(tmp_path: Path):
    pool = tmp_path / "metadata" / "pool_clean"
    leyu = tmp_path / "metadata" / "leyu_am"
    _write(pool / "train.jsonl", [{"clip_id": "et_1", "corpus": "dataset_et", "duration_s": 3600.0, "text": "a"}])
    _write(pool / "dev.jsonl", [{"clip_id": "et_d", "corpus": "dataset_et", "duration_s": 1.0, "text": "d"}])
    _write(pool / "test.jsonl", [{"clip_id": "et_t", "corpus": "dataset_et", "duration_s": 1.0, "text": "t"}])
    _write(leyu / "train.jsonl", [{"clip_id": "leyu_1", "corpus": "leyu", "duration_s": 7200.0, "text": "b"}])
    stats = prepare_pool_leyu(out_dir=tmp_path, pool_dir=pool, leyu_dir=leyu)
    train = [json.loads(l) for l in (tmp_path / "metadata" / "pool_leyu" / "train.jsonl").read_text().splitlines()]
    dev = [json.loads(l) for l in (tmp_path / "metadata" / "pool_leyu" / "dev.jsonl").read_text().splitlines()]
    assert [r["clip_id"] for r in train] == ["et_1", "leyu_1"]
    assert [r["clip_id"] for r in dev] == ["et_d"]
    assert stats["train_hours"] == 3.0
    assert stats["leyu_clips"] == 1


def test_pack_limits_padded_seconds_not_the_sum():
    from data.dataset import pack_by_padded_seconds

    durations = [70.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    batches = pack_by_padded_seconds(durations, list(range(len(durations))), 120.0)
    assert batches[0] == [0]
    for batch in batches:
        peak = max(durations[i] for i in batch)
        assert peak * len(batch) <= 120.0
    flat = [i for batch in batches for i in batch]
    assert flat == list(range(len(durations)))
