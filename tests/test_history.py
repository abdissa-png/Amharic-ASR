from train import _merge_history, _save_history


def test_merge_history_unions_by_epoch_and_later_wins():
    older = [{"epoch": 1, "dev_wer": 0.9}, {"epoch": 2, "dev_wer": 0.8}]
    newer = [{"epoch": 2, "dev_wer": 0.7}, {"epoch": 3, "dev_wer": 0.6}]
    merged = _merge_history(older, newer)
    assert [r["epoch"] for r in merged] == [1, 2, 3]
    assert merged[1]["dev_wer"] == 0.7


def test_save_history_union_keeps_disk_epochs(tmp_path):
    path = tmp_path / "history.json"
    _save_history(path, [{"epoch": 1, "dev_wer": 0.9}, {"epoch": 2, "dev_wer": 0.8}])
    saved = _save_history(path, [{"epoch": 34, "dev_wer": 0.4}], union_disk=True)
    assert [r["epoch"] for r in saved] == [1, 2, 34]


def test_save_history_without_union_replaces(tmp_path):
    path = tmp_path / "history.json"
    _save_history(path, [{"epoch": 1, "dev_wer": 0.9}])
    saved = _save_history(path, [{"epoch": 1, "dev_wer": 0.4}], union_disk=False)
    assert saved == [{"epoch": 1, "dev_wer": 0.4}]
