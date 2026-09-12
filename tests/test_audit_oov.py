from data.audit import leakage_checks
from data.oov import classify_unseen
from phonology.decomposition import likely_morphological_variant, split_compound


def test_leakage_detects_speaker_overlap():
    splits = {
        "train": [{"speaker_id": "a", "prompt_id": "p1", "clip_id": "c1", "text_raw": "x"}],
        "dev": [{"speaker_id": "a", "prompt_id": "p2", "clip_id": "c2", "text_raw": "y"}],
        "test": [{"speaker_id": "b", "prompt_id": "p3", "clip_id": "c3", "text_raw": "z"}],
    }
    report = leakage_checks(splits)
    assert report["clean"] is False
    assert report["speakers_clean"] is False
    assert "a" in report["speaker_overlap_train_dev"]["examples"]


def test_leakage_clean():
    splits = {
        "train": [{"speaker_id": "a", "prompt_id": "p1", "clip_id": "c1", "text_raw": "x"}],
        "dev": [{"speaker_id": "b", "prompt_id": "p2", "clip_id": "c2", "text_raw": "y"}],
        "test": [{"speaker_id": "c", "prompt_id": "p3", "clip_id": "c3", "text_raw": "z"}],
    }
    report = leakage_checks(splits)
    assert report["clean"] is True
    assert report["prompts_clean"] is True


def test_prompt_overlap_does_not_fail_speaker_gate():
    splits = {
        "train": [{"speaker_id": "a", "prompt_id": "p1", "clip_id": "c1", "text_raw": "same"}],
        "dev": [{"speaker_id": "b", "prompt_id": "p1", "clip_id": "c2", "text_raw": "other"}],
        "test": [{"speaker_id": "c", "prompt_id": "p9", "clip_id": "c3", "text_raw": "same"}],
    }
    report = leakage_checks(splits)
    assert report["speakers_clean"] is True
    assert report["clips_clean"] is True
    assert report["clean"] is True
    assert report["prompts_clean"] is False
    assert report["sentence_overlap_train_test"]["n"] == 1


def test_nan_speaker_is_not_overlap():
    splits = {
        "train": [{"speaker_id": "nan", "prompt_id": "p1", "clip_id": "c1", "text_raw": "x"}],
        "dev": [{"speaker_id": "nan", "prompt_id": "p2", "clip_id": "c2", "text_raw": "y"}],
        "test": [{"speaker_id": "c", "prompt_id": "p3", "clip_id": "c3", "text_raw": "z"}],
    }
    assert leakage_checks(splits)["clean"] is True


def test_morph_and_compound():
    vocab = {"ቤት", "ሰው", "ልጅ"}
    assert likely_morphological_variant("ቤቱ", vocab)
    assert split_compound("ቤትሰው", vocab) == ("ቤት", "ሰው")
    assert classify_unseen("hello", vocab, 3) == "loan_word"
