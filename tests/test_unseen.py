from evaluation.unseen_words import seen_unseen_wer


def test_seen_unseen_split():
    v_train = {"ሰላም"}
    refs = ["ሰላም ፍቅር"]
    hyps = ["ሰላም ፍቅር"]
    out = seen_unseen_wer(refs, hyps, v_train)
    assert out["n_seen_ref_words"] == 1
    assert out["n_unseen_ref_words"] == 1
    assert out["seen_word_wer"] == 0.0
    assert out["unseen_word_wer"] == 0.0
