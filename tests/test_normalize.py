from evaluation.geez_eval import normalize
from data.normalize import for_eval, for_training


def test_nfc_and_whitespace():
    assert for_training("  ሰላም   ዓለም  ") == "ሰላም ዓለም"


def test_training_keeps_homophones():
    # ሐ must not be folded in the training target
    text = "ሐኪም"
    assert "ሐ" in for_training(text)
    assert for_eval(text) != for_training(text) or "ሀ" in for_eval(text)


def test_eval_folds_homophones():
    assert normalize("ሐኪም") == normalize("ሀኪም")
    assert normalize("ሠላም") == normalize("ሰላም")
    assert normalize("ዐለም") == normalize("አለም")
    assert normalize("ፀሐይ") == normalize("ጸሀይ")


def test_eval_strips_ethiopic_punct():
    n = normalize("ሰላም። እንደምን ነህ፣")
    assert "።" not in n
    assert "፣" not in n
    assert "ሰላም" in n


def test_lang_tag_stripped():
    from evaluation.geez_eval import strip_lang_tag

    assert strip_lang_tag("[AMH] ሰላም").strip() == "ሰላም"


def test_quotes_and_latin_punct_training():
    t = for_training('“hello” 123!')
    assert '"' not in t
    assert "hello" in t
    assert "123" in t
