from decoding.greedy import collapse_ctc
from decoding.beam import prefix_beam_search
from evaluation.wer import score_texts
from phonology.g2p import g2p
from tokenizer.char import CharTokenizer
import torch
import torch.nn.functional as F


def test_collapse_ctc():
    assert collapse_ctc([0, 1, 1, 0, 2, 2, 2, 0], blank_id=0) == [1, 2]


def test_char_roundtrip():
    tok = CharTokenizer.from_texts(["ሰላም ዓለም", "hello"])
    ids = tok.encode("ሰላም")
    assert tok.decode(ids) == "ሰላም"


def test_score_perfect():
    s = score_texts(["ሰላም።"], ["ሰላም"])
    assert s["wer"] == 0.0
    assert s["cer"] == 0.0
    assert s["deletions"] == 0
    assert s["substitutions"] == 0


def test_score_counts_a_deletion():
    s = score_texts(["a b c"], ["a c"])
    assert s["hits"] == 2
    assert s["deletions"] == 1
    assert s["substitutions"] == 0
    assert s["insertions"] == 0


def test_g2p_selam():
    phones = g2p("ሰላም")
    assert phones[0].startswith("s")
    assert any(p.startswith("l") or p == "la" for p in phones)
    assert phones[-1].startswith("m")


def test_beam_matches_greedy_on_peaked_logits():
    tok = CharTokenizer.from_texts(["ab"])
    ids = tok.encode("ab")
    t, v = 6, tok.vocab_size
    logits = torch.full((t, v), -20.0)
    # a a blank b b blank
    logits[0, ids[0]] = 5.0
    logits[1, ids[0]] = 5.0
    logits[2, tok.blank_id] = 5.0
    logits[3, ids[1]] = 5.0
    logits[4, ids[1]] = 5.0
    logits[5, tok.blank_id] = 5.0
    log_probs = F.log_softmax(logits, dim=-1)
    hyp = prefix_beam_search(log_probs, tok, beam_size=4, blank_id=tok.blank_id, token_top_k=4)
    assert hyp == "ab"


def test_kenlm_labels_blank_and_unique():
    from decoding.kenlm import labels_from_tokenizer

    tok = CharTokenizer.from_texts(["ሰላም ዓለም"])
    labels = labels_from_tokenizer(tok)
    assert labels[tok.blank_id] == ""
    assert labels[tok.unk_id] == "\ufffd"
    nonempty = [x for x in labels if x != ""]
    assert len(nonempty) == len(set(nonempty))


def test_arpa_fallback_writes_header(tmp_path):
    from decoding.train_lm import write_arpa_stupid_backoff

    path = tmp_path / "lm.arpa"
    write_arpa_stupid_backoff([["ሰላም", "ዓለም"], ["ሰላም"]], path, order=2)
    text = path.read_text(encoding="utf-8")
    assert "\\1-grams:" in text
    assert "ሰላም" in text


def test_word_sentence_split_and_clean():
    from decoding.train_lm import clean_sentence, split_sentences, usable_lm_line, wrap_long

    blob = "ኢትዮጵያ በደቡብ አፍሪካ ሲጠጣ፡፡ሃኒም ኢትዮጵያም ይሸነፉ ይሆን? የዋሆች ሆይ!አትስጉ"
    parts = split_sentences(blob)
    assert len(parts) >= 3
    cleaned = [clean_sentence(p) for p in parts]
    assert all("።" not in c and "፡" not in c for c in cleaned)
    assert usable_lm_line(cleaned[0])
    assert not usable_lm_line("hello world test")
    long = " ".join(["ቤት"] * 90)
    wrapped = wrap_long(long, max_words=80)
    assert len(wrapped) == 2


def test_resolve_lm_prefers_word(tmp_path):
    from decoding.kenlm import resolve_lm_dir

    (tmp_path / "lm").mkdir()
    (tmp_path / "lm" / "lm.bin").write_bytes(b"small")
    (tmp_path / "lm" / "unigrams.txt").write_text("a\n", encoding="utf-8")
    word = tmp_path / "lm_word"
    word.mkdir()
    (word / "lm.bin").write_bytes(b"word")
    (word / "unigrams.txt").write_text("b\n", encoding="utf-8")
    model, uni = resolve_lm_dir(tmp_path)
    assert model == word / "lm.bin"
    assert uni == word / "unigrams.txt"
