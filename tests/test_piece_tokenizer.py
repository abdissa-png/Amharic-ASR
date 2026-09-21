from tokenizer.char import BLANK, PAD, UNK
from tokenizer.piece import MARK, PieceTokenizer
from decoding.kenlm import labels_from_tokenizer


def test_piece_roundtrip_and_longest_match():
    tokens = [BLANK, UNK, PAD, MARK, "ሰ", "ላ", "ም", f"{MARK}ሰላም", "ነ", "ው", f"{MARK}ነው", " "]
    tok = PieceTokenizer(tokens)
    assert tok.decode(tok.encode("ሰላም ነው")) == "ሰላም ነው"
    ids = tok.encode("ሰላም")
    assert ids == [tok.stoi[f"{MARK}ሰላም"]]
    ids = tok.encode("ነው")
    assert ids == [tok.stoi[f"{MARK}ነው"]]


def test_piece_falls_back_to_chars():
    tokens = [BLANK, UNK, PAD, MARK, "አ", "ዲ", "ስ"]
    tok = PieceTokenizer(tokens)
    assert tok.decode(tok.encode("አዲስ")) == "አዲስ"
    assert len(tok.encode("አዲስ")) == 4  # ▁ + 3 chars


def test_piece_kenlm_labels_are_bpe():
    tokens = [BLANK, UNK, PAD, MARK, "ሰ", f"{MARK}ሰ"]
    tok = PieceTokenizer(tokens)
    labels = labels_from_tokenizer(tok)
    assert labels[tok.blank_id] == ""
    assert any(s.startswith(MARK) for s in labels)
    nonempty = [x for x in labels if x != ""]
    assert len(nonempty) == len(set(nonempty))
