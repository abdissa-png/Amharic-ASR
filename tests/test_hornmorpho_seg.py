from tokenizer.hornmorpho_seg import surface_align


def test_surface_align_noun_allomorph():
    hit = surface_align("ለቤቶቻችን", "ቤት", "ቤት")
    assert hit is not None
    assert hit["prefix"] + hit["stem"] + hit["suffix"] == "ለቤቶቻችን"
    assert hit["prefix"] == "ለ"
    assert hit["stem"] == "ቤቶ"
    assert hit["suffix"] == "ቻችን"


def test_surface_align_name():
    hit = surface_align("የኢትዮጵያ", "ኢትዮጵያ", "ኢትዮጵያ")
    assert hit == {"prefix": "የ", "stem": "ኢትዮጵያ", "suffix": "", "matched": "ኢትዮጵያ"}


def test_surface_align_templatic_verb_fails():
    # Canonical stem does not appear in the surface verb; we must not invent a split.
    assert surface_align("አስታውቀዋል", "ኣስተ-ኣውቅ", "አስታወቀ") is None
