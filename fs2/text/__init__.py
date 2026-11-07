"""Language front-ends: text normalisation + grapheme-to-phoneme.

Each language module exposes
    SYMBOLS              : list[str]  (index 0 is reserved for padding)
    text_to_phonemes(s)  : list[str]
"""
from . import english, telugu

_FRONTENDS = {"en": english, "te": telugu}


def get_frontend(lang):
    if lang not in _FRONTENDS:
        raise ValueError(f"Unsupported language '{lang}', expected one of {list(_FRONTENDS)}")
    return _FRONTENDS[lang]


class TextProcessor:
    def __init__(self, lang, **kwargs):
        self.lang = lang
        self.fe = get_frontend(lang)
        if hasattr(self.fe, "configure"):
            self.fe.configure(**kwargs)
        self.symbols = ["<pad>"] + list(self.fe.SYMBOLS)
        self.sym2id = {s: i for i, s in enumerate(self.symbols)}

    def phonemize(self, text):
        return self.fe.text_to_phonemes(text)

    def encode(self, phones):
        unknown = [p for p in phones if p not in self.sym2id]
        if unknown:
            raise KeyError(f"Unknown phoneme(s) {sorted(set(unknown))}")
        return [self.sym2id[p] for p in phones]

    def __call__(self, text):
        phones = self.phonemize(text)
        return phones, self.encode(phones)
