"""English front-end.

Pipeline:  raw text -> normalisation (numbers, times, dates, abbreviations)
           -> g2p_en (CMUdict lookup + POS-based homograph disambiguation
              + neural G2P for OOV words) -> ARPAbet phonemes with stress.

Changes w.r.t. the reference repo (ming024/FastSpeech2):
  * The reference relied on MFA alignments and a fixed LibriSpeech lexicon.
    Here phonemes are produced directly from text with g2p_en, so OOV words
    (e.g. "antidisestablishmentarianism", code-mixed "andaroo") still get a
    pronunciation, and heteronyms ("conduct" noun vs verb) are resolved with
    the POS tagger inside g2p_en.
  * Added normalisation for clock times (10:45 AM) and dates (12 November 2026),
    which the reference cleaners did not handle.
  * Explicit word-boundary token "_" and punctuation tokens so the model can
    learn pauses.
"""
import os
import re

import inflect

_inflect = inflect.engine()

ARPABET = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P",
    "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]
_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
PUNCT = [",", ".", "?", "!"]
WB = "_"  # word boundary

SYMBOLS = []
for p in ARPABET:
    if p in _VOWELS:
        SYMBOLS += [p + "0", p + "1", p + "2"]
    else:
        SYMBOLS.append(p)
SYMBOLS += PUNCT + [WB]

_g2p = None
_lexicon_override = {}


def configure(lexicon_override=None, **_):
    """Optional user lexicon: lines 'WORD  PH1 PH2 ...' (overrides g2p)."""
    global _lexicon_override
    _lexicon_override = {}
    if lexicon_override and os.path.exists(lexicon_override):
        with open(lexicon_override, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                _lexicon_override[parts[0].lower()] = parts[1:]


def _get_g2p():
    global _g2p
    if _g2p is None:
        import nltk
        for res, path in [("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
                          ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
                          ("cmudict", "corpora/cmudict")]:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(res, quiet=True)
                except Exception:
                    pass
        from g2p_en import G2p
        _g2p = G2p()
    return _g2p


# ----------------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------------
_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]
_month_re = "|".join(_MONTHS)

_abbrev = [(re.compile(r"\b%s\." % a, re.IGNORECASE), b) for a, b in [
    ("mrs", "misess"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"),
    ("co", "company"), ("jr", "junior"), ("maj", "major"), ("gen", "general"),
    ("drs", "doctors"), ("rev", "reverend"), ("lt", "lieutenant"), ("hon", "honorable"),
    ("sgt", "sergeant"), ("capt", "captain"), ("esq", "esquire"), ("ltd", "limited"),
    ("col", "colonel"), ("ft", "fort"), ("etc", "et cetera"), ("vs", "versus"),
]]


def _num(n):
    return _inflect.number_to_words(n, andword="").replace("-", " ").replace(",", "")


def _year(y):
    y = int(y)
    if 2000 <= y <= 2009:
        return _num(y)
    if 1100 <= y <= 2099 and y % 100 != 0:
        return _num(y // 100) + " " + (("oh " + _num(y % 100)) if y % 100 < 10 else _num(y % 100))
    if 1100 <= y <= 2099:
        return _num(y // 100) + " hundred"
    return _num(y)


def _ordinal(n):
    return _inflect.number_to_words(_inflect.ordinal(int(n)), andword="").replace("-", " ").replace(",", "")


def _time(m):
    h, mnt = int(m.group(1)), int(m.group(2))
    ampm = m.group(3)
    words = _num(h)
    if mnt == 0:
        words += "" if ampm else " o'clock"
    elif mnt < 10:
        words += " oh " + _num(mnt)
    else:
        words += " " + _num(mnt)
    if ampm:
        words += " ay em" if ampm.lower().startswith("a") else " pee em"
    return " " + words + " "


def _date_dmy(m):
    d, mon, y = m.group(1), m.group(2), m.group(3)
    s = "the " + _ordinal(d) + " of " + mon
    if y:
        s += " " + _year(y)
    return s


def _date_mdy(m):
    mon, d, y = m.group(1), m.group(2), m.group(3)
    s = mon + " " + _ordinal(d)
    if y:
        s += " " + _year(y)
    return s


def normalize(text):
    t = text
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.replace("—", ", ").replace("–", " ").replace(";", ",").replace(":", " : ")
    t = re.sub(r"(\d)\s+:\s+(\d)", r"\1:\2", t)          # keep 10:45 together
    for r, s in _abbrev:
        t = r.sub(s, t)
    # times: 10:45 AM / 7:05 p.m. / 18:30
    t = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp]\.?[Mm]\.?)?(?![\d])", _time, t)
    # dates: 12 November 2026 / November 12, 2026
    t = re.sub(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(%s)(?:,?\s+(\d{4}))?" % _month_re, _date_dmy, t, flags=re.IGNORECASE)
    t = re.sub(r"\b(%s)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?" % _month_re, _date_mdy, t, flags=re.IGNORECASE)
    # currency / percent
    t = re.sub(r"\$(\d+(?:\.\d+)?)", lambda m: m.group(1) + " dollars", t)
    t = re.sub(r"₹\s?(\d+(?:\.\d+)?)", lambda m: m.group(1) + " rupees", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s?%", lambda m: m.group(1) + " percent", t)
    # ordinals 1st 2nd
    t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: _ordinal(m.group(1)), t)
    # standalone 4-digit years
    t = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b", lambda m: _year(m.group(1)), t)
    # long unformatted digit strings (phone / account / PIN numbers) are spoken digit by
    # digit in Svarah ("82562848140" -> "eight two five ..."), not as "eighty two billion ..."
    _digit = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    t = re.sub(r"(?<![\d,.])\d{5,}(?![\d,.])", lambda m: " " + " ".join(_digit[int(c)] for c in m.group(0)) + " ", t)
    # decimals and remaining numbers
    t = re.sub(r"(\d+)\.(\d+)", lambda m: _num(int(m.group(1))) + " point " + " ".join(_num(int(c)) for c in m.group(2)), t)
    t = re.sub(r"\d+(,\d{3})*", lambda m: _num(int(m.group(0).replace(",", ""))), t)
    t = t.replace("&", " and ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def text_to_phonemes(text):
    text = normalize(text)
    g2p = _get_g2p()
    phones = []
    # split into words / punctuation, keep original spacing semantics
    tokens = re.findall(r"[A-Za-z']+|[,.?!]", text)
    for tok in tokens:
        if tok in PUNCT:
            if phones and phones[-1] == WB:
                phones.pop()
            phones.append(tok)
            phones.append(WB)
            continue
        w = tok.lower().strip("'")
        if not w:
            continue
        if w in _lexicon_override:
            ph = _lexicon_override[w]
        else:
            ph = None
        if ph is None:
            ph = None  # resolved below with sentence context
        phones.append(("__WORD__", w, ph))
        phones.append(WB)
    # run g2p on the whole sentence so POS-based homograph disambiguation works
    words = [p[1] for p in phones if isinstance(p, tuple)]
    sent_ph = g2p(" ".join(words)) if words else []
    # split g2p output by spaces into per-word chunks
    chunks, cur = [], []
    for p in sent_ph:
        if p == " ":
            if cur:
                chunks.append(cur)
            cur = []
        elif p in PUNCT or not re.match(r"^[A-Z]+[0-2]?$", p):
            continue
        else:
            cur.append(p)
    if cur:
        chunks.append(cur)
    if len(chunks) != len(words):  # fall back to per-word g2p
        chunks = [[q for q in g2p(w) if re.match(r"^[A-Z]+[0-2]?$", q)] for w in words]
    out, k = [], 0
    for p in phones:
        if isinstance(p, tuple):
            ph = p[2] if p[2] is not None else chunks[k]
            k += 1
            out.extend(ph)
        else:
            out.append(p)
    # strip leading/trailing word boundaries, collapse doubles
    clean = []
    for p in out:
        if p == WB and (not clean or clean[-1] == WB):
            continue
        clean.append(p)
    while clean and clean[-1] == WB:
        clean.pop()
    if not clean or clean[-1] not in PUNCT:
        clean.append(".")
    return clean
