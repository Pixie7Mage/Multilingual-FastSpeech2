"""Telugu front-end (new — the reference repo has no Telugu support).

Telugu script is an abugida and very close to phonemic, so a rule-based
grapheme-to-phoneme converter is used instead of MFA/lexicons:

  * every consonant carries an inherent /a/ unless followed by a vowel sign
    (matra) or the virama (్), which suppresses it (conjuncts / geminates);
  * independent vowels and matras map to short/long vowel phonemes;
  * anusvara (ం) becomes a homorganic nasal decided by the following consonant
    (ng / ny / nn / n / m) and /m/ word-finally or before sonorants/sibilants;
  * visarga (ః) -> /h/;  candrabindu (ఁ), ZWJ/ZWNJ and nukta are dropped;
  * vocalic r (ఋ / ృ) -> /r u/.

Numbers are expanded to Telugu words (cardinals, ordinals with the "వ"
suffix, years, clock times), because IndicTTS transcripts are fully written
out in Telugu script while the test sentence T4 contains digits.
"""
import re

# ----------------------------------------------------------------------------
# Phoneme inventory
# ----------------------------------------------------------------------------
VOWELS = ["a", "aa", "i", "ii", "u", "uu", "e", "ee", "ai", "o", "oo", "au"]
CONSONANTS = [
    "k", "kh", "g", "gh", "ng",
    "c", "ch", "j", "jh", "ny",
    "tt", "tth", "dd", "ddh", "nn",
    "t", "th", "d", "dh", "n",
    "p", "ph", "b", "bh", "m",
    "y", "r", "rr", "l", "ll", "zh", "v", "sh", "ss", "s", "h",
    "ts", "dz", "f", "z",
]
PUNCT = [",", ".", "?", "!"]
WB = "_"
SYMBOLS = VOWELS + CONSONANTS + PUNCT + [WB]

_IND_VOWEL = {
    "అ": ["a"], "ఆ": ["aa"], "ఇ": ["i"], "ఈ": ["ii"], "ఉ": ["u"], "ఊ": ["uu"],
    "ఋ": ["r", "u"], "ౠ": ["r", "uu"], "ఌ": ["l", "u"], "ౡ": ["l", "uu"],
    "ఎ": ["e"], "ఏ": ["ee"], "ఐ": ["ai"], "ఒ": ["o"], "ఓ": ["oo"], "ఔ": ["au"],
}
_MATRA = {
    "ా": ["aa"], "ి": ["i"], "ీ": ["ii"], "ు": ["u"], "ూ": ["uu"],
    "ృ": ["r", "u"], "ౄ": ["r", "uu"], "ె": ["e"], "ే": ["ee"], "ై": ["ai"],
    "ొ": ["o"], "ో": ["oo"], "ౌ": ["au"], "ౖ": ["ai"],
}
_CONS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "c", "ఛ": "ch", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "tt", "ఠ": "tth", "డ": "dd", "ఢ": "ddh", "ణ": "nn",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "rr", "ల": "l", "ళ": "ll", "ఴ": "zh",
    "వ": "v", "శ": "sh", "ష": "ss", "స": "s", "హ": "h",
    "ౘ": "ts", "ౙ": "dz",
}
VIRAMA = "్"
ANUSVARA = "ం"
VISARGA = "ః"
_IGNORE = {"ఁ", "‌", "‍", "఼", "ఀ", "ౕ", "ౖ"}

_NASAL_FOR = {}
for c in ["k", "kh", "g", "gh", "ng"]:
    _NASAL_FOR[c] = "ng"
for c in ["c", "ch", "j", "jh", "ny", "ts", "dz"]:
    _NASAL_FOR[c] = "ny"
for c in ["tt", "tth", "dd", "ddh", "nn"]:
    _NASAL_FOR[c] = "nn"
for c in ["t", "th", "d", "dh", "n"]:
    _NASAL_FOR[c] = "n"
for c in ["p", "ph", "b", "bh", "m", "f"]:
    _NASAL_FOR[c] = "m"

# ----------------------------------------------------------------------------
# Number normalisation
# ----------------------------------------------------------------------------
_UNITS = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది",
          "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు",
          "పద్దెనిమిది", "పందొమ్మిది"]
_TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై"}
_HUNDREDS_PREFIX = {1: "నూట", 2: "రెండు వందల", 3: "మూడు వందల", 4: "నాలుగు వందల", 5: "ఐదు వందల",
                    6: "ఆరు వందల", 7: "ఏడు వందల", 8: "ఎనిమిది వందల", 9: "తొమ్మిది వందల"}
_HUNDREDS_EXACT = {1: "వంద", 2: "రెండు వందలు", 3: "మూడు వందలు", 4: "నాలుగు వందలు", 5: "ఐదు వందలు",
                   6: "ఆరు వందలు", 7: "ఏడు వందలు", 8: "ఎనిమిది వందలు", 9: "తొమ్మిది వందలు"}


def _below_100(n):
    if n < 20:
        return _UNITS[n]
    t, u = divmod(n, 10)
    return _TENS[t] if u == 0 else _TENS[t] + " " + _UNITS[u]


def _below_1000(n):
    if n < 100:
        return _below_100(n)
    h, r = divmod(n, 100)
    return _HUNDREDS_EXACT[h] if r == 0 else _HUNDREDS_PREFIX[h] + " " + _below_100(r)


def num_to_telugu(n):
    n = int(n)
    if n < 1000:
        return _below_1000(n)
    if n < 100000:
        th, r = divmod(n, 1000)
        if th == 1:
            head = "వెయ్యి" if r == 0 else "వెయ్యి"
        else:
            head = _below_100(th) + (" వేలు" if r == 0 else " వేల")
        return head if r == 0 else head + " " + _below_1000(r)
    if n < 10000000:
        lk, r = divmod(n, 100000)
        head = ("ఒక లక్ష" if lk == 1 else _below_100(lk) + " లక్షల") if r else ("ఒక లక్ష" if lk == 1 else _below_100(lk) + " లక్షలు")
        return head if r == 0 else head + " " + num_to_telugu(r)
    cr, r = divmod(n, 10000000)
    head = ("ఒక కోటి" if cr == 1 else num_to_telugu(cr) + " కోట్ల") if r else ("ఒక కోటి" if cr == 1 else num_to_telugu(cr) + " కోట్లు")
    return head if r == 0 else head + " " + num_to_telugu(r)


def _ordinal_telugu(n):
    w = num_to_telugu(n)
    last = w.split(" ")[-1]
    stem = " ".join(w.split(" ")[:-1])
    if last.endswith("ై"):
        o = last[:-1] + "య్యవ"          # ఇరవై -> ఇరవయ్యవ
    elif last.endswith(("ు", "ి")):
        o = last[:-1] + "వ"              # పన్నెండు -> పన్నెండవ, ఒకటి -> ఒకటవ
    else:
        o = last + "వ"                   # వంద -> వందవ
    return (stem + " " + o).strip()


def normalize(text):
    t = text.replace("​", "")
    t = re.sub(r"[“”\"'‘’]", "", t)
    t = t.replace("—", ", ").replace("–", " ").replace(";", ",")
    t = t.replace("।", ".").replace("॥", ".")
    # clock time 10:45 -> పది నలభై ఐదు ; 10:00 -> పది
    def _time(m):
        h, mi = int(m.group(1)), int(m.group(2))
        return " " + num_to_telugu(h) + ("" if mi == 0 else " " + num_to_telugu(mi)) + " "
    t = re.sub(r"(\d{1,2}):(\d{2})", _time, t)
    # ordinals written as digits + వ (12వ -> పన్నెండవ)
    t = re.sub(r"(\d+)వ(?![ఀ-౿])", lambda m: " " + _ordinal_telugu(m.group(1)) + " ", t)
    # decimals
    t = re.sub(r"(\d+)\.(\d+)", lambda m: num_to_telugu(m.group(1)) + " పాయింట్ " + " ".join(_UNITS[int(c)] for c in m.group(2)), t)
    t = re.sub(r"\d+(,\d+)*", lambda m: " " + num_to_telugu(m.group(0).replace(",", "")) + " ", t)
    t = t.replace("%", " శాతం ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _word_to_phones(word):
    out = []
    pending = False  # last consonant still owns its inherent 'a'
    for ch in word:
        if ch in _CONS:
            if pending:
                out.append("a")
            out.append(_CONS[ch])
            pending = True
        elif ch in _MATRA:
            out.extend(_MATRA[ch])
            pending = False
        elif ch == VIRAMA:
            pending = False
        elif ch in _IND_VOWEL:
            if pending:
                out.append("a")
            out.extend(_IND_VOWEL[ch])
            pending = False
        elif ch == ANUSVARA:
            if pending:
                out.append("a")
            out.append("<M>")
            pending = False
        elif ch == VISARGA:
            if pending:
                out.append("a")
            out.append("h")
            pending = False
        else:
            # ignore unknown / ZWJ / candrabindu
            continue
    if pending:
        out.append("a")
    # resolve anusvara
    res = []
    for i, p in enumerate(out):
        if p == "<M>":
            nxt = out[i + 1] if i + 1 < len(out) else None
            res.append(_NASAL_FOR.get(nxt, "m"))
        else:
            res.append(p)
    return res


def has_latin(text):
    return re.search(r"[A-Za-z]", text) is not None


def text_to_phonemes(text):
    t = normalize(text)
    phones = []
    for tok in re.findall(r"[ఀ-౿‌‍]+|[,.?!]", t):
        if tok in PUNCT:
            if phones and phones[-1] == WB:
                phones.pop()
            phones.append(tok)
            phones.append(WB)
            continue
        ph = _word_to_phones(tok)
        if ph:
            phones.extend(ph)
            phones.append(WB)
    while phones and phones[-1] == WB:
        phones.pop()
    if not phones or phones[-1] not in PUNCT:
        phones.append(".")
    return phones
