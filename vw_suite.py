#!/usr/bin/env python3
"""
vw_suite.py — Voynich Structural Analysis Suite (inline parser)

Deterministic, leakage-free character-level structural tests for the Voynich
Manuscript (LSI transcription) and control corpora.

Key properties:
- Inline IVTFF tag-aware parser (no iParser dependency)
- Embedded exclusion list for physically missing folios (from iParser_guarded)
- Numeric folio sorting (supports suffix digits like f101r2)
- Train/test split is folio-based by side(s) (leakage-free)
- Bigram/trigram character models with additive smoothing
- ΔBPC = BPC2 − BPC3 (computed from the same rounded values printed)
- Positional bits (init/med/fin)
- Levenshtein ≤1 overlap (token and type)

Splat handling (current default methodology):
- Default: DROP any whitespace-delimited raw token containing '!' (splat marker)
- Optional: --split_splats treats '!' as a boundary (baseline behavior)

Defaults:
- preset=recipe (folios 103–116)
- train_sides=r, test_sides=v
- transcriber=H
- min_len=2
- K=0.1

#!/usr/bin/env python3

vw_suite.py — Voynich Structural Analysis Suite

============================================================
COPY / PASTE TEST BATTERY (REPRODUCIBLE COMMANDS)
============================================================

ASSUMPTIONS
-----------
- LSI transcription file is present as: LSI.txt
- Working directory contains this script
- Default smoothing: K = 0.1
- Default token length floor: min_len = 2
- Default splat handling: DROP tokens containing '!'

------------------------------------------------------------
CORE SANITY CHECKS (RECTO ↔ VERSO)
------------------------------------------------------------

# Recipe section (folios 103–116), train recto → test verso (Takahashi)
python vw_suite.py --preset recipe --train_sides r --test_sides v --transcriber H

# Reverse sanity check: train verso → test recto (Takahashi)
python vw_suite.py --preset recipe --train_sides v --test_sides r --transcriber H


------------------------------------------------------------
SECTIONAL TESTS (RECTO → VERSO)
------------------------------------------------------------

# Herbal section (folios 1–66), Takahashi
python vw_suite.py --preset herbal --train_sides r --test_sides v --transcriber H

# Middle / "between" section (folios 67–102), Takahashi
python vw_suite.py --preset between --train_sides r --test_sides v --transcriber H

# Entire corpus, Takahashi
python vw_suite.py --preset all --train_sides r --test_sides v --transcriber H


------------------------------------------------------------
TRANSCRIBER ROBUSTNESS TESTS (HERBAL, RECTO → VERSO)
------------------------------------------------------------

# Takahashi (H)
python vw_suite.py --preset herbal --train_sides r --test_sides v --transcriber H

# Friedman (F)
python vw_suite.py --preset herbal --train_sides r --test_sides v --transcriber F

# Courier (C)
python vw_suite.py --preset herbal --train_sides r --test_sides v --transcriber C

# Stolfi (U)
python vw_suite.py --preset herbal --train_sides r --test_sides v --transcriber U


------------------------------------------------------------
FULL CORPUS TRANSCRIBER CHECK (RECTO → VERSO)
------------------------------------------------------------

# Entire corpus, Stolfi
python vw_suite.py --preset all --train_sides r --test_sides v --transcriber U


------------------------------------------------------------
SPLAT HANDLING CONTROLS
------------------------------------------------------------

# DEFAULT (methodology used in paper):
#   Tokens containing '!' are DROPPED
python vw_suite.py --preset recipe --train_sides r --test_sides v --transcriber H

# OPTIONAL CONTROL:
#   Treat '!' as a word boundary instead of dropping tokens
python vw_suite.py --preset recipe --train_sides r --test_sides v --transcriber H --split_splats


------------------------------------------------------------
SMOOTHING SENSITIVITY CHECK
------------------------------------------------------------

# Increase additive smoothing (example K = 1.0)
python vw_suite.py --preset recipe --train_sides r --test_sides v --transcriber H --K 1.0

------------------------------------------------------------
RANDOMIZED SHUFFLE TEST
------------------------------------------------------------

# Shuffles characters *within each token only* (token lengths and per-token character multisets preserved);
# destroys internal adjacency while retaining token boundaries and global character inventory.
python vw_suite.py --preset recipe --train_sides r --test_sides v --randomize within_tokens --rand_seed 1234

# Shuffles all characters *globally across the corpus* (overall character frequencies preserved);
# destroys token boundaries, positional structure, and token-level morphology.
python vw_suite.py --preset recipe --train_sides r --test_sides v --randomize global_chars --rand_seed 1234


------------------------------------------------------------
GUTENBERG CONTROL CORPUS (50/50 SPLIT)
------------------------------------------------------------

# Run on a Project Gutenberg text (auto strips headers)
python vw_suite.py --corpus text --text_file <file name> --min_len 2 --K 0.1


------------------------------------------------------------
EXPECTED OUTPUT METRICS
------------------------------------------------------------

- Bigram legality reject %
- Bigram bits / character (BPC2)
- Trigram bits / character (BPC3)
- ΔBPC = (Bigram BPC − Trigram BPC), computed from printed values
- Positional bits per character:
    * initial
    * medial
    * final
- Levenshtein ≤1 overlap (token %, type %)

All printed values are deterministically rounded and stable across runs.

============================================================
"""

import argparse
import math
import unicodedata
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from decimal import Decimal
import random
from typing import Optional, List, Dict, Set, Tuple


# ============================================================
# OUTPUT FORMATTING (LOCKED / DETERMINISTIC)
# ============================================================

def _q4(x: Optional[float]):
    """Quantize to 4 decimals deterministically for printed metrics and derived deltas."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return Decimal(f"{x:.4f}")

def _fmt_q(q: Optional[Decimal]) -> str:
    return "NA" if q is None else format(q, "f")

def _fmt(x: Optional[float], nd: int = 4) -> str:
    return "NA" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


# ============================================================
# NORMALIZATION + TOKENIZATION
# ============================================================

SPLAT_CHAR = "!"  # splat marker used in our experiments

def normalize_az(s: str) -> str:
    """
    Convert to a-z only; anything else becomes a boundary.
    (Used for the --split_splats mode and for Gutenberg text.)
    """
    s = unicodedata.normalize("NFD", s)
    out: List[str] = []
    prev_space = False
    for ch in s:
        if unicodedata.category(ch) == "Mn":
            continue
        c = ch.lower()
        if "a" <= c <= "z":
            out.append(c)
            prev_space = False
        else:
            if not prev_space:
                out.append(" ")
                prev_space = True
    return "".join(out)

def tokenize(text: str, min_len: int = 2) -> List[str]:
    norm = normalize_az(text)
    return [w for w in norm.split() if len(w) >= min_len]

def tokenize_payload(payload: str, min_len: int = 2, *, split_splats: bool = False) -> List[str]:
    """
    Tokenize an IVTFF payload.

    split_splats=True:
        Treat '!' as a boundary (i.e., like any non-letter). Uses normalize_az().
    split_splats=False (paper default):
        DROP tokens that contain '!' BUT only after normalizing separators so '.' does not glue neighbors
        into the same raw chunk and get them incorrectly dropped.
    """
    if split_splats:
        return tokenize(payload, min_len)

    # Normalize separators FIRST but preserve '!' for detection.
    # This prevents "wordA.wordB!wordC" from being treated as one chunk and nuking wordA/wordC.
    s = unicodedata.normalize("NFD", payload)
    out: List[str] = []
    buf: List[str] = []
    has_splat = False

    def flush():
        nonlocal buf, has_splat
        if buf and not has_splat:
            w = "".join(buf)
            if len(w) >= min_len:
                out.append(w)
        buf = []
        has_splat = False

    for ch in s:
        if unicodedata.category(ch) == "Mn":
            continue
        c = ch.lower()
        if c == SPLAT_CHAR:
            has_splat = True
            # keep accumulating letters in this token; it will be dropped at flush()
            continue
        if "a" <= c <= "z":
            buf.append(c)
        else:
            flush()

    flush()
    return out


# ============================================================
# EXCLUSION LIST (physically missing folios)
# ============================================================

EXCLUDED_FOLIOS = {
    'f12r','f12v','f59r','f59v','f60r','f60v','f61r','f61v',
    'f62r','f62v','f63r','f63v','f64r','f64v','f101r2',
    'f109r','f109v','f110r','f110v','f116v'
}


# ============================================================
# LSI PARSER (INLINE, IVTFF-AWARE)
# ============================================================

_FOLIO_ID_RE  = re.compile(r"^f(\d+)(r|v)(\d*)$", re.IGNORECASE)
_LSI_LINE_RE  = re.compile(r"^<f(\d+)(r|v)(\d*)\.[^;]*;([A-Za-z])>\s*(.*)$", re.IGNORECASE)
_TAG_RE       = re.compile(r"<[^>]+>")

def folio_num(fid: str) -> int:
    m = _FOLIO_ID_RE.match(fid)
    if not m:
        raise ValueError(f"Malformed folio ID: {fid}")
    return int(m.group(1))

def folio_side(fid: str) -> str:
    m = _FOLIO_ID_RE.match(fid)
    if not m:
        raise ValueError(f"Malformed folio ID: {fid}")
    return m.group(2).lower()

def folio_suffix(fid: str) -> int:
    m = _FOLIO_ID_RE.match(fid)
    if not m:
        raise ValueError(f"Malformed folio ID: {fid}")
    s = m.group(3)
    return int(s) if s else 0

def folio_sort_key(fid: str):
    # deterministic: numeric folio, then r before v, then suffix int
    side = folio_side(fid)
    side_key = 0 if side == "r" else 1
    return (folio_num(fid), side_key, folio_suffix(fid))

def parse_lsi_lines(lsi_path: str, transcriber: str = "H"):
    """
    Yields (folio_id, payload) for the specified transcriber.
    Strips IVTFF tags and skips excluded folios.
    """
    text = Path(lsi_path).read_text(encoding="utf-8", errors="replace")
    tr = (transcriber or "").strip().upper()
    if len(tr) != 1:
        raise ValueError("--transcriber must be a single character (e.g., H, C, F, U)")

    found_any = False
    for line in text.splitlines():
        if not line.startswith("<f"):
            continue
        m = _LSI_LINE_RE.match(line)
        if not m:
            continue
        num, side, sfx, trc, payload = m.groups()
        if trc.upper() != tr:
            continue
        fid = f"f{int(num)}{side.lower()}{sfx}"
        if fid in EXCLUDED_FOLIOS:
            continue
        payload = _TAG_RE.sub(" ", payload)
        payload = payload.strip()
        if not payload:
            continue
        found_any = True
        yield fid, payload

    if not found_any:
        raise ValueError(f"No valid transcription lines found for transcriber '{tr}' in LSI file")

def list_folios(lsi_path: str, transcriber: str = "H") -> List[str]:
    return sorted({fid for fid,_ in parse_lsi_lines(lsi_path, transcriber)}, key=folio_sort_key)

def select_folios(lsi_path: str, start: Optional[int], end: Optional[int], sides: str, transcriber: str = "H") -> List[str]:
    allowed = set((sides or "").lower())
    if not allowed or not allowed.issubset({"r","v"}):
        raise ValueError("--train_sides/--test_sides must be one of: r, v, rv")
    out: List[str] = []
    for fid in list_folios(lsi_path, transcriber):
        n = folio_num(fid)
        if start is not None and n < start:
            continue
        if end is not None and n > end:
            continue
        if folio_side(fid) not in allowed:
            continue
        out.append(fid)
    out.sort(key=folio_sort_key)
    return out

def load_voynich_tokens(lsi_path: str, folios: List[str], min_len: int, transcriber: str = "H", *, split_splats: bool = False) -> List[str]:
    fs = set(folios)
    toks: List[str] = []
    for fid, payload in parse_lsi_lines(lsi_path, transcriber):
        if fid in fs:
            toks.extend(tokenize_payload(payload, min_len, split_splats=split_splats))
    return toks


# ============================================================
# GUTENBERG HANDLING (optional control corpus)
# ============================================================

_GUT_START_RE = re.compile(r"\*{0,3}\s*START OF (?:THIS|THE)\s+PROJECT GUTENBERG", re.IGNORECASE)
_GUT_END_RE   = re.compile(r"\*{0,3}\s*END OF (?:THIS|THE)\s+PROJECT GUTENBERG", re.IGNORECASE)

def strip_gutenberg(text: str) -> str:
    """
    More robust Gutenberg stripper:
    - Accepts common START/END marker variants (case-insensitive, with/without ***)
    - If markers not found, falls back to heuristic windowing without crashing.
    """
    lines = text.splitlines()

    start_idx = None
    end_idx = None

    for i, l in enumerate(lines):
        if _GUT_START_RE.search(l) or "START OF THE PROJECT GUTENBERG EBOOK" in l.upper():
            start_idx = i + 1
            break

    for i in range(len(lines) - 1, -1, -1):
        l = lines[i]
        if _GUT_END_RE.search(l) or "END OF THE PROJECT GUTENBERG EBOOK" in l.upper():
            end_idx = i
            break

    if start_idx is not None and end_idx is not None and start_idx < end_idx:
        return "\n".join(lines[start_idx:end_idx])

    # Heuristic fallback: try to find START within first 500 lines, END within last 500 lines
    # If still not found, return full text (deterministic, non-crashing).
    scan_head = lines[:min(500, len(lines))]
    scan_tail = lines[max(0, len(lines) - 500):]

    for i, l in enumerate(scan_head):
        if "PROJECT GUTENBERG" in l.upper() and "START" in l.upper():
            start_idx = i + 1
            break

    for j in range(len(scan_tail) - 1, -1, -1):
        l = scan_tail[j]
        if "PROJECT GUTENBERG" in l.upper() and "END" in l.upper():
            end_idx = (len(lines) - len(scan_tail)) + j
            break

    if start_idx is not None and end_idx is not None and start_idx < end_idx:
        return "\n".join(lines[start_idx:end_idx])

    print("Warning: Gutenberg START/END markers not found; using full text.", file=sys.stderr)
    return text

def load_text_tokens(path: str, min_len: int) -> List[str]:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    core = strip_gutenberg(raw)
    return tokenize(core, min_len)


# ============================================================
# LEAKAGE-FREE OOV HANDLING
# ============================================================

# Dedicated OOV bucket symbol used internally for scoring only.
# This is NOT a Voynich character; it is introduced by this script when TEST contains a-z letters never seen in TRAIN.
UNK_CHAR = "\x00"  # internal OOV bucket (never produced by normalize_az from Voynich because Voynich tokens are a-z already)

def train_char_inventory(train_tokens: List[str]) -> Set[str]:
    return set("".join(train_tokens))

def map_oov_token(w: str, train_chars: Set[str]) -> str:
    return "".join((c if c in train_chars else UNK_CHAR) for c in w)

def map_oov_tokens(tokens: List[str], train_chars: Set[str]) -> List[str]:
    return [map_oov_token(w, train_chars) for w in tokens]


# ============================================================
# METRICS
# ============================================================

def bigram_legality_reject_rate(train_tokens: List[str], test_tokens: List[str]) -> float:
    """
    Token-level reject%: a TEST token is rejected if it contains ANY bigram
    that was not seen in TRAIN.

    IMPORTANT (consistency fix):
    - We evaluate bigrams on the same sequence used by the n-gram model:
      seq = "^" + token + "$"
    """
    seen = set()

    # TRAIN bigrams (with boundaries)
    for w in train_tokens:
        seq = "^" + w + "$"
        for i in range(len(seq) - 1):
            seen.add(seq[i:i+2])

    rejects = 0

    # TEST bigrams (with boundaries)
    for w in test_tokens:
        seq = "^" + w + "$"
        for i in range(len(seq) - 1):
            if seq[i:i+2] not in seen:
                rejects += 1
                break

    return rejects / len(test_tokens) * 100 if test_tokens else float("nan")


def train_ngram(tokens: List[str], n: int):
    c = Counter()
    ctx = Counter()
    alph = set("^$")

    for w in tokens:
        seq = "^"*(n-1) + w + "$"
        alph |= set(seq)
        for i in range(n-1, len(seq)):
            ng = tuple(seq[i-(n-1):i+1])
            c[ng] += 1
            ctx[ng[:-1]] += 1

    return c, ctx, alph, n

def score_ngram(tokens: list[str], model, K: float) -> float:
    """
    BPC IS LOCKED (DO NOT CHANGE):
    - We score end-of-word '$' (it's included in lp)
    - We do NOT count '$' in the denominator (chars counts surface letters only)
    This matches the legacy behavior your submitted ΔBPC was based on.
    """
    c, ctx, alph, n = model
    V = len(alph)  # LOCKED legacy event space (includes '^' and '$'); do not change

    lp = 0.0
    chars = 0

    for w in tokens:
        seq = "^" * (n - 1) + w + "$"   # ALWAYS score '$'
        for i in range(n - 1, len(seq)):
            ng = tuple(seq[i - (n - 1): i + 1])
            p = (c.get(ng, 0) + K) / (ctx.get(ng[:-1], 0) + K * V)
            lp += -math.log2(p)
            if ng[-1] != "$":          # LOCKED: denominator = surface letters only
                chars += 1

    return lp / chars if chars else float("nan")


def positional_bits(train: List[str], test: List[str], K: float, train_chars: Set[str]) -> Dict[str, float]:
    """
    Leakage-free positional cross-entropy (bits/event):

    init:  P(c1 | START)          (word-initial)
    med:   P(c_{i+1} | c_i)       for within-word transitions only (char -> char)
    fin:   P(END | c_last)        (word-final termination)

    Smoothing event spaces (correctly separated):
    - init: next symbols are (train_chars + UNK_CHAR)                  => V_start
    - med:  next symbols are (train_chars + UNK_CHAR)                  => V_med
    - fin:  next symbols are {END} only (single-outcome termination)   => V_fin = 1
    """
    START = "^"
    END = "$"

    # Train counts
    c_init = Counter()     # (START, c1)
    ctx_init = Counter()   # START
    c_med = Counter()      # (c_i, c_{i+1})
    ctx_med = Counter()    # c_i
    c_fin = Counter()      # (c_last, END)
    ctx_fin = Counter()    # c_last

    for w in train:
        if not w:
            continue

        # init
        c1 = w[0]
        c_init[(START, c1)] += 1
        ctx_init[START] += 1

        # med (within-word transitions only)
        if len(w) >= 2:
            for i in range(len(w) - 1):
                c0 = w[i]
                c1n = w[i + 1]
                c_med[(c0, c1n)] += 1
                ctx_med[c0] += 1

        # fin (termination)
        cl = w[-1]
        c_fin[(cl, END)] += 1
        ctx_fin[cl] += 1

    V_start = len(train_chars) + 1   # chars + UNK
    V_med   = len(train_chars) + 1   # chars + UNK (NO END in medial)
    V_fin   = 1                      # only END is a legal next event

    out: Dict[str, float] = {}

    # INIT: START -> c1
    lp = 0.0
    steps = 0
    denom_init = ctx_init.get(START, 0) + K * V_start
    for w in test:
        if not w:
            continue
        c1 = w[0]
        p = (c_init.get((START, c1), 0) + K) / denom_init
        lp += -math.log2(p)
        steps += 1
    out["init"] = lp / steps if steps else float("nan")

    # MED: c_i -> c_{i+1}  (NO END allowed here)
    lp = 0.0
    steps = 0
    denom_cache: Dict[str, float] = {}
    for w in test:
        if len(w) < 2:
            continue
        for i in range(len(w) - 1):
            c0 = w[i]
            c1n = w[i + 1]
            denom = denom_cache.get(c0)
            if denom is None:
                denom = ctx_med.get(c0, 0) + K * V_med
                denom_cache[c0] = denom
            p = (c_med.get((c0, c1n), 0) + K) / denom
            lp += -math.log2(p)
            steps += 1
    out["med"] = lp / steps if steps else float("nan")

    # FIN: c_last -> END  (single-outcome termination; V=1)
    lp = 0.0
    steps = 0
    denom_cache_fin: Dict[str, float] = {}
    for w in test:
        if not w:
            continue
        cl = w[-1]
        denom = denom_cache_fin.get(cl)
        if denom is None:
            denom = ctx_fin.get(cl, 0) + K * V_fin
            denom_cache_fin[cl] = denom
        p = (c_fin.get((cl, END), 0) + K) / denom
        lp += -math.log2(p)
        steps += 1
    out["fin"] = lp / steps if steps else float("nan")

    return out

# ============================================================
# LEVENSHTEIN ≤ 1
# ============================================================

def _del_sigs(w: str):
    for i in range(len(w)):
        yield w[:i] + w[i+1:]

def build_edit1_index(types: Set[str]):
    sig2lens = defaultdict(set)
    subs = set()
    for t in types:
        L = len(t)
        for s in _del_sigs(t):
            sig2lens[s].add(L)
        for i in range(L):
            subs.add(t[:i] + "*" + t[i+1:])
    return types, sig2lens, subs

def has_lev1(w: str, train: Set[str], sig2lens, subs) -> bool:
    if w in train:
        return True
    L = len(w)
    for i in range(L):
        if w[:i] + "*" + w[i+1:] in subs:
            return True
    if (L + 1) in sig2lens.get(w, ()):
        return True
    for s in _del_sigs(w):
        if s in train or L in sig2lens.get(s, ()):
            return True
    return False

def dist01(train_tokens: List[str], test_tokens: List[str]) -> Dict[str, float]:
    train = set(train_tokens)
    _, sig2lens, subs = build_edit1_index(train)
    cnt = Counter(test_tokens)
    tot = sum(cnt.values())
    d0 = d1 = t0 = t1 = 0
    for w, c in cnt.items():
        if w in train:
            d0 += c; t0 += 1
        elif has_lev1(w, train, sig2lens, subs):
            d1 += c; t1 += 1
    return {
        "le1_tok": (d0 + d1) / tot * 100 if tot else float("nan"),
        "le1_type": (t0 + t1) / len(cnt) * 100 if cnt else float("nan"),
    }


def shuffle_within(tokens: List[str], rng: random.Random) -> List[str]:
    out: List[str] = []
    for w in tokens:
        ch = list(w)
        rng.shuffle(ch)
        out.append("".join(ch))
    return out

def shuffle_global(tokens: List[str], rng: random.Random) -> List[str]:
    pool = [c for w in tokens for c in w]
    rng.shuffle(pool)
    out: List[str] = []
    k = 0
    for w in tokens:
        out.append("".join(pool[k:k+len(w)]))
        k += len(w)
    return out


# ============================================================
# MAIN
# ============================================================

PRESETS = {
    "recipe": (103, 116),
    "herbal": (1, 66),
    "between": (67, 102),
    "all": (None, None),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lsi", default="LSI.txt")
    ap.add_argument("--preset", default="recipe")
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int)
    ap.add_argument("--train_sides", default="r")
    ap.add_argument("--test_sides", default="v")
    ap.add_argument("--min_len", type=int, default=2)
    ap.add_argument("--corpus", choices=["voynich", "text"], default="voynich")
    ap.add_argument("--text_file")
    ap.add_argument("--transcriber", default="H")
    ap.add_argument("--K", type=float, default=0.1)
    ap.add_argument("--split_splats", action="store_true",
                    help="Optional: split tokens on splat (!) by treating it as a boundary (baseline). Default: drop any token containing !.")
    ap.add_argument(
        "--randomize",
        choices=["none", "within_tokens", "global_chars"],
        default="none",
        help="Randomize TEST tokens only: within_tokens or global_chars"
    )
    ap.add_argument(
        "--rand_seed",
        type=int,
        default=1,
        help="Random seed for TEST randomization"
    )
    args = ap.parse_args()

    K = args.K

    # ----------------------------
    # Load tokens (train/test)
    # ----------------------------
    if args.corpus == "text":
        if not args.text_file:
            raise ValueError("--text_file is required when --corpus text")
        tokens = load_text_tokens(args.text_file, args.min_len)
        mid = len(tokens) // 2
        train = tokens[:mid]
        test = tokens[mid:]
    else:
        if not Path(args.lsi).exists():
            raise FileNotFoundError("LSI file not found")

        if args.preset in PRESETS:
            s, e = PRESETS[args.preset]
        else:
            s, e = (args.start, args.end)
            if args.start is None and args.end is None:
                raise ValueError(f"Unknown --preset '{args.preset}'. Use one of: {', '.join(PRESETS.keys())} or provide --start/--end.")

        start = args.start if args.start is not None else s
        end   = args.end   if args.end   is not None else e

        train_f = select_folios(args.lsi, start, end, args.train_sides, args.transcriber)
        test_f  = select_folios(args.lsi, start, end, args.test_sides, args.transcriber)

        train = load_voynich_tokens(args.lsi, train_f, args.min_len, args.transcriber, split_splats=args.split_splats)
        test  = load_voynich_tokens(args.lsi, test_f,  args.min_len, args.transcriber, split_splats=args.split_splats)

    # ----------------------------
    # Optional randomization (TEST only)
    # ----------------------------
    if args.randomize != "none":
        rng = random.Random(args.rand_seed)
        if args.randomize == "within_tokens":
            test = shuffle_within(test, rng)
        elif args.randomize == "global_chars":
            test = shuffle_global(test, rng)

    # ----------------------------
    # Leakage-free OOV mapping
    # ----------------------------
    train_chars = train_char_inventory(train)
    test_mapped = map_oov_tokens(test, train_chars)

    # ----------------------------
    # Metrics
    # ----------------------------
    rej = bigram_legality_reject_rate(train, test_mapped)

    bi = train_ngram(train, 2)
    tri = train_ngram(train, 3)

    # Leakage-free smoothing alphabet sizes (TRAIN ONLY + fixed UNK bucket + boundaries)
    # For score_ngram: alphabet includes '^' and '$' in the model, plus UNK for OOV mapped in TEST.
    bi_bpc  = score_ngram(test_mapped, bi, K)
    tri_bpc = score_ngram(test_mapped, tri, K)


    # Quantize printed values; compute delta from the same rounded values.
    bi_q = _q4(bi_bpc)
    tri_q = _q4(tri_bpc)
    delta_q = (bi_q - tri_q) if (bi_q is not None and tri_q is not None) else None

    pos = positional_bits(train, test_mapped, K, train_chars=train_chars)

    # Levenshtein overlap should use the original tokens (not OOV-mapped),
    # because it is a lexical overlap proxy, not a probabilistic model score.
    d01 = dist01(train, test)

    # ----------------------------
    # Output
    # ----------------------------
    print("| Metric | Value |")
    print("| --- | --- |")
    print(f"| Bigram legality reject% | {_fmt(rej,3)} |")
    print(f"| Bigram bits/char | {_fmt_q(bi_q)} |")
    print(f"| Trigram bits/char | {_fmt_q(tri_q)} |")
    print(f"| ΔBPC (Bigram−Trigram) | {_fmt_q(delta_q)} |")
    print(f"| Positional bits (init) | {_fmt(pos['init'])} |")
    print(f"| Positional bits (med) | {_fmt(pos['med'])} |")
    print(f"| Positional bits (fin) | {_fmt(pos['fin'])} |")
    print(f"| Levenshtein ≤1 token % | {_fmt(d01['le1_tok'],2)} |")
    print(f"| Levenshtein ≤1 type % | {_fmt(d01['le1_type'],2)} |")

if __name__ == "__main__":
    main()
