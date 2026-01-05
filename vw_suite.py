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
    Leakage-free legality:
    - Build seen bigrams from TRAIN only.
    - TEST is assumed already OOV-mapped to UNK_CHAR (if desired).
    """
    seen = set()
    for w in train_tokens:
        for i in range(len(w) - 1):
            seen.add(w[i:i+2])

    rejects = 0
    for w in test_tokens:
        for i in range(len(w) - 1):
            if w[i:i+2] not in seen:
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

    # Deterministic ordering for alph (for stable derived counts)
    alph = set(alph)
    return c, ctx, alph

def ngram_bpc(train_tokens: List[str], test_tokens: List[str], n: int, K: float) -> float:
    """
    Bits per character for an n-gram character model with additive smoothing.
    - train/test tokens are expected to be already OOV-mapped as desired.
    - Model uses start symbols ^ and end symbol $.
    """
    c, ctx, alph = train_ngram(train_tokens, n)
    V = len(alph)
    if V == 0:
        return float("nan")

    total_bits = 0.0
    total_chars = 0

    for w in test_tokens:
        seq = "^"*(n-1) + w + "$"
        for i in range(n-1, len(seq)):
            ng = tuple(seq[i-(n-1):i+1])
            context = ng[:-1]
            num = c.get(ng, 0)
            den = ctx.get(context, 0)
            p = (num + K) / (den + K*V)
            pos = i - (n-1)
            if 0 <= pos < len(w):  # only score real characters, not the terminal '$'
                total_bits += -math.log2(p)
                total_chars += 1

    return (total_bits / total_chars) if total_chars > 0 else float("nan")

def positional_bpc(train_tokens: List[str], test_tokens: List[str], K: float) -> Dict[str, float]:
    """
    Position-specific BPC for initial, medial, final characters.
    Unigram positional model with additive smoothing.
    """
    # TRAIN counts
    counts = {"init": Counter(), "med": Counter(), "fin": Counter()}
    totals = {"init": 0, "med": 0, "fin": 0}
    alph = set()

    for w in train_tokens:
        if not w:
            continue
        alph |= set(w)
        counts["init"][w[0]] += 1
        totals["init"] += 1
        if len(w) > 2:
            for ch in w[1:-1]:
                counts["med"][ch] += 1
                totals["med"] += 1
        if len(w) > 1:
            counts["fin"][w[-1]] += 1
            totals["fin"] += 1

    V = max(1, len(alph))

    # TEST scoring
    out = {}
    for key in ["init", "med", "fin"]:
        bits = 0.0
        nchar = 0
        for w in test_tokens:
            if not w:
                continue
            if key == "init":
                chs = [w[0]]
            elif key == "fin":
                chs = [w[-1]] if len(w) > 1 else []
            else:
                chs = list(w[1:-1]) if len(w) > 2 else []
            for ch in chs:
                num = counts[key].get(ch, 0)
                den = totals[key]
                p = (num + K) / (den + K*V)
                bits += -math.log2(p)
                nchar += 1
        out[key] = bits / nchar if nchar > 0 else float("nan")
    return out

def levenshtein_le1(a: str, b: str) -> bool:
    """
    True iff Levenshtein distance <= 1 (fast exact check for <=1).
    """
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    # Same length: at most one substitution
    if la == lb:
        dif = 0
        for x, y in zip(a, b):
            if x != y:
                dif += 1
                if dif > 1:
                    return False
        return True
    # Ensure a is shorter
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    # Now lb = la+1: at most one insertion in b
    i = j = 0
    dif = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            dif += 1
            if dif > 1:
                return False
            j += 1
    return True

def le1_overlap(train_tokens: List[str], test_tokens: List[str]) -> Tuple[float, float]:
    """
    Returns (token_overlap_pct, type_overlap_pct) where overlap is defined by existence of any train token
    within Levenshtein distance <= 1.
    """
    if not test_tokens:
        return (float("nan"), float("nan"))

    train_types = sorted(set(train_tokens))
    test_types = sorted(set(test_tokens))

    # token-level
    hit = 0
    for w in test_tokens:
        ok = False
        # early exit by length window using sorted list? simplest: brute by length bucket
        # build length buckets once
        # (kept deterministic / clear rather than micro-optimized)
        for t in train_types:
            if abs(len(t) - len(w)) > 1:
                continue
            if levenshtein_le1(t, w):
                ok = True
                break
        if ok:
            hit += 1
    token_pct = hit / len(test_tokens) * 100

    # type-level
    thit = 0
    for w in test_types:
        ok = False
        for t in train_types:
            if abs(len(t) - len(w)) > 1:
                continue
            if levenshtein_le1(t, w):
                ok = True
                break
        if ok:
            thit += 1
    type_pct = thit / len(test_types) * 100 if test_types else float("nan")

    return (token_pct, type_pct)


# ============================================================
# RANDOMIZATION CONTROLS
# ============================================================

def randomize_within_tokens(tokens: List[str], rng: random.Random) -> List[str]:
    out = []
    for w in tokens:
        chars = list(w)
        rng.shuffle(chars)
        out.append("".join(chars))
    return out

def randomize_global_chars(tokens: List[str], rng: random.Random) -> List[str]:
    lengths = [len(w) for w in tokens]
    pool = list("".join(tokens))
    rng.shuffle(pool)
    out = []
    idx = 0
    for L in lengths:
        out.append("".join(pool[idx:idx+L]))
        idx += L
    return out


# ============================================================
# PRESETS
# ============================================================

PRESETS = {
    "recipe": (103, 116),
    "herbal": (1, 66),
    "between": (67, 102),
    "all": (None, None),
}



# ============================================================
# INTERACTIVE MENU (optional)
# ============================================================

def _prompt(msg: str, default: str = "") -> str:
    d = f" [{default}]" if default else ""
    try:
        s = input(f"{msg}{d}: ").strip()
    except EOFError:
        s = ""
    return s if s else default

def _prompt_choice(msg: str, choices: Set[str], default: str) -> str:
    while True:
        v = _prompt(msg, default).strip()
        if v in choices:
            return v
        print(f"Invalid choice. Options: {', '.join(sorted({c.upper() for c in choices}))}")

def build_args_from_menu() -> List[str]:
    """
    Interactive menu that builds argv (without printing the constructed flags).
    Designed for: python vw_suite.py   (no args)

    Supports:
      - Any preset in forward (r->v) or reverse (v->r)
      - Voynich corpus (LSI) OR external text corpus (--corpus text)
    """
    print()
    print("Voynich Structural Analysis")
    print()

    corpus = _prompt_choice("Corpus (V=Voynich LSI, T=Text file)", {"V","T","v","t"}, "V").upper()

    args: List[str] = []

    # Common optional knobs (apply to both corpus types)
    k  = _prompt("Smoothing K", "0.1")
    ml = _prompt("Min token length", "2")
    rnd = _prompt_choice("Randomize TEST (N=none, W=within_tokens, G=global_chars)",
                         {"N","W","G","n","w","g"}, "N").upper()
    seed = "1234"
    if rnd != "N":
        seed = _prompt("Random seed", "1234")

    args += ["--K", k]
    args += ["--min_len", ml]

    if rnd == "W":
        args += ["--randomize", "within_tokens", "--rand_seed", seed]
    elif rnd == "G":
        args += ["--randomize", "global_chars", "--rand_seed", seed]

    if corpus == "T":
        # Text corpus path
        print()
        print("Text corpus mode")
        print()

        tf = _prompt("Text file path", "")
        if not tf:
            print("No text file provided. Exiting.")
            raise SystemExit(0)

        args = ["--corpus", "text", "--text_file", tf] + args
        return args

    # Voynich (LSI) corpus mode
    print()
    print("Voynich (LSI) mode")
    print()
    print("1) Recipe (103–116)")
    print("2) Herbal (1–66)")
    print("3) Between (67–102)")
    print("4) Entire corpus")
    print("X) Exit")
    print()

    sel = _prompt_choice("Select section (1-4 or X)", {"1","2","3","4","X","x"}, "1").upper()
    if sel == "X":
        raise SystemExit(0)

    if sel == "1":
        preset = "recipe"
    elif sel == "2":
        preset = "herbal"
    elif sel == "3":
        preset = "between"
    else:
        preset = "all"

    direction = _prompt_choice("Direction (F=recto→verso, R=verso→recto)", {"F","R","f","r"}, "F").upper()
    if direction == "F":
        train_sides, test_sides = "r", "v"
    else:
        train_sides, test_sides = "v", "r"

    tr = _prompt_choice("Transcriber (H,C,F,U)", {"H","C","F","U","h","c","f","u"}, "H").upper()
    spl = _prompt_choice("Splat handling (D=drop tokens with !, S=split on !)", {"D","S","d","s"}, "D").upper()

    # LSI path override (keep existing default behavior intact)
    lsi = _prompt("LSI transcription path", "LSI.txt")
    if lsi:
        args = ["--lsi", lsi] + args

    # Voynich-specific flags
    args += ["--corpus", "voynich"]
    args += ["--preset", preset]
    args += ["--train_sides", train_sides, "--test_sides", test_sides]
    args += ["--transcriber", tr]

    if spl == "S":
        args += ["--split_splats"]

    return args



# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Voynich Structural Analysis Suite")
    ap.add_argument("--lsi", default="LSI.txt", help="Path to LSI transcription file")
    ap.add_argument("--corpus", default="voynich", choices=["voynich","text"], help="Corpus type: voynich (LSI) or text (Gutenberg)")
    ap.add_argument("--text_file", default=None, help="Path to text file when --corpus text")

    ap.add_argument("--preset", default="recipe", choices=sorted(PRESETS.keys()), help="Preset folio range")
    ap.add_argument("--train_sides", default="r", help="Train sides: r, v, or rv")
    ap.add_argument("--test_sides", default="v", help="Test sides: r, v, or rv")
    ap.add_argument("--transcriber", default="H", help="Transcriber code (H,C,F,U,...)")
    ap.add_argument("--min_len", type=int, default=2, help="Minimum token length")
    ap.add_argument("--K", type=float, default=0.1, help="Additive smoothing K")

    ap.add_argument("--split_splats", action="store_true", help="Treat '!' as boundary instead of dropping tokens containing it")
    ap.add_argument("--randomize", default="none", choices=["none","within_tokens","global_chars"],
                    help="Randomization control for TEST tokens")
    ap.add_argument("--rand_seed", type=int, default=1234, help="Random seed for randomization controls")

    return ap


# ============================================================
# MAIN RUN
# ============================================================

def run_voynich(args: argparse.Namespace):
    start, end = PRESETS[args.preset]
    train_folios = select_folios(args.lsi, start, end, args.train_sides, args.transcriber)
    test_folios  = select_folios(args.lsi, start, end, args.test_sides,  args.transcriber)

    train_tokens_raw = load_voynich_tokens(args.lsi, train_folios, args.min_len, args.transcriber, split_splats=args.split_splats)
    test_tokens_raw  = load_voynich_tokens(args.lsi, test_folios,  args.min_len, args.transcriber, split_splats=args.split_splats)

    # Leakage-free OOV mapping: map TEST chars not seen in TRAIN into UNK_CHAR
    train_chars = train_char_inventory(train_tokens_raw)
    train_tokens = train_tokens_raw
    test_tokens  = map_oov_tokens(test_tokens_raw, train_chars)

    # Optional randomization of TEST
    rng = random.Random(args.rand_seed)
    if args.randomize == "within_tokens":
        test_tokens = randomize_within_tokens(test_tokens, rng)
    elif args.randomize == "global_chars":
        test_tokens = randomize_global_chars(test_tokens, rng)

    # Metrics
    rej = bigram_legality_reject_rate(train_tokens, test_tokens)
    bpc2 = ngram_bpc(train_tokens, test_tokens, n=2, K=args.K)
    bpc3 = ngram_bpc(train_tokens, test_tokens, n=3, K=args.K)
    d_bpc = None
    qbpc2 = _q4(bpc2)
    qbpc3 = _q4(bpc3)
    if qbpc2 is not None and qbpc3 is not None:
        d_bpc = qbpc2 - qbpc3

    pos = positional_bpc(train_tokens, test_tokens, K=args.K)
    tok_ov, type_ov = le1_overlap(train_tokens, test_tokens)

    # Output (pipe-free)
    print()
    print("=== Voynich Structural Analysis ===")
    print(f"Preset: {args.preset}")
    print(f"Train sides: {args.train_sides}   Test sides: {args.test_sides}")
    print(f"Transcriber: {args.transcriber}")
    print(f"Min token length: {args.min_len}   K: {args.K}")
    print(f"Splat handling: {'SPLIT' if args.split_splats else 'DROP TOKENS WITH !'}")
    print(f"Randomize: {args.randomize}   Seed: {args.rand_seed}")
    print(f"Train folios: {len(train_folios)}   Test folios: {len(test_folios)}")
    print(f"Train tokens: {len(train_tokens)}   Test tokens: {len(test_tokens)}")
    print()

    print(f"Bigram legality reject %: { _fmt(rej, 4) }")
    print(f"Bigram BPC (BPC2):        { _fmt(bpc2, 4) }")
    print(f"Trigram BPC (BPC3):       { _fmt(bpc3, 4) }")
    print(f"ΔBPC (BPC2−BPC3):         { _fmt_q(d_bpc) }")
    print()

    print("Positional BPC")
    print(f"  initial: { _fmt(pos.get('init'), 4) }")
    print(f"  medial:  { _fmt(pos.get('med'), 4) }")
    print(f"  final:   { _fmt(pos.get('fin'), 4) }")
    print()

    print("Levenshtein ≤1 overlap")
    print(f"  token %: { _fmt(tok_ov, 4) }")
    print(f"  type %:  { _fmt(type_ov, 4) }")
    print()


def run_text(args: argparse.Namespace):
    if not args.text_file:
        raise ValueError("--text_file is required when --corpus text")

    tokens_raw = load_text_tokens(args.text_file, args.min_len)

    # Simple 50/50 split by token order for control corpus
    mid = len(tokens_raw) // 2
    train_tokens = tokens_raw[:mid]
    test_tokens_raw  = tokens_raw[mid:]

    # Leakage-free OOV mapping
    train_chars = train_char_inventory(train_tokens)
    test_tokens = map_oov_tokens(test_tokens_raw, train_chars)

    # Optional randomization of TEST
    rng = random.Random(args.rand_seed)
    if args.randomize == "within_tokens":
        test_tokens = randomize_within_tokens(test_tokens, rng)
    elif args.randomize == "global_chars":
        test_tokens = randomize_global_chars(test_tokens, rng)

    rej = bigram_legality_reject_rate(train_tokens, test_tokens)
    bpc2 = ngram_bpc(train_tokens, test_tokens, n=2, K=args.K)
    bpc3 = ngram_bpc(train_tokens, test_tokens, n=3, K=args.K)
    qbpc2 = _q4(bpc2)
    qbpc3 = _q4(bpc3)
    d_bpc = None
    if qbpc2 is not None and qbpc3 is not None:
        d_bpc = qbpc2 - qbpc3

    pos = positional_bpc(train_tokens, test_tokens, K=args.K)
    tok_ov, type_ov = le1_overlap(train_tokens, test_tokens)

    print()
    print("=== Text Control Structural Analysis ===")
    print(f"Text file: {args.text_file}")
    print(f"Min token length: {args.min_len}   K: {args.K}")
    print(f"Randomize: {args.randomize}   Seed: {args.rand_seed}")
    print(f"Train tokens: {len(train_tokens)}   Test tokens: {len(test_tokens)}")
    print()

    print(f"Bigram legality reject %: { _fmt(rej, 4) }")
    print(f"Bigram BPC (BPC2):        { _fmt(bpc2, 4) }")
    print(f"Trigram BPC (BPC3):       { _fmt(bpc3, 4) }")
    print(f"ΔBPC (BPC2−BPC3):         { _fmt_q(d_bpc) }")
    print()

    print("Positional BPC")
    print(f"  initial: { _fmt(pos.get('init'), 4) }")
    print(f"  medial:  { _fmt(pos.get('med'), 4) }")
    print(f"  final:   { _fmt(pos.get('fin'), 4) }")
    print()

    print("Levenshtein ≤1 overlap")
    print(f"  token %: { _fmt(tok_ov, 4) }")
    print(f"  type %:  { _fmt(type_ov, 4) }")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    # If run with no args in an interactive terminal, show menu and build args for the user.
    if len(sys.argv) == 1 and sys.stdin.isatty():
        argv = build_args_from_menu()
        # Replace sys.argv so argparse sees the constructed arguments
        sys.argv = [sys.argv[0]] + argv

    ap = build_parser()
    args = ap.parse_args(argv)

    if args.corpus == "voynich":
        run_voynich(args)
    else:
        run_text(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
