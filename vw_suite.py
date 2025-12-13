#!/usr/bin/env python3
import argparse
import math
import unicodedata
import random
from collections import Counter, defaultdict
from pathlib import Path


# ----------------------------
# Usage
#-----------------------------

# Recipe, rectos → versos
# python vw_suite.py --preset recipe --train_sides r --test_sides v
#
# Herbal, rectos → versos
# python vw_suite.py --preset herbal --train_sides r --test_sides v
#
# Train on versos, test on rectos (reverse sanity check)
# python vw_suite.py --preset recipe --train_sides v --test_sides r
#
# Whole corpus, train on rectos, test on versos
# python vw_suite.py --preset all --train_sides r --test_sides v
#
# Custom folio range
# python vw_suite.py --start 90 --end 116 --train_sides r --test_sides v
#
# Write CSV outputs
# python vw_suite.py --preset recipe --train_sides r --test_sides v --csv_prefix vw_recipe
#
# Symmetric (Default)
# python vw_suite.py --preset recipe --train_sides r --test_sides v --symmetric_short_filter true
#
# Asymmetric (test-only filter)
# python vw_suite.py --preset recipe --train_sides r --test_sides v --symmetric_short_filter false


# ----------------------------
# Normalization
# ----------------------------
def normalize_az(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    out = []
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

def tokenize(text: str, min_len: int = 2) -> list[str]:
    norm = normalize_az(text)
    return [w for w in norm.split() if len(w) >= min_len]

# ----------------------------
# Voynich via iParser
# ----------------------------
def import_iparser(iparser_dir: str):
    import sys
    sys.path.insert(0, iparser_dir)
    import iParser_guarded as ip
    return ip

def folio_num(folio_id: str) -> int | None:
    # folio IDs like "f103r", "f114v" etc.
    digits = "".join(ch for ch in folio_id if ch.isdigit())
    if not digits:
        return None
    return int(digits)

def select_folios(ip, lsi_path: str, transcriber: str,
                  start: int | None, end: int | None,
                  sides: str) -> list[str]:
    """
    sides: 'r', 'v', or 'both'
    """
    all_f = ip.list_folios(lsi_path, transcriber=transcriber)
    out = []
    for f in all_f:
        if not f.startswith("f"):
            continue
        n = folio_num(f)
        if n is None:
            continue
        if start is not None and n < start:
            continue
        if end is not None and n > end:
            continue

        if sides == "r" and not f.endswith("r"):
            continue
        if sides == "v" and not f.endswith("v"):
            continue
        out.append(f)
    return sorted(out)

def load_tokens_for_folios(ip, lsi_path: str, transcriber: str,
                          folios: list[str], min_len: int) -> list[str]:
    toks = []
    for fol in folios:
        for (_, _, line) in ip.parse_folio_lines_normalized(lsi_path, fol, transcriber=transcriber):
            toks.extend(tokenize(line, min_len=min_len))
    return toks

# ----------------------------
# Metrics: bigram legality (anywhere)
# ----------------------------
def bigram_legality_reject_rate(train_tokens: list[str], test_tokens: list[str]) -> float:
    train_bi = set()
    for w in train_tokens:
        for i in range(len(w) - 1):
            train_bi.add(w[i:i+2])

    rejects = 0
    for w in test_tokens:
        ok = True
        for i in range(len(w) - 1):
            if w[i:i+2] not in train_bi:
                ok = False
                break
        if not ok:
            rejects += 1

    return (rejects / len(test_tokens) * 100.0) if test_tokens else float("nan")

# ----------------------------
# Metrics: n-gram bits per character (bigram/trigram)
# ----------------------------
K = 0.1  # fixed add-k smoothing

def train_char_ngram(tokens: list[str], n: int):
    counts = Counter()
    ctx = Counter()
    alph = set(["^", "$"])
    for w in tokens:
        seq = "^"*(n-1) + w + "$"
        for ch in seq:
            alph.add(ch)
        for i in range(n-1, len(seq)):
            ng = tuple(seq[i-(n-1):i+1])
            counts[ng] += 1
            ctx[ng[:-1]] += 1
    return counts, ctx, alph, n

def score_char_ngram_bits_per_char(tokens: list[str], model):
    counts, ctx, alph, n = model
    V = len(alph)
    logp = 0.0
    chars = 0
    for w in tokens:
        seq = "^"*(n-1) + w + "$"
        for i in range(n-1, len(seq)):
            ng = tuple(seq[i-(n-1):i+1])
            p = (counts.get(ng, 0) + K) / (ctx.get(ng[:-1], 0) + K*V)
            logp += -math.log2(p)
            if ng[-1] != "$":
                chars += 1
    bits = (logp / chars) if chars else float("nan")
    ppl = (2**bits) if chars else float("nan")
    return bits, ppl

# ----------------------------
# Metrics: positional bigram bits (init/med/final)
# ----------------------------
def positional_bigram_bits(train_tokens: list[str], test_tokens: list[str]):
    # Model per zone: init/med/fin, where zone is based on bigram position i
    counts = {z: Counter() for z in ("init", "med", "fin")}
    ctx = {z: Counter() for z in ("init", "med", "fin")}
    V = 26

    for w in train_tokens:
        L = len(w)
        for i in range(L - 1):
            z = "init" if i == 0 else ("fin" if i == L - 2 else "med")
            prev, nxt = w[i], w[i+1]
            counts[z][(prev, nxt)] += 1
            ctx[z][prev] += 1

    out = {}
    for z in ("init", "med", "fin"):
        logp = 0.0
        steps = 0
        for w in test_tokens:
            L = len(w)
            for i in range(L - 1):
                zz = "init" if i == 0 else ("fin" if i == L - 2 else "med")
                if zz != z:
                    continue
                prev, nxt = w[i], w[i+1]
                p = (counts[z].get((prev, nxt), 0) + K) / (ctx[z].get(prev, 0) + K*V)
                logp += -math.log2(p)
                steps += 1
        out[z] = (logp / steps) if steps else float("nan")

    return out

# ----------------------------
# Metrics: edit distance <=1 proxy (dist0/dist1)
# ----------------------------
def build_dist1_index(train_types: set[str]):
    train_set = set(train_types)
    delmap = defaultdict(set)  # deleted_string -> set(lengths of originals)
    for w in train_set:
        L = len(w)
        for i in range(L):
            delmap[w[:i] + w[i+1:]].add(L)
    return train_set, delmap

def has_dist1(w: str, train_set: set[str], delmap) -> bool:
    L = len(w)

    # insertion: a training word of length L+1 deletes to w
    if (w in delmap) and ((L + 1) in delmap[w]):
        return True

    for i in range(L):
        d = w[:i] + w[i+1:]

        # deletion from w matches a training word
        if d in train_set:
            return True

        # substitution: w delete-one equals train delete-one (same length)
        if (d in delmap) and (L in delmap[d]):
            return True

    return False

def dist01_rates(train_tokens: list[str], test_tokens: list[str]):
    train_types = set(train_tokens)
    train_set, delmap = build_dist1_index(train_types)

    test_counts = Counter(test_tokens)
    total_tok = sum(test_counts.values())
    total_types = len(test_counts)

    d0_tok = d1_tok = 0
    d0_type = d1_type = 0

    for w, c in test_counts.items():
        if w in train_set:
            d0_tok += c
            d0_type += 1
        elif has_dist1(w, train_set, delmap):
            d1_tok += c
            d1_type += 1

    if total_tok == 0 or total_types == 0:
        return {}

    return {
        "dist0_tok_pct": d0_tok / total_tok * 100.0,
        "dist1_tok_pct": d1_tok / total_tok * 100.0,
        "le1_tok_pct": (d0_tok + d1_tok) / total_tok * 100.0,
        "le1_type_pct": (d0_type + d1_type) / total_types * 100.0,
    }

# ----------------------------
# positional legality: onset / interior / coda
# ----------------------------
def train_onset_interior_coda(train_tokens: list[str]):
    onsets = set()
    interior = set()
    codas = set()
    for w in train_tokens:
        L = len(w)
        if L >= 2:
            onsets.add(w[:2])
            codas.add(w[-2:])
        # interior: positions 1..L-3 (bigram start index)
        for i in range(1, L - 2):
            interior.add(w[i:i+2])
    return onsets, interior, codas

def reject_onset_interior_coda(w: str, sets) -> bool:
    onsets, interior, codas = sets
    L = len(w)
    if L < 2:
        return True
    if w[:2] not in onsets:
        return True
    if w[-2:] not in codas:
        return True
    for i in range(1, L - 2):
        if w[i:i+2] not in interior:
            return True
    return False

def onset_interior_coda_reject_rate(train_tokens: list[str], test_tokens: list[str]) -> float:
    sets = train_onset_interior_coda(train_tokens)
    rejects = sum(1 for w in test_tokens if reject_onset_interior_coda(w, sets))
    return rejects / len(test_tokens) * 100.0 if test_tokens else float("nan")

# ----------------------------
# Extended distance-from-edge legality
# Zones: onset, i1, mid, i2, coda
# ----------------------------
def train_edge_sets(train_tokens: list[str]):
    onsets = set()
    i1 = set()
    mid = set()
    i2 = set()
    codas = set()

    for w in train_tokens:
        L = len(w)
        if L < 2:
            continue
        onsets.add(w[:2])
        codas.add(w[-2:])

        if L >= 3:
            i1.add(w[1:3])
            i2.add(w[-3:-1])

        if L >= 6:
            for i in range(2, L - 3):  # 2..L-4 inclusive
                mid.add(w[i:i+2])

    return onsets, i1, mid, i2, codas

def reject_edge_word(w: str, sets, include_short: bool, exclude_short: bool) -> bool:
    onsets, i1, mid, i2, codas = sets
    L = len(w)

    if L < 2:
        return True

    if exclude_short and L < 4:
        return False  # excluded from evaluation in that mode

    # onset/coda always
    if w[:2] not in onsets:
        return True
    if w[-2:] not in codas:
        return True

    # interior zones only if present (include_short mode)
    if L >= 3:
        seg1 = w[1:3]
        seg2 = w[-3:-1]
        if seg1 not in i1:
            return True
        if seg2 not in i2:
            return True

    if L >= 6:
        for i in range(2, L - 3):
            if w[i:i+2] not in mid:
                return True

    return False

def edge_reject_rate(train_tokens: list[str], test_tokens: list[str],
                     exclude_short: bool,
                     symmetric_training: bool = True) -> float:
    # symmetric training is reviewer-proof; keep it True by default
    if exclude_short and symmetric_training:
        train_tokens = [w for w in train_tokens if len(w) >= 4]
        test_scoped = [w for w in test_tokens if len(w) >= 4]
    elif exclude_short:
        test_scoped = [w for w in test_tokens if len(w) >= 4]
    else:
        test_scoped = list(test_tokens)

    sets = train_edge_sets(train_tokens)

    if not test_scoped:
        return float("nan")

    rejects = 0
    for w in test_scoped:
        if reject_edge_word(w, sets, include_short=not exclude_short, exclude_short=exclude_short):
            rejects += 1

    return rejects / len(test_scoped) * 100.0

# ----------------------------
# Reporting helpers
# ----------------------------
def md_table(headers, rows) -> str:
    line1 = "| " + " | ".join(headers) + " |"
    line2 = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join([line1, line2] + body)

def write_csv(prefix: str, name: str, headers, rows):
    out = Path(f"{prefix}_{name}.csv")
    with out.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    return str(out)

# ----------------------------
# Section presets
# ----------------------------
PRESETS = {
    "recipe": (103, 116),
    "herbal": (1, 66),
    "between": (67, 102),  # your “crazy” combined block between herbal and recipe
    "all": (None, None),
}

# ----------------------------
# Strip Gutenberg Headers
# ----------------------------

def strip_gutenberg(text: str) -> str:
    lines = text.splitlines()
    start = None
    end = None

    for i, line in enumerate(lines):
        if "START OF THE PROJECT GUTENBERG EBOOK" in line:
            start = i + 1
        elif "END OF THE PROJECT GUTENBERG EBOOK" in line:
            end = i
            break

    if start is None or end is None or start >= end:
        raise ValueError("Gutenberg START/END markers not found")

    return "\n".join(lines[start:end])

def load_tokens_from_text_file(path: str, min_len: int) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    core = strip_gutenberg(raw)
    return tokenize(core, min_len=min_len)

# ----------------------------
# Shuffle Test
# ----------------------------

def randomize_tokens_within_word(tokens: list[str], seed: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    for w in tokens:
        if len(w) <= 1:
            out.append(w)
            continue
        chars = list(w)
        rng.shuffle(chars)
        out.append("".join(chars))
    return out

def randomize_tokens_global(tokens: list[str], seed: int) -> list[str]:
    rng = random.Random(seed)
    lengths = [len(w) for w in tokens]
    pool = [ch for w in tokens for ch in w]
    rng.shuffle(pool)

    out = []
    idx = 0
    for L in lengths:
        out.append("".join(pool[idx:idx+L]))
        idx += L
    return out

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Voynich parser-locked analysis suite (LSI + iParser)")
    ap.add_argument("--lsi", default="LSI.txt", help="Path to LSI.txt")
    ap.add_argument("--iparser_dir", default=".", help="Directory containing iParser_guarded.py")
    ap.add_argument("--transcriber", default="H", help="Transcriber layer (default H)")

    ap.add_argument("--preset", choices=sorted(PRESETS.keys()), default="recipe",
                    help="Section preset: recipe/herbal/between/all")
    ap.add_argument("--start", type=int, default=None, help="Custom start folio (overrides preset)")
    ap.add_argument("--end", type=int, default=None, help="Custom end folio (overrides preset)")

    ap.add_argument("--train_sides", choices=["r", "v", "both"], default="r",
                    help="Training side(s): r, v, or both")
    ap.add_argument("--test_sides", choices=["r", "v", "both"], default="v",
                    help="Testing side(s): r, v, or both")

    ap.add_argument("--min_len", type=int, default=2, help="Minimum token length (default 2)")
    ap.add_argument("--csv_prefix", default=None, help="If set, write tables to CSV with this prefix")
    ap.add_argument(
        "--symmetric_short_filter",
        choices=["true", "false"],
        default="true",
        help="When exclude_short=True (edge-zone test): "
             "true = apply <4 filter to BOTH train and test; "
             "false = apply <4 filter to TEST only."
    )
    ap.add_argument(
        "--corpus",
        choices=["voynich", "text"],
        default="voynich",
        help="Input corpus type: voynich (LSI+iParser) or text (plain text file)"
    )
    ap.add_argument(
        "--text_file",
        default=None,
        help="Path to plain text file (required if --corpus text)"
    )
    ap.add_argument(
        "--randomize",
        choices=["none", "within_word", "global"],
        default="none",
        help="Randomization control applied to TEST tokens only (default none)."
    )
    ap.add_argument(
        "--rand_seed",
        type=int,
        default=12345,
        help="RNG seed for randomization control."
    )
    args = ap.parse_args()

    sym = (args.symmetric_short_filter == "true")

    # ----------------------------
    # INGESTION
    # ----------------------------
    if args.corpus == "text":
        if not args.text_file:
            raise ValueError("--text_file is required when --corpus text")

        train_f = []
        test_f = []
        start = None
        end = None

        all_tokens = load_tokens_from_text_file(args.text_file, args.min_len)
        split = len(all_tokens) // 2
        train_tokens = all_tokens[:split]
        test_tokens  = all_tokens[split:]

    else:
        # Voynich path (original behavior)
        ip = import_iparser(args.iparser_dir)

        all_ids = ip.list_folios(args.lsi, transcriber=args.transcriber)

        canon_r = [f for f in all_ids if f.endswith("r")]
        canon_v = [f for f in all_ids if f.endswith("v")]
        sub_r   = [f for f in all_ids if ("r" in f and not f.endswith("r") and f.rstrip("0123456789").endswith("r"))]
        sub_v   = [f for f in all_ids if ("v" in f and not f.endswith("v") and f.rstrip("0123456789").endswith("v"))]

        print("\n=== Corpus inventory (iParser folio IDs) ===")
        print(f"Total IDs returned by iParser: {len(all_ids)}")
        print(f"Canonical rectos (end with 'r'): {len(canon_r)}")
        print(f"Canonical versos (end with 'v'): {len(canon_v)}")
        print(f"Sub-pages like r2/r3/...: {len(sub_r)}")
        print(f"Sub-pages like v2/v3/...: {len(sub_v)}")

        preset_start, preset_end = PRESETS[args.preset]
        start = args.start if args.start is not None else preset_start
        end = args.end if args.end is not None else preset_end

        train_sides = args.train_sides
        test_sides = args.test_sides

        if train_sides == "both":
            train_f = select_folios(ip, args.lsi, args.transcriber, start, end, "r") + \
                      select_folios(ip, args.lsi, args.transcriber, start, end, "v")
        else:
            train_f = select_folios(ip, args.lsi, args.transcriber, start, end, train_sides)

        if test_sides == "both":
            test_f = select_folios(ip, args.lsi, args.transcriber, start, end, "r") + \
                     select_folios(ip, args.lsi, args.transcriber, start, end, "v")
        else:
            test_f = select_folios(ip, args.lsi, args.transcriber, start, end, test_sides)

        train_tokens = load_tokens_for_folios(ip, args.lsi, args.transcriber, train_f, args.min_len)
        test_tokens  = load_tokens_for_folios(ip, args.lsi, args.transcriber, test_f, args.min_len)

    # ----------------------------
    # RANDOMIZATION CONTROL (test-only)
    # ----------------------------
    if args.randomize != "none":
        if args.randomize == "within_word":
            test_tokens = randomize_tokens_within_word(test_tokens, args.rand_seed)
        elif args.randomize == "global":
            test_tokens = randomize_tokens_global(test_tokens, args.rand_seed)

    # ----------------------------
    # COMPUTE SUITE (unchanged)
    # ----------------------------
    anywhere_rej = bigram_legality_reject_rate(train_tokens, test_tokens)

    bi_model = train_char_ngram(train_tokens, n=2)
    tri_model = train_char_ngram(train_tokens, n=3)

    bi_bits, bi_ppl = score_char_ngram_bits_per_char(test_tokens, bi_model)
    tri_bits, tri_ppl = score_char_ngram_bits_per_char(test_tokens, tri_model)
    delta = bi_bits - tri_bits

    pos = positional_bigram_bits(train_tokens, test_tokens)
    d01 = dist01_rates(train_tokens, test_tokens)

    claude_rej = onset_interior_coda_reject_rate(train_tokens, test_tokens)

    edge_excl = edge_reject_rate(train_tokens, test_tokens, exclude_short=True,  symmetric_training=sym)
    edge_incl = edge_reject_rate(train_tokens, test_tokens, exclude_short=False, symmetric_training=True)

    # ----------------------------
    # REPORT
    # ----------------------------
    print("\n=== Voynich Suite (parser-locked) ===")

    if args.corpus == "text":
        print(f"Corpus: TEXT file={args.text_file}")
    else:
        print(f"Range: {start}–{end}  Preset: {args.preset}")
        print(f"Train sides: {args.train_sides}  Test sides: {args.test_sides}")
        print(f"Train folios: {len(train_f)}  Test folios: {len(test_f)}")

    print(f"Train tokens: {len(train_tokens)}  Test tokens: {len(test_tokens)}")
    print(f"Min token length: {args.min_len}")

    headers = ["Metric", "Value"]
    rows = [
        ("Bigram legality reject% (anywhere)", f"{anywhere_rej:.3f}"),
        ("Onset/interior/coda reject%", f"{claude_rej:.3f}"),
        ("Edge-zones reject% (exclude <4, symmetric_short_filter=" + str(sym) + ")", f"{edge_excl:.3f}"),
        ("Edge-zones reject% (include <4, symmetric)", f"{edge_incl:.3f}"),
        ("Bigram bits/char", f"{bi_bits:.4f}"),
        ("Bigram perplexity", f"{bi_ppl:.4f}"),
        ("Trigram bits/char", f"{tri_bits:.4f}"),
        ("Trigram perplexity", f"{tri_ppl:.4f}"),
        ("Δ bits (Bigram − Trigram)", f"{delta:.4f}"),
        ("Positional bits (init)", f"{pos['init']:.4f}"),
        ("Positional bits (med)", f"{pos['med']:.4f}"),
        ("Positional bits (fin)", f"{pos['fin']:.4f}"),
        ("Dist0 token %", f"{d01.get('dist0_tok_pct', float('nan')):.2f}"),
        ("Dist1 token %", f"{d01.get('dist1_tok_pct', float('nan')):.2f}"),
        ("<=1 token %", f"{d01.get('le1_tok_pct', float('nan')):.2f}"),
        ("<=1 type %", f"{d01.get('le1_type_pct', float('nan')):.2f}"),
    ]
    print("\n" + md_table(headers, rows))

    if args.csv_prefix:
        out = write_csv(args.csv_prefix, "summary", headers, rows)
        print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
