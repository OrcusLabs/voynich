#!/usr/bin/env python3
"""
vw_suite_allinone_paper_v20.py — Voynich Structural Analysis Suite (inline parser)

Deterministic, leakage-free character-level structural tests for the Voynich
Manuscript (LSI transcription) and control corpora.

- Inline IVTFF tag-aware parser (no iParser dependency)
- Embedded exclusion list for physically missing folios (from iParser_guarded)
- Numeric folio sorting (supports suffix digits like f101r2)
- Train/test split is folio-based by side(s)

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


# ----------------------------
# Usage (copy/paste examples)
# ----------------------------
#
# Recipe section (folios 103–116), rectos → versos
# python vw_suite.py --preset recipe --train_sides r --test_sides v
#
# Herbal section (folios 1–66), rectos → versos
# python vw_suite_v20.py --preset herbal --train_sides r --test_sides v
#
# Reverse sanity check: train on versos, test on rectos
# python vw_suite_v20.py --preset recipe --train_sides v --test_sides r
#
# Whole corpus, train on rectos, test on versos
# python vw_suite_v20.py --preset all --train_sides r --test_sides v
#
# Custom folio range (inclusive), rectos → versos
# python vw_suite_v20.py --start 90 --end 116 --train_sides r --test_sides v
#
# Set smoothing parameter K (example: K=1.0)
# python vw_suite_v20.py --preset recipe --train_sides r --test_sides v --K 1.0
#
# Shuffle TEST tokens only (within-word shuffle)
# python vw_suite_v20.py --preset recipe --train_sides r --test_sides v --randomize within_tokens --rand_seed 1
#
# Shuffle TEST tokens only (global character shuffle; preserves token lengths)
# python vw_suite_v20.py --preset recipe --train_sides r --test_sides v --randomize global_chars --rand_seed 1
#
# Dump unseen bigram counts (CSV)
# python vw_suite_v20.py --preset recipe --train_sides r --test_sides v --dump_unseen_bigrams out/unseen_bigrams.csv
#
# Run on a Project Gutenberg text (auto-strips headers, splits 50/50)
# python vw_suite_v20.py --corpus text --text_file pg11940.txt --min_len 2 --K 0.1


# ============================================================
# NORMALIZATION + TOKENIZATION
# ============================================================

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

# ============================================================
# EXCLUSION LIST (physically missing folios)
# (Embedded from iParser_guarded)
# ============================================================

EXCLUDED_FOLIOS = {
    'f12r', 'f12v', 'f59r', 'f59v', 'f60r', 'f60v', 'f61r', 'f61v', 'f62r', 'f62v', 'f63r', 'f63v', 'f64r', 'f64v', 'f101r2', 'f109r', 'f109v', 'f110r', 'f110v', 'f116v'
}

# ============================================================
# LSI PARSER (INLINE, IVTFF-AWARE)
# ============================================================

# Base folio id: f<num><side><optional_suffix_digits>
_FOLIO_ID_RE = re.compile(r"^f(\d+)(r|v)(\d*)$", re.IGNORECASE)

# IVTFF transcription line:
#   <f103r.1,@P0;H> payload...
# Capture: num, side, suffix, transcriber, payload
_LSI_LINE_RE = re.compile(r"^<f(\d+)(r|v)(\d*)\.[^;]*;([A-Za-z])>\s*(.*)$", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")

def folio_num(fid: str) -> int:
    m = _FOLIO_ID_RE.match(fid.strip())
    if not m:
        raise ValueError(f"Malformed folio ID: {fid}")
    return int(m.group(1))

def folio_side(fid: str) -> str:
    m = _FOLIO_ID_RE.match(fid.strip())
    if not m:
        raise ValueError(f"Malformed folio ID: {fid}")
    return m.group(2).lower()

def folio_suffix(fid: str) -> int:
    m = _FOLIO_ID_RE.match(fid.strip())
    if not m:
        raise ValueError(f"Malformed folio ID: {fid}")
    sfx = m.group(3)
    return int(sfx) if sfx else 0

def folio_sort_key(fid: str):
    return (folio_num(fid), folio_side(fid), folio_suffix(fid))

def parse_lsi_lines(lsi_path: str, transcriber: str = "H"):
    """
    Yields (folio_id, payload) for the specified transcriber.
    Skips excluded folios.
    """
    text = Path(lsi_path).read_text(encoding="utf-8", errors="replace")
    found = False
    tr = (transcriber or "").strip()
    if len(tr) != 1:
        raise ValueError("--transcriber must be a single character (e.g., H, C, F, N, U)")
    tr = tr.upper()

    for ln, line in enumerate(text.splitlines(), start=1):
        if not line.startswith("<f"):
            continue
        m = _LSI_LINE_RE.match(line)
        if not m:
            continue
        num, side, sfx, trc, payload = m.groups()
        if trc.upper() != tr:
            continue
        fid = f"f{int(num)}{side.lower()}{sfx}"
        if fid.lower() in EXCLUDED_FOLIOS:
            continue
        payload = _TAG_RE.sub(" ", payload)
        if not payload.strip():
            print(f"WARNING: empty payload on line {ln}", file=sys.stderr)
            continue
        found = True
        yield fid, payload

    if not found:
        raise ValueError(f"No valid transcription lines found for transcriber '{tr}' in LSI file")

def list_folios(lsi_path: str, transcriber: str = "H"):
    fols = set()
    for fid, _payload in parse_lsi_lines(lsi_path, transcriber=transcriber):
        fols.add(fid)
    out = sorted(fols, key=folio_sort_key)
    return out

def select_folios(lsi_path, start, end, sides, transcriber: str = "H"):
    allowed = set((sides or "").lower())
    if not allowed or not allowed.issubset({"r", "v"}):
        raise ValueError("--train_sides/--test_sides must be one of: r, v, rv")
    out = []
    for fid in list_folios(lsi_path, transcriber=transcriber):
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

def load_voynich_tokens(lsi_path, folios, min_len, transcriber: str = "H"):
    tokens = []
    folio_set = set(folios)
    for fid, payload in parse_lsi_lines(lsi_path, transcriber=transcriber):
        if fid in folio_set:
            tokens.extend(tokenize(payload, min_len))
    return tokens

# ============================================================
# GUTENBERG HANDLING
# ============================================================

def strip_gutenberg(text: str) -> str:
    lines = text.splitlines()
    start = end = None
    for i, l in enumerate(lines):
        if "START OF THE PROJECT GUTENBERG EBOOK" in l:
            start = i + 1
        if "END OF THE PROJECT GUTENBERG EBOOK" in l:
            end = i
            break
    if start is None or end is None:
        raise ValueError("Gutenberg START/END markers not found")
    return "\n".join(lines[start:end])

def load_text_tokens(path, min_len):
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    core = strip_gutenberg(raw)
    return tokenize(core, min_len)

# ============================================================
# METRICS
# ============================================================

def bigram_legality_reject_rate(train_tokens, test_tokens):
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

def unseen_bigram_counts(train_tokens, test_tokens):
    seen = set()
    for w in train_tokens:
        for i in range(len(w) - 1):
            seen.add(w[i:i+2])
    c = Counter()
    for w in test_tokens:
        for i in range(len(w) - 1):
            bg = w[i:i+2]
            if bg not in seen:
                c[bg] += 1
    return c

def train_ngram(tokens, n):
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

def score_ngram(tokens, model, K):
    c, ctx, alph, n = model
    V = len(alph)
    lp = chars = 0
    for w in tokens:
        seq = "^"*(n-1) + w + "$"
        for i in range(n-1, len(seq)):
            ng = tuple(seq[i-(n-1):i+1])
            p = (c.get(ng, 0) + K) / (ctx.get(ng[:-1], 0) + K*V)
            lp += -math.log2(p)
            if ng[-1] != "$":
                chars += 1
    if chars == 0:
        return float("nan"), float("nan")
    bpc = lp / chars
    return bpc, 2 ** bpc

def positional_bits(train, test, K):
    counts = {z: Counter() for z in ("init", "med", "fin")}
    ctx = {z: Counter() for z in ("init", "med", "fin")}
    for w in train:
        L = len(w)
        for i in range(L - 1):
            z = "init" if i == 0 else "fin" if i == L - 2 else "med"
            counts[z][(w[i], w[i+1])] += 1
            ctx[z][w[i]] += 1
    out = {}
    for z in counts:
        lp = steps = 0
        for w in test:
            L = len(w)
            for i in range(L - 1):
                zz = "init" if i == 0 else "fin" if i == L - 2 else "med"
                if zz != z:
                    continue
                p = (counts[z].get((w[i], w[i+1]), 0) + K) / (ctx[z].get(w[i], 0) + K*26)
                lp += -math.log2(p)
                steps += 1
        out[z] = lp / steps if steps else float("nan")
    return out

# ============================================================
# LEVENSHTEIN ≤ 1 (EXACT)
# ============================================================

def _deletion_signatures(w: str):
    for i in range(len(w)):
        yield w[:i] + w[i+1:]

def build_edit1_index(train_types):
    """
    Build indices for edit distance <= 1 checks:
      - deletion signatures (for insertion/deletion)
      - substitution signatures (for Hamming-1 substitutions)
    """
    sig2lens = defaultdict(set)
    subst_sigs = set()

    for t in train_types:
        L = len(t)

        # deletion signatures for insertion/deletion
        for sig in _deletion_signatures(t):
            sig2lens[sig].add(L)

        # substitution signatures: replace one position with '*'
        # e.g. "abcd" -> "*bcd", "a*cd", "ab*d", "abc*"
        for i in range(L):
            subst_sigs.add(t[:i] + "*" + t[i+1:])

    return train_types, sig2lens, subst_sigs


def has_levenshtein_le1(w, train_set, sig2lens, subst_sigs):
    # Exact match
    if w in train_set:
        return True

    L = len(w)

    # Substitution (same length, Hamming distance 1) via signature lookup
    # w is within 1 substitution of some train type iff any wildcard signature matches
    for i in range(L):
        if (w[:i] + "*" + w[i+1:]) in subst_sigs:
            return True

    # Insertion/deletion (via deletion signatures)
    if (L + 1) in sig2lens.get(w, ()):
        return True
    for sig in _deletion_signatures(w):
        if sig in train_set:
            return True
        if L in sig2lens.get(sig, ()):
            return True

    return False


def dist01(train, test):
    train_types = set(train)
    train_set, sig2lens, subst_sigs = build_edit1_index(train_types)
    cnt = Counter(test)
    tot_tok = sum(cnt.values())
    tot_type = len(cnt)
    if tot_tok == 0 or tot_type == 0:
        return dict(d0_tok=float("nan"), d1_tok=float("nan"),
                    le1_tok=float("nan"), le1_type=float("nan"))
    d0_tok = d1_tok = d0_type = d1_type = 0
    for w, c in cnt.items():
        if w in train_set:
            d0_tok += c
            d0_type += 1
        elif has_levenshtein_le1(w, train_set, sig2lens, subst_sigs):
            d1_tok += c
            d1_type += 1
    return {
        "d0_tok": d0_tok / tot_tok * 100,
        "d1_tok": d1_tok / tot_tok * 100,
        "le1_tok": (d0_tok + d1_tok) / tot_tok * 100,
        "le1_type": (d0_type + d1_type) / tot_type * 100,
    }

# ============================================================
# SHUFFLE CONTROLS
# ============================================================

def shuffle_within(tokens, rng):
    out = []
    for w in tokens:
        ch = list(w)
        rng.shuffle(ch)
        out.append("".join(ch))
    return out

def shuffle_global(tokens, rng):
    pool = [c for w in tokens for c in w]
    rng.shuffle(pool)
    out = []
    k = 0
    for w in tokens:
        out.append("".join(pool[k:k+len(w)]))
        k += len(w)
    return out

# -------------------------
# CRILS (Conditional Repairability of Illicit Local Structures)
# -------------------------

def _train_bigram_set(train_tokens):
    seen = set()
    for w in train_tokens:
        for i in range(len(w) - 1):
            seen.add(w[i:i+2])
    return seen

def _token_illicit_by_bigram(w: str, seen_bigrams: set[str]) -> bool:
    # Illicit if ANY internal bigram is unseen in train
    for i in range(len(w) - 1):
        if w[i:i+2] not in seen_bigrams:
            return True
    return False

def _unseen_bigram_instances(w: str, seen_bigrams: set[str]) -> int:
    c = 0
    for i in range(len(w) - 1):
        if w[i:i+2] not in seen_bigrams:
            c += 1
    return c

def _count_window_violations(tokens, seen_bigrams, i, window):
    lo = max(0, i - window)
    hi = min(len(tokens) - 1, i + window)
    v = 0
    for j in range(lo, hi + 1):
        v += _unseen_bigram_instances(tokens[j], seen_bigrams)
    return v

def _mutate_del(w: str, rng):
    if len(w) <= 1:
        return None
    i = rng.randrange(len(w))
    return w[:i] + w[i+1:]

def _mutate_swap(w: str, rng):
    if len(w) <= 1:
        return None
    candidates = [i for i in range(len(w)-1) if w[i] != w[i+1]]
    if not candidates:
        return None
    i = rng.choice(candidates)
    return w[:i] + w[i+1] + w[i] + w[i+2:]


def run_crils(train_tokens, test_tokens, *, ops=("del", "swap"), samples=2000, window=2, seed=1):
    """
    CRILS (v2): evaluate ALL ops for EACH illicit token.
    Returns: summary_dict, rows(list of dicts)
    """
    import random
    rng = random.Random(seed)

    seen_bi = _train_bigram_set(train_tokens)

    illicit_idxs = [i for i, w in enumerate(test_tokens) if _token_illicit_by_bigram(w, seen_bi)]
    if not illicit_idxs:
        return (
            dict(
                N_illicit=0,
                N_eval=0,
                mean_delta=float("nan"),
                pct_pos=float("nan"),
                pct_zero=float("nan"),
                pct_neg=float("nan"),
            ),
            []
        )

    if samples is not None and samples > 0 and len(illicit_idxs) > samples:
        illicit_idxs = rng.sample(illicit_idxs, samples)

    ops = tuple(o.strip() for o in ops if o.strip())
    if not ops:
        ops = ("del",)

    rows = []
    deltas = []

    for i in illicit_idxs:
        before = _count_window_violations(test_tokens, seen_bi, i, window)
        w0 = test_tokens[i]

        for op in ops:
            if op == "del":
                w1 = _mutate_del(w0, rng)
            elif op == "swap":
                w1 = _mutate_swap(w0, rng)
            else:
                w1 = None

            if not w1:
                continue

            tmp = list(test_tokens)
            tmp[i] = w1
            after = _count_window_violations(tmp, seen_bi, i, window)

            delta = before - after  # positive = repaired (fewer violations)
            deltas.append(delta)

            rows.append({
                "token_index": i,
                "op": op,
                "token_before": w0,
                "token_after": w1,
                "len_before": len(w0),
                "len_after": len(w1),
                "before_violations": before,
                "after_violations": after,
                "delta": delta,
            })

    if not deltas:
        return (
            dict(
                N_illicit=len(illicit_idxs),
                N_eval=0,
                mean_delta=float("nan"),
                pct_pos=float("nan"),
                pct_zero=float("nan"),
                pct_neg=float("nan"),
            ),
            rows
        )

    n = len(deltas)
    pos = sum(1 for d in deltas if d > 0)
    zer = sum(1 for d in deltas if d == 0)
    neg = sum(1 for d in deltas if d < 0)
    mean_delta = sum(deltas) / n

    summary = dict(
        N_illicit=len(illicit_idxs),  # number of illicit TOKENS sampled
        N_eval=n,                     # number of evaluated (token,op) pairs
        mean_delta=mean_delta,
        pct_pos=pos / n * 100,
        pct_zero=zer / n * 100,
        pct_neg=neg / n * 100,
    )
    return summary, rows

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
    ap.add_argument(
        "--randomize",
        choices=["none", "within_tokens", "global_chars"],
        default="none",
        help="Randomization control applied to TEST tokens only (default none)."
    )

    ap.add_argument(
        "--rand_seed",
        type=int,
        default=1
    )
    ap.add_argument("--dump_unseen_bigrams", default=None)
    ap.add_argument("--max_unseen_bigrams", type=int, default=5000)
    ap.add_argument("--K", type=float, default=0.1)
    
# -------------------------
# CRILS (Conditional Repairability of Illicit Local Structures)
# -------------------------
    ap.add_argument("--crils", action="store_true",
                    help="Run CRILS: local edit-distance-1 perturbations on illicit tokens and score repairability.")
    ap.add_argument("--crils_ops", default="del,swap",
                    help="Comma list: del,swap. (Boundary ops intentionally omitted for v1.)")
    ap.add_argument("--crils_samples", type=int, default=2000,
                    help="Illicit TEST tokens to sample (0 = ALL illicit; >0 = random sample).")
    ap.add_argument("--crils_window", type=int, default=2,
                    help="Context window size (tokens left/right) used when scoring violation deltas.")
    ap.add_argument("--crils_out_csv", default=None,
                    help="Optional CSV path to write per-sample CRILS rows.")
    args = ap.parse_args()

    K = args.K

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
        if args.preset not in PRESETS and args.start is None and args.end is None:
            raise ValueError(f"Unknown --preset '{args.preset}'. Use one of: {', '.join(PRESETS.keys())} or provide --start/--end.")

        start = args.start if args.start is not None else s
        end = args.end if args.end is not None else e
        train_f = select_folios(args.lsi, start, end, args.train_sides, args.transcriber)
        test_f = select_folios(args.lsi, start, end, args.test_sides, args.transcriber)
        train = load_voynich_tokens(args.lsi, train_f, args.min_len, args.transcriber)
        test = load_voynich_tokens(args.lsi, test_f, args.min_len, args.transcriber)

    if args.randomize != "none":
        import random
        rng = random.Random(args.rand_seed)
        if args.randomize == "within_tokens":
            test = shuffle_within(test, rng)
        else:  # global_chars
            test = shuffle_global(test, rng)

    rej = bigram_legality_reject_rate(train, test)

    bi = train_ngram(train, 2)
    tri = train_ngram(train, 3)
    bi_bpc, _ = score_ngram(test, bi, K)
    tri_bpc, _ = score_ngram(test, tri, K)
    delta = bi_bpc - tri_bpc if not math.isnan(bi_bpc) and not math.isnan(tri_bpc) else float("nan")
    pos = positional_bits(train, test, K)
    d01 = dist01(train, test)

    crils_summary = None
    if args.crils:
        ops = tuple(x.strip() for x in args.crils_ops.split(",") if x.strip())
        crils_summary, crils_rows = run_crils(
            train, test,
            ops=ops,
            samples=args.crils_samples,
            window=args.crils_window,
            seed=args.rand_seed
        )

        if args.crils_out_csv:
            outp = Path(args.crils_out_csv)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with outp.open("w", encoding="utf-8") as f:
                f.write("token_index,op,token_before,token_after,len_before,len_after,before_violations,after_violations,delta\n")
                for r in crils_rows:
                    f.write(f"{r['token_index']},{r['op']},{r['token_before']},{r['token_after']},"
                            f"{r['len_before']},{r['len_after']},{r['before_violations']},{r['after_violations']},{r['delta']}\n")


    def fmt(x, nd=4):
        return "NA" if math.isnan(x) else f"{x:.{nd}f}"

    print("| Metric | Value |")
    print("| --- | --- |")
    print(f"| Bigram legality reject% (anywhere) | {fmt(rej,3)} |")
    print(f"| Bigram bits/char | {fmt(bi_bpc)} |")
    print(f"| Trigram bits/char | {fmt(tri_bpc)} |")
    print(f"| Δ bits (Bigram − Trigram) | {fmt(delta)} |")
    print(f"| Positional bits (init) | {fmt(pos['init'])} |")
    print(f"| Positional bits (med) | {fmt(pos['med'])} |")
    print(f"| Positional bits (fin) | {fmt(pos['fin'])} |")
    print(f"| Levenshtein 0 token % (exact match) | {fmt(d01['d0_tok'],2)} |")
    print(f"| Levenshtein 1 token % (one edit) | {fmt(d01['d1_tok'],2)} |")
    print(f"| Levenshtein ≤1 token % | {fmt(d01['le1_tok'],2)} |")
    print(f"| Levenshtein ≤1 type % | {fmt(d01['le1_type'],2)} |")
    if crils_summary:
        print(f"| CRILS N illicit (sampled pool) | {crils_summary['N_illicit']} |")
        print(f"| CRILS N evaluated (non-noop) | {crils_summary['N_eval']} |")
        print(f"| CRILS mean Δ violations (before−after) | {fmt(crils_summary['mean_delta'],4)} |")
        print(f"| CRILS % Δ>0 (repair) | {fmt(crils_summary['pct_pos'],2)} |")
        print(f"| CRILS % Δ=0 | {fmt(crils_summary['pct_zero'],2)} |")
        print(f"| CRILS % Δ<0 (breakage) | {fmt(crils_summary['pct_neg'],2)} |")

if __name__ == "__main__":
    main()

