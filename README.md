# Voynich Structural Analysis Suite

A deterministic, character-level analysis suite for testing **structural learnability**
in the Voynich Manuscript and control corpora.

This project is designed for **academic use, reproducibility, and peer review**.
It performs **no decipherment**, uses **no neural or AI models**, and makes
**no semantic assumptions**.

---

## Purpose

This suite addresses a focused research question:

Does the Voynich Manuscript exhibit learnable character-level structure
beyond what is expected from shuffled or synthetic controls?

The suite evaluates this by comparing train/test splits of the corpus and
measuring structural degradation under controlled randomization.

---

## Design Principles

- Deterministic (identical input → identical output)
- Leakage-free (folio-based or 50/50 splits only)
- Parser-locked (only attested Voynich lines)
- Character-level only
- Reviewer-friendly (single script, copy/paste CLI)

---

## Supported Corpora

### Voynich Manuscript
- LSI transcription (IVTFF-aware)
- Transcriber-selectable (default: H)
- Section presets:
  - herbal (folios 1–66)
  - between (folios 67–102)
  - recipe (folios 103–116)
  - all

### Plain Text Controls
- Project Gutenberg texts
- Gutenberg headers stripped automatically
- Corpus split 50/50 for train/test

---

## Metrics

Structural Legality
- Bigram legality reject rate
- Onset / interior / coda legality
- Multi-zone edge legality (short-token controlled)

Entropy & Predictability
- Bigram bits per character
- Trigram bits per character
- Δ bits (Bigram − Trigram)
- Positional entropy (init / medial / final)

Continuity
- Exact-match tokens (edit distance 0)
- Edit distance ≤1 (tokens and types)

Negative Controls
- Within-word shuffle (test-only)
- Global character shuffle (test-only)

---

## Requirements

- Python 3.10+
- Standard library only

---

## Installation

git clone https://github.com/YOUR_USERNAME/voynich-structural-suite.git  
cd voynich-structural-suite  

Place LSI.txt in the working directory.

---

## Usage (copy/paste examples)

Recipe section (folios 103–116), rectos → versos  
python vw_suite.py --preset recipe --train_sides r --test_sides v

Herbal section (folios 1–66), rectos → versos  
python vw_suite.py --preset herbal --train_sides r --test_sides v

Reverse sanity check (train versos → test rectos)  
python vw_suite.py --preset recipe --train_sides v --test_sides r

Whole corpus  
python vw_suite.py --preset all --train_sides r --test_sides v

Custom folio range  
python vw_suite.py --start 90 --end 116 --train_sides r --test_sides v

Set smoothing parameter  
python vw_suite.py --preset recipe --train_sides r --test_sides v --K 1.0

Shuffle TEST tokens only (within-word)  
python vw_suite.py --preset recipe --train_sides r --test_sides v --randomize within_word --rand_seed 1

Shuffle TEST tokens only (global)  
python vw_suite.py --preset recipe --train_sides r --test_sides v --randomize global --rand_seed 1

Dump unseen bigram counts  
python vw_suite.py --preset recipe --train_sides r --test_sides v --dump_unseen_bigrams out/unseen_bigrams.csv

Run on Project Gutenberg text  
python vw_suite.py --corpus text --text_file pg11940.txt --min_len 2 --K 0.1

---

## Output

Results are printed as a Markdown-formatted table suitable for:
- Papers
- Preprints
- Supplementary materials

---

## Interpretation Notes

- Lower bits/char = higher predictability
- Higher reject rates = structural breakdown
- Shuffle tests should strongly degrade all metrics
- Δ(Bigram − Trigram) is a comparative diagnostic, not a structure score

This suite does not claim decipherment.

---

## Status

- Canonical, paper-ready version
- Actively used in Voynich structural analysis
- Backward-compatible CLI expected

---

## License

MIT License
