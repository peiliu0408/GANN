#!/usr/bin/env bash
# ============================================================================
# One-command testing: evaluate a checkpoint on a split (default: test).
#
# Usage:
#   bash test.sh ckpt                            # test split, best recipe (greedy)
#   bash test.sh ckpt test beam 5 1.5            # custom beam / length penalty
#
# Best decoding recipe for CIDEr-D is GREEDY (SCST optimizes greedy reward);
# beam search only helps BLEU-1. Defaults below follow that finding.
# ============================================================================
set -e
cd "$(dirname "$0")"
CKPT="${1:-ckpt}"
SPLIT="${2:-test}"
METHOD="${3:-greedy}"
BEAM="${4:-2}"
LENPEN="${5:-4.0}"

if [ "$METHOD" = "beam" ]; then
    python3 scripts/eval.py --checkpoint "$CKPT" --eval_splits "$SPLIT" \
        --sample_method beam --beam_size "$BEAM" --length_penalty "$LENPEN"
else
    python3 scripts/eval.py --checkpoint "$CKPT" --eval_splits "$SPLIT" \
        --sample_method greedy
fi
