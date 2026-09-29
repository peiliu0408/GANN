#!/usr/bin/env bash
# ============================================================================
# One-command training: check data -> stage-1 (XE) -> stage-2 (SCST)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash train.sh        # full two-stage training
#   ARCH=paper SEED=44 bash train.sh            # override hyperparameters
#   RESUME_XE=ckpt/model-best.pth bash train.sh  # skip stage 1
# ============================================================================
set -e
cd "$(dirname "$0")"

# ------------------------------- best recipe (env-overridable)
ARCH="${ARCH:-medium}"            # small | paper | medium
BS="${BS:-64}"                    # batch size per GPU
SEED="${SEED:-43}"                # RNG seed
WARMUP="${WARMUP:-20000}"         # noam warmup steps (stage 1)
LAMBDA="${LAMBDA:-0.2}"           # label-loss weight
XE_EPOCHS="${XE_EPOCHS:-30}"      # stage-1 epoch cap (early stop 5)
SCST_EPOCHS="${SCST_EPOCHS:-25}"  # stage-2 epoch cap (early stop 8)
SCST_LR="${SCST_LR:-2.5e-5}"      # fixed lr for stage 2 (noam disabled)
SAMPLE_N="${SAMPLE_N:-5}"         # sampled branches per step in SCST

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "==> GPU: $CUDA_VISIBLE_DEVICES | arch=$ARCH bs=$BS seed=$SEED lambda=$LAMBDA"

# ------------------------------- data checks
for f in data/captions/dataset_coco.json \
         data/processed/vocab.json \
         data/processed/feat_index_packed.json \
         data/processed/labels.json \
         data/processed/labels_train.npz \
         data/processed/labels_val.npz \
         data/processed/labels_test.npz \
         data/processed/coco-train-idxs.pkl \
         data/features/cocobu_att_packed/data.f32; do
    [ -f "$f" ] || { echo "MISSING: $f -- see readme.txt 'Data layout'"; exit 1; }
done
echo "==> data OK"

COMMON_ARGS=(--karpathy_json data/captions/dataset_coco.json
             --input_json data/processed/vocab.json
             --feature_index data/processed/feat_index_packed.json
             --feature_mode packed
             --labels_json data/processed/labels.json
             --labels_vec data/processed/labels_{split}.npz
             --cider_cache data/processed/coco-train-idxs.pkl
             --batch_size "$BS" --seed "$SEED"
             --arch "$ARCH" --label_loss_weight "$LAMBDA")

# ------------------------------- stage 1: XE
XE_DIR="log_xe_${ARCH}_seed${SEED}"
if [ -n "$RESUME_XE" ]; then
    echo "==> stage 1 skipped (RESUME_XE=$RESUME_XE)"
    XE_BEST="$RESUME_XE"
else
    mkdir -p "$XE_DIR"
    echo "==> stage 1/2: XE training -> $XE_DIR"
    python3 scripts/train.py \
        --id "xe_${ARCH}_seed${SEED}" \
        --phase xe --max_epochs "$XE_EPOCHS" --early_stop_patience 5 \
        --init_scheme bert --label_smoothing 0.1 --weight_tie 1 --ema_decay 0.999 \
        --noamopt 1 --noamopt_warmup "$WARMUP" \
        "${COMMON_ARGS[@]}" 2>&1 | tee "$XE_DIR/train.log"
    XE_BEST="$XE_DIR/model-best.pth"
fi

# ------------------------------- stage 2: SCST
SCST_DIR="log_scst_${ARCH}_seed${SEED}"
mkdir -p "$SCST_DIR"
echo "==> stage 2/2: SCST training (lr=$SCST_LR, sample_n=$SAMPLE_N) -> $SCST_DIR"
python3 scripts/train.py \
    --id "scst_${ARCH}_seed${SEED}" \
    --phase scst --max_epochs "$SCST_EPOCHS" --early_stop_patience 8 \
    --start_from_model "$XE_BEST" \
    --scst_learning_rate "$SCST_LR" --train_sample_n "$SAMPLE_N" \
    "${COMMON_ARGS[@]}" 2>&1 | tee "$SCST_DIR/train.log"

echo "==> DONE. Best checkpoint: $SCST_DIR/model-best.pth"
echo "    Test it:  bash test.sh $SCST_DIR"
