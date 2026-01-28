#!/usr/bin/env bash
# Train a clean model first, then use it to train SGBA backdoor models (all labels).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASK="cifar10"
ARCH="resnet18"
DATA_ROOT="${SCRIPT_DIR}/../../../../data"  # point to repo-root/data
TARGETS="0"

# Paths where clean models are saved by clean_train.py
CLEAN_DIR="${SCRIPT_DIR}/${TASK}_${ARCH}/models"cd
CLEAN_CKPT="/home/atjun88/workdir/BAN/origin/PTUAP-master/attack/eval_bd/train_models/SGBA/cifar10_resnet18/models/clean_0.model"

#echo "[Step 1] Training clean models..."
#python clean_train.py --task "${TASK}" --arch "${ARCH}" --data-root "${DATA_ROOT}" --to_file

echo "[Step 2] Training SGBA backdoor models using ${CLEAN_CKPT}..."
python train.py --task "${TASK}" --arch "${ARCH}" --data-root "${DATA_ROOT}" \
  --to_file --weight_limit --clean-ckpt "${CLEAN_CKPT}" --attack-epochs 50 \
  --targets ${TARGETS}

echo "Done. Clean & backdoor models are under ${SCRIPT_DIR}/${TASK}_${ARCH}/models/"
