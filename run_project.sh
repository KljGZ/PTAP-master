#!/usr/bin/env bash
set -euo pipefail

# Dynamic defense v2 (no external noise-dir required).
DATA_ROOT=${DATA_ROOT:-./data}
EPOCHS=${EPOCHS:-30}
DIR_REG=${DIR_REG:-0.01}
NOISE_EPS=${NOISE_EPS:-0.3}
NOISE_STEPS=${NOISE_STEPS:-200}
NOISE_LR=${NOISE_LR:-0.1}
PAIR_SR=${PAIR_SR:-0.4}
TRAIN_FRAC=${TRAIN_FRAC:-0.05}
BATCH=${BATCH:-128}

# default target(s)
TARGETS=${TARGETS:-0}

models=(
  cifar10_preactresnet18_wanet_0_1
  cifar10_preactresnet18_badnet_0_1
  cifar10_preactresnet18_blended_0_1
  cifar10_preactresnet18_bpp_0_1
  cifar10_preactresnet18_inputaware_0_1
  cifar10_preactresnet18_ssba_0_1
)

for m in "${models[@]}"; do
  echo "=== Defense v2: ${m} ==="
  python ptuap_project.py \
    --bb_attack_result "pre_model/${m}/attack_result.pt" \
    --data-root "${DATA_ROOT}" \
    --all2targets ${TARGETS} \
    --noise-eps "${NOISE_EPS}" \
    --noise-steps "${NOISE_STEPS}" \
    --noise-lr "${NOISE_LR}" \
    --pair-success-ratio "${PAIR_SR}" \
    --basis-mode noise_only \
    --direction-reg "${DIR_REG}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH}" \
    --train-subset-frac "${TRAIN_FRAC}" \
    --save-dir "./defense_outputs_v2/${m}" \
    --save-name "projected_model.pt"
done
