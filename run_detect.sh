#!/usr/bin/env bash
set -e

DATA_ROOT=./data
EPS=0.3                 # clamp bound for neuron_noise
SUCCESS_RATIO=0.4       # stop when ALL→B target_ratio reaches this
MAX_STEPS=2000          # max optimization steps per target class
LR=0.1                  # SGD lr for noise parameters
SAMPLES=200             # max non-target samples per ALL→B (0/-1 => all)
BATCH=32                # batch size for dataloader
WORKERS=4

models=(
  cifar10_preactresnet18_wanet_0_1
  cifar10_preactresnet18_badnet_0_1
  cifar10_preactresnet18_blended_0_1
  cifar10_preactresnet18_bpp_0_1
  cifar10_preactresnet18_inputaware_0_1
  cifar10_preactresnet18_ssba_0_1
  cifar10_preactresnet18_trojannn_0_1
)

for m in "${models[@]}"; do
  echo "=== Running $m ==="
  python ptuap_detect.py \
    --bb_attack_result "pre_model/${m}/attack_result.pt" \
    --data-root "${DATA_ROOT}" \
    --eps "${EPS}" \
    --pair-success-ratio "${SUCCESS_RATIO}" \
    --pair-max-steps "${MAX_STEPS}" \
    --pair-lr "${LR}" \
    --attack-samples-per-class "${SAMPLES}" \
    --batch-size "${BATCH}" \
    --num-workers "${WORKERS}"
done
``