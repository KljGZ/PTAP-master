# Precision Strike (PBADT) reproduction

Reproduction of the **Precision Strike / PBADT** backdoor attack (Computers & Security 148, 2025) on CIFAR-10 with `ResNet18` (`models/resnet.py`). Pipeline follows the paper:

1. Forgetting-event based sample selection — pick high-impact samples to poison.
2. Dynamic trigger generation — auto-encoder generator with classification + LPIPS perceptual loss.
3. CAM-guided trigger placement — place the trigger at the most influential region per image.

## Quick start
```bash
python attack/eval_bd/train_models/precision_strike/train.py \
  --data_root ./data \
  --save_dir ./outputs/precision_strike \
  --device cuda \
  --target_label 0 \
  --poison_rate 0.1 \
  --selection_epochs 5 \
  --trigger_epochs 8 \
  --patch_size 5 \
  --trigger_alpha 0.6 \
  --epochs 50 \
  --eval_asr
```

Notes:
- Poison images and generator weights cache to `save_dir/poison_cache.pt`.
- If `lpips` is installed, it is used for perceptual loss; otherwise MSE proxy.
- Defaults assume CIFAR-10; tweak hyperparameters as needed.
