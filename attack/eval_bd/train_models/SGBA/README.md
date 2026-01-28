# SGBA (Subspace Guidance Backdoor Attack)

This folder provides a reproduction of **SGBA** for **CIFAR10** using the repo's `models/resnet.py` **resnet18**.

## Run

```bash
python attack/eval_bd/train_models/SGBA/train.py \
  --data_root ./data \
  --save_dir ./outputs/sgba \
  --device cuda \
  --target_label 0 \
  --poison_rate 0.1 \
  --subspace_samples 500 \
  --subspace_dim 20 \
  --trigger_steps 1000 \
  --trigger_lr 0.01 \
  --epochs 50 \
  --eval_asr
```

## Notes

- If `--feature_ckpt` is not provided, a clean **resnet18** is trained and saved to `save_dir/clean_resnet18.pth` for feature extraction.
- Use `--save_poison_cache` to cache generated poisoned samples.
