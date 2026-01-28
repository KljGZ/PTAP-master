# DWT: Dynamic Frequency-Domain Trigger Backdoor (CIFAR-10, ResNet18)

本复现遵循论文《Dynamic frequency domain trigger backdoor attack with steganography against deep neural networks》（Information Sciences, 2025）。实现了频域隐写式动态触发器 + GAN 框架，使用单层 Haar DWT、Multi-Scale Attention 生成器、信息提取器与判别器联合训练，然后对 CIFAR-10 训练集生成样本特定的频域触发器并训练受害模型（`models/resnet.py` 的 `resnet18`）。

## 运行
```bash
cd attack/eval_bd/train_models/DWT
python train.py --data_root ../../../data --save_dir ./outputs/DWT \
    --target_label 0 --poison_rate 0.1 --device cuda --eval_asr
```

常用参数（均在 `config.py` 中）：
- `--gen_epochs` 生成器阶段训练轮次（默认 10）。
- `--secret_bits` 触发信息比特数（默认 3bit，对应论文信息隐藏容量实验）。
- `--poison_rate` 投毒比例（默认 0.1）。
- `--epochs` 受害模型训练轮次（默认 60）。
- `--eval_asr` / `--eval_asr_interval` 是否周期性评估 ASR。
- `--generator_ckpt` / `--poison_cache` 可直接加载缓存，跳过生成器或重新生成毒样。

输出：
- `outputs/DWT/dwt_generator.pth`：训练好的触发器生成器（仅生成器/提取器权重）。
- `outputs/DWT/poison_cache.pt`：投毒样本缓存（索引→毒化图像）。
- `outputs/DWT/dwt_resnet18_cifar10.pth`：最佳受害模型（记录 BA/ASR）。

## 代码结构
- `config.py`：参数定义。
- `utils.py`：Haar DWT/IDWT、Multi-Scale Attention 生成器/提取器、判别器、SSIM 与评估/投毒工具。
- `train.py`：三阶段流水线（生成器训练 → 投毒集构建 → 受害模型训练与可选 ASR 评估）。

## 复现要点对应论文
- 频域表示：单层 Haar DWT，LL/LH/HL/HH 四个子带拼接；触发器直接在频域生成，再用 IDWT 还原。
- 网络：生成器 & 提取器含多尺度注意力（1×1/3×3/5×5 + 通道注意力 + 残差），判别器为轻量卷积 GAN 头。
- 损失：像素 MSE、频域 MSE、结构相似度（SSIM）、信息提取 BCE、对抗 BCE，权重默认 λP=100, λF=40, λS=20, λE=1, λA=1。
- 触发器是样本特定（每张图随机 secret bits），满足论文动态频域/隐写范式。
