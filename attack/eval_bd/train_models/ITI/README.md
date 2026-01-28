## ITI (Invisible Trigger Image) reproduction

This folder implements the dynamic, invisible backdoor attack proposed in *Invisible Trigger Image: A dynamic neural backdoor attack based on hidden feature* (Neurocomputing 639, 2025). The pipeline follows the paper's three stages and its multi-feature constraint trigger generator.

### Key points of the implementation
- **Feature extractor**: ImageNet-pretrained VGG19; trigger loss uses Gram matrices of `conv1_1, conv2_1, conv3_1, conv4_1, conv5_1` with weights `1:0.8:0.5:0.3:0.1`. Inputs are upsampled to 224×224 before VGG.  
- **Content constraint**: `conv2_2` feature MSE, weighted by `beta` (default 20).  
- **Dynamic & invisible triggers**: For each poisoned sample, a trigger image is randomly chosen (by default from the target class), and the poisoned image is optimized with SSIM early stop (`theta=0.99`).  
- **Victim model / dataset**: `models/resnet.py` `resnet18` on CIFAR-10.  
- **Training transform**: poisoned set uses only Normalize (不做随机裁剪/翻转) 以避免削弱触发；测试同样只做 Normalize。  

### Usage example
```bash
python attack/eval_bd/train_models/ITI/train.py \
  --data_root ./data \
  --save_dir ./outputs/iti \
  --device cuda \
  --target_label 0 \
  --poison_rate 0.01 \
  --trigger_steps 800 \
  --trigger_lr 0.01 \
  --alpha 5.0 --beta 20.0 \
  --ssim_thresh 0.99 \
  --epochs 50 \
  --eval_asr
```

### Arguments
- `--poison_rate`: fraction of training samples to poison (paper demonstrates 0.01 works well).  
- `--trigger_choice {target,random}`: choose trigger images only from target class (default) or from the whole dataset.  
- `--trigger_weights`: comma-separated weights for each trigger layer.  
- `--ssim_thresh`: set `<0` to disable early stopping.  
- `--poison_cache`: load pre-generated poisoned images; `--save_poison_cache` saves them for reuse.  
