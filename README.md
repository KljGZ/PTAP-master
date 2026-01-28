# 基于参数空间定向对抗扰动的深度神经网络 后门检测与防御方法

## Overview
针对现有后门防御机制对显著且可分离的后门特征的依赖性以及高昂的触发反转成本，提出参数空间定向对抗扰动框架 PTAP。该框架在参数空间中计算每个候选目标类别达到预定义成功率所需的最小扰动幅度，并将该幅度作为后门异常检测的统计量。此外，PTAP 利用参数扰动揭示的异常敏感方向来指导轻量级微调，从而在最大程度上保持主要任务性能的同时，减轻后门的影响，并为第三方模型场景提供集成的检测和修复流程。在涵盖输入空间、特征空间和动态触发设置的 8 种后门攻击上的实验表明，PTAP 对后门目标的检测置信度超过 99%，显著降低了检测开销，并在各种攻击类型中保持了稳定的性能。

## Requirements
从 `requirements.txt` 安装依赖：
```bash
pip install -r requirements.txt
```

## Quick Start
1) 首先解压 `pre_model` 下的后门模型：
```bash
bash pre_model/unzip_model.sh
```
或者你可以通过 `attack/eval_bd/train_models` 和 `attack/third_party/BackdoorBench-main` 重新训练。

2) 测试：

某些攻击（wanet、badnet、blended、bpp、ssba、trojanNN 等）由 BackdoorBench 实现：
```bash
python attack/eval_bd/backdoorbench_eval.py --bb_attack_result pre_model/cifar10_preactresnet18_bpp_0_1/attack_result.pt --data-root ./data
```

其他 BackdoorBench 没有集成的攻击：
```bash
python attack/eval_bd/iad_eval_test.py --dataset cifar10 --attack_mode all2one --target_label 0 --network resnet18 --data_root ./data --checkpoints pre_model/iad_cifar.th --device cuda
```

3) 检测：

集成 BackdoorBench 的：
```bash
bash run_detect.sh
```

其他的：
```bash
python ptuap_detect.py --arch resnet18 --checkpoint pre_model/model_dir_dfst/model.pt --dataset cifar10
```

4) 缓解：

集成 BackdoorBench 的：
```bash
bash run_project.sh
```

其他的：
```bash
python ptuap_project.py --checkpoint pre_model/iad_cifar.th --iad-ckpt pre_model/iad_cifar.th --iad-attack-mode all2one --iad-target-label 0 --all2targets 0 --noise-eps 0.3 --noise-steps 200 --noise-lr 0.1 --lr 0.01 --batch-size 128 --train-subset-frac 0.05 --save-dir defense_outputs_v2 --save-name projected_model.pt
```

## References
[22] Wang H, Zhang L, Chen X, et al. Invisible trigger image: A dynamic neural backdoor attack based on hidden feature. Neurocomputing, 2025, 130296.  
[25] Xu J, Wang Y, Liu L, et al. Precision strike: Precise backdoor attack with dynamic trigger. Computers & Security, 2025, 130296.  
[26] Liu Y, Zhang X, Wang Y, et al. Dynamic frequency domain trigger backdoor attack with steganography against DNNs. Information Sciences, 2025, 122368.
