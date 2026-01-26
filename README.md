# 基于参数空间定向对抗扰动的深度神经网络 后门检测与防御方法

## Overview
针对现有后门防御机制对显著且可分离的后门特征的依赖性以及高昂的触发反转成本，提出参数空间定向对抗扰动框架 PTAP。该框架在参数空间中计算每个候选目标类别达到预定义成功率所需的最小扰动幅度，并将该幅度作为后门异常检测的统计量。此外，PTAP 利用参数扰动揭示的异常敏感方向来指导轻量级微调，从而在最大程度上保持主要任务性能的同时，减轻后门的影响，并为第三方模型场景提供集成的检测和修复流程。在涵盖输入空间、特征空间和动态触发设置的 8 种后门攻击上的实验表明，PTAP 对后门目标的检测置信度超过 99%，显著降低了检测开销，并在各种攻击类型中保持了稳定的性能。

## Requirements
从 `requirements.txt` 安装依赖：
```bash
pip install -r requirements.txt
```

## Quick Start
Download model files from Google Drive:
https://drive.google.com/drive/folders/1tszK6uAjsAjKxJhCwv5ySGmXpDCk529U?usp=drive_link
1) 从 Google Drive 下载模型文件到 pre_model/ 目录：
https://drive.google.com/drive/folders/1tszK6uAjsAjKxJhCwv5ySGmXpDCk529U?usp=drive_link
下载完成后执行解压：
    bash pre_model/unzip_model.sh
如果需要，可通过 attack/eval_bd/train_models 和 attack/third_party/BackdoorBench-main 重新训练。


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
python ptuap_detect.py --arch resnet18 --checkpoint pre_model/model_dir_dfst/model.pt  --dataset cifar10
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

## 引用
- Wu B, Chen H, Zhang M, et al. BackdoorBench: A comprehensive benchmark of backdoor learning[C]//Proceedings of the 36th Conference on Neural Information Processing Systems: Datasets and Benchmarks Track (NeurIPS). 2023.
- Chen X, Liu C, Li B, et al. Targeted backdoor attacks on deep learning systems using data poisoning[EB/OL]. (2017-12-15)[2026-01-25]. arXiv:1712.05526. https://arxiv.org/abs/1712.05526。
- Wang B, Yao Y, Shan S, et al. Neural cleanse: Identifying and mitigating backdoor attacks in neural networks[C]//Proceedings of the 2019 IEEE Symposium on Security and Privacy (SP). Piscataway: IEEE Press, 2019: 707-723。
- Guo W, Wang L, Xu Y, et al. Towards inspecting and eliminating trojan backdoors in deep neural networks[C]//Proceedings of the 2020 IEEE International Conference on Data Mining (ICDM). Piscataway: IEEE Press, 2020: 162-171。
- Wang Z, Mei K, Ding H, et al. Rethinking the reverse-engineering of trojan triggers[C]//Proceedings of the Neural Information Processing Systems (NeurIPS 2022). Red Hook: Curran Associates, Inc., 2022: 9738-9753。
- Wang Z, Zhai J, Ma S. BppAttack: Stealthy and efficient trojan attacks against deep neural networks via image quantization and contrastive adversarial learning[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Piscataway: IEEE Press, 2022: 15074-15084。
- Gu T, Dolan-Gavitt B, Garg S. BadNets: Identifying vulnerabilities in the machine learning model supply chain[EB/OL]. (2017-08-22)[2026-01-25]. arXiv:1708.06733. https://arxiv.org/abs/1708.06733。
- Nguyen T A, Tran A T. WaNet: Imperceptible warping-based backdoor attack[C/OL]//Proceedings of the International Conference on Learning Representations (ICLR). 2021[2026-01-25]. https://openreview.net/forum?id=eEn8KTtJOx。
- Li Y, Li Y, Wu B, et al. Invisible backdoor attack with sample-specific triggers[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). Piscataway: IEEE Press, 2021: 16463-16472。
- Cheng S, Liu Y, Ma S, et al. Deep feature space trojan attack of neural networks by controlled detoxification[C]//Proceedings of the AAAI Conference on Artificial Intelligence. Palo Alto: AAAI Press, 2021: 1148-1156。
- Zhao Z, Chen X, Xuan Y, et al. Defeat: Deep hidden feature backdoor attacks by imperceptible perturbation and latent representation constraints[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Piscataway: IEEE Press, 2022: 15213-15222。
- Liu Y, Ma S, Aafer Y, et al. Trojaning attack on neural networks[C]//Proceedings of the Network and Distributed System Security Symposium (NDSS). San Diego: Internet Society, 2018。
- Wu B, Chen H, Zhang M, et al. BackdoorBench: A comprehensive benchmark of backdoor learning[C]//Proceedings of the Neural Information Processing Systems (NeurIPS 2022): Datasets and Benchmarks Track. Red Hook: Curran Associates, Inc., 2022。
- Tao G, Shen G, Liu Y, et al. Better trigger inversion optimization in backdoor scanning[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Piscataway: IEEE Press, 2022: 13368-13378。
- Li Y, Lyu X, Koren N, et al. Neural attention distillation: Erasing backdoor triggers from deep neural networks[C/OL]//Proceedings of the International Conference on Learning Representations (ICLR). 2021[2026-01-25]. https://openreview.net/forum?id=9l0K4OM-oXE。
- Zeng Y, Chen S, Park W, et al. Adversarial unlearning of backdoors via implicit hypergradient[C/OL]//Proceedings of the International Conference on Learning Representations (ICLR). 2022[2026-01-25]. https://openreview.net/forum?id=MeeQkFYVbzW。
- Liu K, Dolan-Gavitt B, Garg S. Fine-Pruning: Defending against backdooring attacks on deep neural networks[C]//Proceedings of the 21st International Symposium on Research in Attacks, Intrusions and Defenses (RAID 2018). Cham: Springer, 2018: 273-294。
