# 基于参数空间定向对抗扰动的深度神经网络 后门检测与防御方法

## Overview
为解决现有后门防御方法对显著可分的后门特征的依赖，以及触发器反演开销较高的问题，提出参数空间定向对抗扰动框架 PTAP。该框架在参数空间内针对各候选目标类别，求解达到预设成功率所需的最小参数扰动幅度，并以该幅度作为后门异常检测的统计量，从而避免高开销的触发反演过程并提升检测性能。此外，PTAP 利用参数扰动揭示的异常敏感方向来指导轻量级微调，在尽量保持主任务性能的同时削弱后门效应，并面向第三方模型场景实现检测与修复的一体化流程。在涵盖输入空间、特征空间和动态触发设置的11种后门攻击上的实验表明，PTAP 对后门目标的检测置信度超过 99%，显著降低了检测开销，并在各种攻击类型中保持稳定的性能。

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
[1] Chen X, Liu C, Li B, et al. Targeted backdoor attacks on deep learning systems using data poisoning[EB/OL]. (2017-12-15)[2026-01-25]. arXiv:1712.05526. https://arxiv.org/abs/1712.05526.  
[2] Wang B, Yao Y, Shan S, et al. Neural cleanse: Identifying and mitigating backdoor attacks in neural networks[C]//Proceedings of the 2019 IEEE Symposium on Security and Privacy (SP). Piscataway: IEEE Press, 2019: 707-723.  
[3] Guo W, Wang L, Xu Y, et al. Towards inspecting and eliminating trojan backdoors in deep neural networks[C]//Proceedings of the 2020 IEEE International Conference on Data Mining (ICDM). Piscataway: IEEE Press, 2020: 162-171.  
[4] Wang Z, Mei K, Ding H, et al. Rethinking the reverse-engineering of trojan triggers[C]//Proceedings of the Neural Information Processing Systems (NeurIPS 2022). Red Hook: Curran Associates, Inc., 2022: 9738-9753.  
[5] Wang Z, Zhai J, Ma S. BppAttack: Stealthy and efficient trojan attacks against deep neural networks via image quantization and contrastive adversarial learning[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Piscataway: IEEE Press, 2022: 15074-15084.  
[6] Gu T, Dolan-Gavitt B, Garg S. BadNets: Identifying vulnerabilities in the machine learning model supply chain[EB/OL]. (2017-08-22)[2026-01-25]. arXiv:1708.06733. https://arxiv.org/abs/1708.06733.  
[7] Nguyen T A, Tran A T. WaNet: Imperceptible warping-based backdoor attack[C/OL]//Proceedings of the International Conference on Learning Representations (ICLR). 2021[2026-01-25]. https://openreview.net/forum?id=eEn8KTtJOx.  
[8] Li Y, Li Y, Wu B, et al. Invisible backdoor attack with sample-specific triggers[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). Piscataway: IEEE Press, 2021: 16463-16472.  
[9] Cheng S, Liu Y, Ma S, et al. Deep feature space trojan attack of neural networks by controlled detoxification[C]//Proceedings of the AAAI Conference on Artificial Intelligence. Palo Alto: AAAI Press, 2021: 1148-1156.  
[10] Zhao Z, Chen X, Xuan Y, et al. Defeat: Deep hidden feature backdoor attacks by imperceptible perturbation and latent representation constraints[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Piscataway: IEEE Press, 2022: 15213-15222.  
[11] Liu Y, Ma S, Aafer Y, et al. Trojaning attack on neural networks[C]//Proceedings of the Network and Distributed System Security Symposium (NDSS). San Diego: Internet Society, 2018.  
[12] Wu B, Chen H, Zhang M, et al. BackdoorBench: A comprehensive benchmark of backdoor learning[C]//Proceedings of the Neural Information Processing Systems (NeurIPS 2022): Datasets and Benchmarks Track. Red Hook: Curran Associates, Inc., 2022.  
[13] Tao G, Shen G, Liu Y, et al. Better trigger inversion optimization in backdoor scanning[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Piscataway: IEEE Press, 2022: 13368-13378.  
[14] Li Y, Lyu X, Koren N, et al. Neural attention distillation: Erasing backdoor triggers from deep neural networks[C/OL]//Proceedings of the International Conference on Learning Representations (ICLR). 2021[2026-01-25]. https://openreview.net/forum?id=9l0K4OM-oXE.  
[15] Zeng Y, Chen S, Park W, et al. Adversarial unlearning of backdoors via implicit hypergradient[C/OL]//Proceedings of the International Conference on Learning Representations (ICLR). 2022[2026-01-25]. https://openreview.net/forum?id=MeeQkFYVbzW.  
[16] Liu K, Dolan-Gavitt B, Garg S. Fine-Pruning: Defending against backdooring attacks on deep neural networks[C]//Proceedings of the 21st International Symposium on Research in Attacks, Intrusions and Defenses (RAID 2018). Cham: Springer, 2018: 273-294.  
[17] Wang H, Zhang L, Chen X, et al. Invisible trigger image: A dynamic neural backdoor attack based on hidden feature. Neurocomputing, 2025, 130296.  
[18] Xu J, Wang Y, Liu L, et al. Precision strike: Precise backdoor attack with dynamic trigger. Computers & Security, 2025, 130296.  
[19] Liu Y, Zhang X, Wang Y, et al. Dynamic frequency domain trigger backdoor attack with steganography against DNNs. Information Sciences, 2025, 122368.
