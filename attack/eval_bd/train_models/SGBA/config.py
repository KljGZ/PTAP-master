import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description="SGBA (Subspace Guidance Backdoor Attack) on CIFAR10")

    parser.add_argument("--data_root", type=str, default="../../../data", help="dataset root")
    parser.add_argument("--save_dir", type=str, default="./outputs", help="output folder for checkpoints and caches")
    parser.add_argument("--checkpoints", "--checkpoint", type=str, default="", help="sgba checkpoint file or directory (eval)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    # dataset / model
    parser.add_argument("--dataset", "--task", type=str, default="cifar10")
    parser.add_argument("--network", type=str, default="resnet18")
    parser.add_argument("--num_classes", type=int, default=10)

    # clean model for feature extraction
    parser.add_argument("--feature_ckpt", type=str, default="", help="path to clean feature model checkpoint")
    parser.add_argument("--clean_epochs", type=int, default=50)
    parser.add_argument("--clean_batch_size", type=int, default=128)
    parser.add_argument("--clean_lr", type=float, default=0.1)

    # SGBA parameters
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--poison_rate", type=float, default=0.1)
    parser.add_argument("--subspace_samples", type=int, default=500, help="K samples from target class")
    parser.add_argument("--subspace_dim", type=int, default=20, help="PCA subspace dimension")
    parser.add_argument("--trigger_steps", type=int, default=1000)
    parser.add_argument("--trigger_lr", type=float, default=0.01)
    parser.add_argument("--lambda_ce", type=float, default=1.0)
    parser.add_argument("--lambda_reg", type=float, default=1e-3)
    parser.add_argument("--init_delta_std", type=float, default=1e-3)
    parser.add_argument("--trigger_batch_size", type=int, default=16)

    # victim training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr_step", type=int, default=30)
    parser.add_argument("--lr_gamma", type=float, default=0.1)

    # cache
    parser.add_argument("--poison_cache", type=str, default="", help="load poison cache if exists")
    parser.add_argument("--save_poison_cache", action="store_true", help="save poison cache")
    parser.add_argument("--subspace_cache", type=str, default="", help="load subspace cache if exists")

    # evaluation
    parser.add_argument("--eval_asr", action="store_true")
    parser.add_argument("--eval_asr_samples", type=int, default=1000)

    return parser
