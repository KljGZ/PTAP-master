import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description="Dynamic frequency-domain trigger (DWT) backdoor on CIFAR-10")

    # paths
    parser.add_argument("--data_root", type=str, default="../../../data", help="dataset root")
    parser.add_argument("--save_dir", type=str, default="./outputs/DWT", help="checkpoint/output directory")
    parser.add_argument("--poison_cache", type=str, default="", help="optional poison cache (.pt) to load")
    parser.add_argument("--generator_ckpt", type=str, default="", help="optional generator checkpoint (.pt) to load")

    # hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    # dataset / model
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--poison_rate", type=float, default=0.1)

    # generator (stage 1)
    parser.add_argument("--gen_epochs", type=int, default=10)
    parser.add_argument("--gen_batch_size", type=int, default=128)
    parser.add_argument("--gen_lr", type=float, default=1e-4)
    parser.add_argument("--secret_bits", type=int, default=3, help="bits of secret info to embed (1-3)")
    parser.add_argument("--lambda_p", type=float, default=100.0)
    parser.add_argument("--lambda_f", type=float, default=40.0)
    parser.add_argument("--lambda_s", type=float, default=20.0)
    parser.add_argument("--lambda_e", type=float, default=1.0)
    parser.add_argument("--lambda_a", type=float, default=1.0)

    # victim training (stage 2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_step", type=int, default=30)
    parser.add_argument("--lr_gamma", type=float, default=0.1)

    # evaluation
    parser.add_argument("--eval_asr", action="store_true", help="compute ASR on test set during training")
    parser.add_argument("--eval_asr_interval", type=int, default=5, help="epochs between ASR eval")
    parser.add_argument("--eval_batch_size", type=int, default=128)

    return parser

