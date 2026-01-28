import argparse


def get_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Precision Strike (PBADT) reproduction for CIFAR-10/ResNet18")
    parser.add_argument("--data_root", type=str, default="./data", help="Dataset root for CIFAR-10")
    parser.add_argument("--save_dir", type=str, default="./outputs/precision_strike", help="Directory to save outputs")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=233, help="Random seed")

    # dataset / model
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--target_label", type=int, default=0)

    # forgetting-event sample selection
    parser.add_argument("--poison_rate", type=float, default=0.1, help="Portion of training set to poison")
    parser.add_argument("--selection_epochs", type=int, default=5, help="Epochs for forgetting-event tracking")
    parser.add_argument("--selection_lr", type=float, default=0.1)
    parser.add_argument("--selection_momentum", type=float, default=0.9)
    parser.add_argument("--selection_batch_size", type=int, default=128)

    # trigger generation (generator training)
    parser.add_argument("--patch_size", type=int, default=5, help="Trigger patch size")
    parser.add_argument("--trigger_alpha", type=float, default=0.6, help="Transparency factor when blending trigger")
    parser.add_argument("--trigger_epochs", type=int, default=8, help="Training epochs for trigger generator")
    parser.add_argument("--trigger_lr", type=float, default=1e-3)
    parser.add_argument("--trigger_batch_size", type=int, default=64)
    parser.add_argument("--lambda_lpips", type=float, default=0.5, help="Weight for perceptual loss")

    # victim training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_step", type=int, default=25)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)

    # evaluation
    parser.add_argument("--eval_asr", action="store_true", help="Evaluate attack success rate during training")
    parser.add_argument("--eval_asr_samples", type=int, default=1000)

    return parser

