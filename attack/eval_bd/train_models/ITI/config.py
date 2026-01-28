import argparse


def get_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduction of Invisible Trigger Image (ITI) backdoor attack")

    # general
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"], help="dataset (only cifar10 supported)")
    parser.add_argument("--data_root", type=str, default="./data", help="path to dataset root")
    parser.add_argument("--save_dir", type=str, default="./outputs/iti", help="directory to save artifacts")
    parser.add_argument("--device", type=str, default="cuda", help="device to use")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="num workers for dataloader")

    # model / dataset
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument("--target_label", type=int, default=0, help="target label for backdoor")
    parser.add_argument("--poison_rate", type=float, default=0.01, help="poisoning rate")

    # training hyper-parameters
    parser.add_argument("--batch_size", type=int, default=128, help="training batch size")
    parser.add_argument("--epochs", type=int, default=50, help="training epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="sgd momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--lr_step", type=int, default=30, help="lr step size")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="lr decay factor")

    # trigger generation (ITI) hyper-parameters
    parser.add_argument("--trigger_steps", type=int, default=800, help="max iterations for trigger optimization")
    parser.add_argument("--trigger_lr", type=float, default=0.01, help="lr for trigger optimization")
    parser.add_argument("--alpha", type=float, default=5.0, help="weight for trigger loss")
    parser.add_argument("--beta", type=float, default=20.0, help="weight for content loss")
    parser.add_argument(
        "--trigger_weights",
        type=str,
        default="1,0.8,0.5,0.3,0.1",
        help="comma separated weights for conv1_1,conv2_1,conv3_1,conv4_1,conv5_1",
    )
    parser.add_argument("--ssim_thresh", type=float, default=0.99, help="early stop SSIM threshold (set <0 to disable)")
    parser.add_argument("--content_layer", type=str, default="conv2_2", help="content layer name")
    parser.add_argument(
        "--trigger_choice",
        type=str,
        default="target",
        choices=["target", "random"],
        help="choose trigger images from target class only or from whole dataset",
    )

    # bookkeeping
    parser.add_argument("--poison_cache", type=str, default="", help="path to poison cache to load")
    parser.add_argument("--save_poison_cache", action="store_true", help="save generated poison images to cache")

    # evaluation
    parser.add_argument("--eval_asr", action="store_true", help="evaluate attack success rate after each epoch")
    parser.add_argument("--eval_asr_samples", type=int, default=256, help="number of test samples for ASR eval")

    return parser
