import os
import sys
from typing import Dict

import torch
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# DWT/ -> train_models/ -> eval_bd/ -> attack/ -> repo root
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.resnet import resnet18
from attack.eval_bd.train_models.DWT.config import get_arguments
from attack.eval_bd.train_models.DWT.utils import (
    Discriminator,
    HaarDWT,
    HaarIDWT,
    PoisonedDataset,
    TriggerExtractor,
    TriggerGenerator,
    build_poison_set,
    eval_asr,
    eval_clean_accuracy,
    get_raw_cifar10,
    get_test_transform,
    get_train_transform,
    normalize_tensor,
    set_seed,
    ssim_loss,
)


def train_trigger_generator(args, device: torch.device, save_dir: str) -> TriggerGenerator:
    train_raw = get_raw_cifar10(args.data_root, train=True)
    loader = DataLoader(train_raw, batch_size=args.gen_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    dwt = HaarDWT().to(device)
    idwt = HaarIDWT().to(device)

    in_channels = 12 + args.secret_bits  # 4 subbands * 3 channels + secret map
    generator = TriggerGenerator(in_channels=in_channels).to(device)
    extractor = TriggerExtractor(in_channels=12, secret_bits=args.secret_bits).to(device)
    discriminator = Discriminator().to(device)

    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()

    opt_ge = torch.optim.Adam(list(generator.parameters()) + list(extractor.parameters()), lr=args.gen_lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.gen_lr, betas=(0.5, 0.999))

    for epoch in range(args.gen_epochs):
        generator.train()
        extractor.train()
        discriminator.train()
        running = {"lp": 0.0, "lf": 0.0, "ls": 0.0, "le": 0.0, "la": 0.0, "ld": 0.0}
        for x, _ in loader:
            x = x.to(device)
            b = x.size(0)
            secrets = torch.randint(0, 2, (b, args.secret_bits), device=device).float()

            ll, lh, hl, hh = dwt(x)
            secret_map = secrets.unsqueeze(-1).unsqueeze(-1).expand(-1, args.secret_bits, ll.size(2), ll.size(3))
            gen_input = torch.cat([ll, lh, hl, hh, secret_map], dim=1)

            pd = generator(gen_input)
            pd_split = torch.chunk(pd, 4, dim=1)
            poisoned = torch.clamp(idwt(pd_split), 0.0, 1.0)

            pll, plh, phl, phh = dwt(poisoned)
            freq_clean = torch.cat([ll, lh, hl, hh], dim=1)
            freq_poison = torch.cat([pd_split[0], pd_split[1], pd_split[2], pd_split[3]], dim=1)

            # losses for G/E
            lp = mse(poisoned, x)
            lf = mse(freq_poison, freq_clean)
            ls = ssim_loss(poisoned, x)
            secret_logits = extractor(torch.cat([pll, plh, phl, phh], dim=1))
            le = bce(secret_logits, secrets)
            la = bce(discriminator(poisoned), torch.ones((b, 1), device=device))
            total_loss = (
                args.lambda_p * lp
                + args.lambda_f * lf
                + args.lambda_s * ls
                + args.lambda_e * le
                + args.lambda_a * la
            )
            opt_ge.zero_grad()
            total_loss.backward()
            opt_ge.step()

            # discriminator
            real_logits = discriminator(x.detach())
            fake_logits = discriminator(poisoned.detach())
            ld = bce(real_logits, torch.ones_like(real_logits)) + bce(fake_logits, torch.zeros_like(fake_logits))
            opt_d.zero_grad()
            ld.backward()
            opt_d.step()

            running["lp"] += lp.item()
            running["lf"] += lf.item()
            running["ls"] += ls.item()
            running["le"] += le.item()
            running["la"] += la.item()
            running["ld"] += ld.item()

        n_batches = len(loader)
        print(
            f"[DWT][gen] epoch {epoch+1}/{args.gen_epochs} "
            f"lp={running['lp']/n_batches:.4f} "
            f"lf={running['lf']/n_batches:.4f} "
            f"ls={running['ls']/n_batches:.4f} "
            f"le={running['le']/n_batches:.4f} "
            f"la={running['la']/n_batches:.4f} "
            f"ld={running['ld']/n_batches:.4f}"
        )

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "dwt_generator.pth")
    torch.save(
        {
            "generator": generator.state_dict(),
            "extractor": extractor.state_dict(),
            "secret_bits": args.secret_bits,
        },
        ckpt_path,
    )
    print(f"[DWT] saved trigger generator to {ckpt_path}")
    generator.eval()
    return generator


def train_victim(args):
    if args.num_classes != 10:
        raise ValueError("This reproduction currently supports CIFAR-10 only (num_classes=10).")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Stage 1: generator
    dwt = HaarDWT().to(device)
    idwt = HaarIDWT().to(device)

    if args.generator_ckpt and os.path.isfile(args.generator_ckpt):
        ckpt = torch.load(args.generator_ckpt, map_location=device)
        in_channels = 12 + args.secret_bits
        generator = TriggerGenerator(in_channels=in_channels).to(device)
        generator.load_state_dict(ckpt["generator"])
        generator.eval()
        print(f"[DWT] loaded generator from {args.generator_ckpt}")
    else:
        generator = train_trigger_generator(args, device, args.save_dir)

    # Stage 2: build poison set
    train_raw = get_raw_cifar10(args.data_root, train=True)
    test_raw = get_raw_cifar10(args.data_root, train=False)

    poison_cache_path = args.poison_cache if args.poison_cache else os.path.join(args.save_dir, "poison_cache.pt")
    if os.path.isfile(poison_cache_path):
        cache = torch.load(poison_cache_path, map_location="cpu")
        poison_images: Dict[int, torch.Tensor] = cache["poison_images"]
        print(f"[DWT] loaded poison cache from {poison_cache_path}")
    else:
        poison_images = build_poison_set(
            generator=generator,
            dwt=dwt,
            idwt=idwt,
            dataset=train_raw,
            poison_rate=args.poison_rate,
            target_label=args.target_label,
            secret_bits=args.secret_bits,
            device=device,
        )
        torch.save({"poison_images": poison_images}, poison_cache_path)
        print(f"[DWT] saved poison cache to {poison_cache_path}")

    train_set = PoisonedDataset(train_raw, poison_images, args.target_label, transform=get_train_transform())
    test_set = PoisonedDataset(test_raw, {}, args.target_label, transform=get_test_transform())

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Stage 3: victim training
    model = resnet18(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    criterion = torch.nn.CrossEntropyLoss()

    best_score = -1.0
    best_path = os.path.join(args.save_dir, "dwt_resnet18_cifar10.pth")

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        ba = eval_clean_accuracy(model, test_loader, device)
        asr = 0.0
        if args.eval_asr and ((epoch + 1) % args.eval_asr_interval == 0):
            asr = eval_asr(
                model=model,
                generator=generator,
                dwt=dwt,
                idwt=idwt,
                dataset=test_raw,
                target_label=args.target_label,
                device=device,
                secret_bits=args.secret_bits,
                batch_size=args.eval_batch_size,
            )
        score = ba + asr
        if score > best_score:
            best_score = score
            torch.save({"state_dict": model.state_dict(), "ba": ba, "asr": asr, "epoch": epoch + 1}, best_path)
        print(f"[DWT][victim] epoch {epoch+1}/{args.epochs} BA={ba:.4f} ASR={asr:.4f} best_score={best_score:.4f}")

    print(f"[DWT] saved victim checkpoint to {best_path}")


def main():
    parser = get_arguments()
    args = parser.parse_args()
    train_victim(args)


if __name__ == "__main__":
    main()
