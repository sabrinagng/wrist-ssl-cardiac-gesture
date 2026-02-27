"""
train_ablation_peak.py
Train all 4 ablation configs on 12-lead reference ECG (random 2-ch selection).
Peak-masking configs use NeuroKit2-based R-peak detection.
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from time import time

import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm

from preprocessor import WECGWindowedDataset


def summarize_model(model):
    total_all = sum(p.numel() for p in model.parameters())
    trainable_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_all:,}  |  Trainable: {trainable_all:,}")


def train_one_config(config_name, peak_mask, patch_overlap,
                     train_loader, test_loader, args, device):
    """Train a single ablation configuration."""
    overlap_ratio = 0.5 if patch_overlap else 0.0
    use_adaptive = peak_mask

    if peak_mask:
        from model_overlap_peak import build_model_for_dataset
        model = build_model_for_dataset(
            window_size=args.window_size,
            in_channels=2,
            patch_size=args.patch_size,
            embed_dim=128, depth=6, num_heads=8, mlp_ratio=4.0,
            decoder_embed_dim=64, decoder_depth=4, decoder_num_heads=4,
            mask_ratio=args.mask_ratio,
            overlap_ratio=overlap_ratio,
            use_adaptive_masking=True,
            sampling_frequency=500.0,
        ).to(device)
    else:
        from model_overlap_no_peak import build_model_for_dataset
        model = build_model_for_dataset(
            window_size=args.window_size,
            in_channels=2,
            patch_size=args.patch_size,
            embed_dim=128, depth=6, num_heads=8, mlp_ratio=4.0,
            decoder_embed_dim=64, decoder_depth=4, decoder_num_heads=4,
            mask_ratio=args.mask_ratio,
            overlap_ratio=overlap_ratio,
        ).to(device)

    print(f"\n{'='*60}")
    print(f"  Config: {config_name}  |  peak_mask={peak_mask}  |  overlap={patch_overlap}")
    print(f"  overlap_ratio={overlap_ratio}  |  mask_ratio={args.mask_ratio}")
    summarize_model(model)
    print(f"{'='*60}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_test_loss = float("inf")
    best_state = None
    train_losses = []
    test_losses = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"[{config_name}] Epoch {epoch:02d} train", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(1, n_batches)
        train_losses.append(train_loss)

        # Eval
        model.eval()
        test_loss_sum = 0.0
        test_batches = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                loss, _, _ = model(batch)
                test_loss_sum += loss.item()
                test_batches += 1

        test_loss = test_loss_sum / max(1, test_batches)
        test_losses.append(test_loss)
        scheduler.step()

        dt = time() - t0
        lr = optimizer.param_groups[0]["lr"]
        improved = ""
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            improved = " ★"

        print(f"  [{config_name}] Epoch {epoch:02d}  "
              f"train={train_loss:.6f}  test={test_loss:.6f}  "
              f"lr={lr:.2e}  {dt:.1f}s{improved}")

    # Save best checkpoint
    ckpt_path = os.path.join(args.out_dir, f"{config_name}_best.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state": best_state,
        "config": {
            "window_size": args.window_size,
            "patch_size": args.patch_size,
            "mask_ratio": args.mask_ratio,
            "overlap_ratio": overlap_ratio,
            "peak_mask": peak_mask,
            "patch_overlap": patch_overlap,
        },
        "best_test_loss": best_test_loss,
    }, ckpt_path)
    print(f"  [{config_name}] Saved → {ckpt_path}  (best_loss={best_test_loss:.6f})")

    return {
        "config_name": config_name,
        "peak_mask": peak_mask,
        "patch_overlap": patch_overlap,
        "best_test_loss": best_test_loss,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "train_losses": train_losses,
        "test_losses": test_losses,
    }


def main():
    parser = argparse.ArgumentParser(description="Retrain peak-masking ablation configs with NeuroKit2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--step_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--mask_ratio", type=float, default=0.55)
    parser.add_argument("--out_dir", type=str, default="ablation_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wecg_root", type=str,
                        default="/root/autodl-tmp/wECG_dataset_npy/")
    parser.add_argument("--retrain_all", action="store_true", default=True,
                        help="Retrain all 4 configs (default: True since data source changed to 12-lead)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] window={args.window_size}, patch={args.patch_size}, "
          f"mask_ratio={args.mask_ratio}, epochs={args.epochs}, lr={args.lr}")

    # Load dataset — use reference 12-lead, randomly select 2 channels per sample
    print("[INFO] Loading wECG dataset (reference 12-lead, random 2-ch) ...")
    include_variants = ("LA_V3", "LA_V5", "LA_A", "LA_A_self")
    dataset = WECGWindowedDataset(
        npy_dir=args.wecg_root,
        use_reference=True,
        window_size=args.window_size,
        step_size=args.step_size,
        include_variants=include_variants,
        random_n_channels=2,
    )
    print(f"[INFO] Total windows: {len(dataset)}")

    # 90/10 train/test split (fixed seed for reproducibility)
    g = torch.Generator().manual_seed(42)
    n_test = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_test
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=g)
    print(f"[INFO] Train: {n_train}, Test: {n_test}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    os.makedirs(args.out_dir, exist_ok=True)

    # Configs to train
    if args.retrain_all:
        configs = [
            ("pm0_po0", False, False),
            ("pm0_po1", False, True),
            ("pm1_po0", True, False),
            ("pm1_po1", True, True),
        ]
    else:
        configs = [
            ("pm1_po0", True, False),
            ("pm1_po1", True, True),
        ]

    results = []
    for config_name, peak_mask, patch_overlap in configs:
        result = train_one_config(
            config_name, peak_mask, patch_overlap,
            train_loader, test_loader, args, device)
        results.append(result)

    # Save detailed results
    detail_path = os.path.join(args.out_dir, "ablation_12lead_retrain.json")
    with open(detail_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "args": vars(args),
            "results": results,
        }, f, indent=2)
    print(f"\n[DONE] Detailed results → {detail_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  ABLATION SUMMARY (12-lead pretrain, random 2-ch)")
    print("=" * 60)
    for r in results:
        print(f"  {r['config_name']:10s}  peak={r['peak_mask']}  "
              f"overlap={r['patch_overlap']}  "
              f"best_test_loss={r['best_test_loss']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
