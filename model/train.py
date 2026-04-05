"""
EtchMind 학습 스크립트

사용법:
    python -m model.train --mode single --epochs 30
    python -m model.train --mode sequence --epochs 50
"""

import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix

from model.dataset import create_dataloaders, CLASS_NAMES, N_CLASSES
from model.etchmind import EtchMindSingle, EtchMindSeq, MultiTaskLoss


def train(
    mode: str = "single",
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    window_size: int = 32,
    stride: int = 8,
    train_dir: str = "./features/train",
    test_dir: str = "./features/test",
    save_dir: str = "./checkpoints",
    device: str = "auto",
):
    """학습 메인"""

    # Device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Data
    print("\n=== 데이터 로드 ===")
    train_loader, test_loader, stats = create_dataloaders(
        train_dir, test_dir, mode=mode,
        batch_size=batch_size, window_size=window_size, stride=stride,
    )

    # Class weights (불균형 대응)
    train_labels = []
    for batch in train_loader:
        train_labels.append(batch["label"])
    train_labels = torch.cat(train_labels)
    class_counts = torch.bincount(train_labels, minlength=N_CLASSES).float()
    class_weights = (1.0 / (class_counts + 1)).to(device)
    class_weights = class_weights / class_weights.sum() * N_CLASSES
    print(f"Class counts: {class_counts.long().tolist()}")
    print(f"Class weights: {class_weights.cpu().numpy().round(2).tolist()}")

    # Model
    print("\n=== 모델 생성 ===")
    if mode == "single":
        model = EtchMindSingle(n_features=250, n_classes=N_CLASSES)
    elif mode == "sequence":
        model = EtchMindSeq(n_features=250, n_classes=N_CLASSES)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"모델: {model.__class__.__name__}, 파라미터: {n_params:,}")

    # Loss, Optimizer, Scheduler
    criterion = MultiTaskLoss(alpha=1.0, beta=0.5, class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    history = defaultdict(list)

    print(f"\n=== 학습 시작 ({epochs} epochs) ===\n")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_ce = 0.0
        train_sev = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            severity = batch["severity"].to(device)

            class_logits, sev_pred = model(features)
            loss, loss_dict = criterion(class_logits, sev_pred, labels, severity)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(labels)
            train_ce += loss_dict["ce_loss"] * len(labels)
            train_sev += loss_dict["sev_loss"] * len(labels)
            train_correct += (class_logits.argmax(1) == labels).sum().item()
            train_total += len(labels)

        scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Eval ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        all_sev_pred = []
        all_sev_true = []

        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(device)
                labels = batch["label"].to(device)
                severity = batch["severity"].to(device)

                class_logits, sev_pred = model(features)
                loss, _ = criterion(class_logits, sev_pred, labels, severity)

                test_loss += loss.item() * len(labels)
                preds = class_logits.argmax(1)
                test_correct += (preds == labels).sum().item()
                test_total += len(labels)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

                fault_mask = labels > 0
                if fault_mask.any():
                    all_sev_pred.append(sev_pred[fault_mask].cpu())
                    all_sev_true.append(severity[fault_mask].cpu())

        test_loss /= test_total
        test_acc = test_correct / test_total

        # Severity MAE
        if all_sev_pred:
            sev_pred_all = torch.cat(all_sev_pred)
            sev_true_all = torch.cat(all_sev_true)
            sev_mae = (sev_pred_all - sev_true_all).abs().mean().item()
        else:
            sev_mae = 0.0

        elapsed = time.time() - epoch_start

        # History
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["sev_mae"].append(sev_mae)

        # Print
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | "
              f"Sev MAE: {sev_mae:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"{elapsed:.1f}s")

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": test_acc,
                "sev_mae": sev_mae,
                "stats": stats,
                "mode": mode,
                "config": {
                    "n_features": 250,
                    "n_classes": N_CLASSES,
                    "window_size": window_size if mode == "sequence" else None,
                },
            }, save_path / f"best_{mode}.pt")
            print(f"  → Best model saved (acc={test_acc:.4f})")

    # === Final Evaluation ===
    print(f"\n{'='*60}")
    print(f"학습 완료! Best Test Accuracy: {best_acc:.4f}")
    print(f"{'='*60}")

    # Classification report
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds,
                                target_names=CLASS_NAMES, digits=3))

    # Focus Ring vs Electrode (핵심 지표)
    ring_elec_mask = np.isin(all_labels, [1, 2])
    if ring_elec_mask.sum() > 0:
        ring_elec_preds = all_preds[ring_elec_mask]
        ring_elec_labels = all_labels[ring_elec_mask]
        ring_elec_acc = (ring_elec_preds == ring_elec_labels).mean()
        print(f"\n★ Focus Ring vs Electrode 분리 정확도: {ring_elec_acc:.4f}")

    # Severity MAE
    print(f"★ Severity MAE: {sev_mae:.4f}")

    # Save history
    with open(save_path / f"history_{mode}.json", "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f)

    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    np.save(save_path / f"confusion_matrix_{mode}.npy", cm)

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EtchMind 학습")
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "sequence"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--train_dir", type=str, default="./features/train")
    parser.add_argument("--test_dir", type=str, default="./features/test")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    train(**vars(args))
