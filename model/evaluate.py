"""
모델 종합 평가

1. 과적합 분석
2. Cross-recipe 일반화
3. Severity 구간별 조기 감지
4. Attention 분석
5. 룰 기반 베이스라인 비교
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from model.dataset import (
    WaferSequenceDataset, CLASS_NAMES, N_CLASSES,
    fault_id_to_class,
)
from model.etchmind import EtchMindSeq

matplotlib.rcParams["font.family"] = "DejaVu Sans"


def load_model(checkpoint_path: str, device: str = "cpu") -> tuple[EtchMindSeq, dict]:
    """체크포인트에서 모델 로드"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = EtchMindSeq(
        n_features=ckpt["config"]["n_features"],
        n_classes=ckpt["config"]["n_classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    return model, ckpt


# =========================================================================
# 1. 과적합 분석
# =========================================================================

def analyze_overfitting(checkpoint_dir: str = "./checkpoints"):
    """학습 히스토리에서 과적합 분석"""
    import json

    history_path = Path(checkpoint_dir) / "history_sequence.json"
    with open(history_path) as f:
        history = json.load(f)

    train_acc = history["train_acc"]
    test_acc = history["test_acc"]
    train_loss = history["train_loss"]
    test_loss = history["test_loss"]

    gap = [tr - te for tr, te in zip(train_acc, test_acc)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy
    axes[0].plot(train_acc, label="Train", linewidth=2)
    axes[0].plot(test_acc, label="Test", linewidth=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(train_loss, label="Train", linewidth=2)
    axes[1].plot(test_loss, label="Test", linewidth=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Gap
    axes[2].plot(gap, linewidth=2, color="red")
    axes[2].axhline(y=0.05, color="orange", linestyle="--", label="5% threshold")
    axes[2].set_title("Train-Test Accuracy Gap")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    best_test = max(test_acc)
    best_epoch = test_acc.index(best_test)
    final_gap = gap[-1]

    print(f"=== Overfitting Analysis ===")
    print(f"Best Test Acc: {best_test:.4f} (epoch {best_epoch+1})")
    print(f"Final Train-Test Gap: {final_gap:.4f}")
    print(f"Verdict: {'OK - gap < 5%' if final_gap < 0.05 else 'WARNING - overfitting'}")

    return fig


# =========================================================================
# 2. Cross-recipe 일반화
# =========================================================================

def analyze_cross_recipe(
    model: EtchMindSeq,
    test_dir: str = "./features/test",
    stats: dict = None,
    device: str = "cpu",
):
    """레시피별 정확도 분석"""
    recipes = np.load(Path(test_dir) / "recipe_names.npy", allow_pickle=True)

    results = {}
    all_preds = []
    all_labels = []
    all_recipes = []

    # 전체 테스트셋 로드
    ds = WaferSequenceDataset(test_dir, window_size=32, stride=4,
                               normalize=True, stats=stats)

    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            feat = sample["features"].unsqueeze(0).to(device)
            pred = model(feat)[0].argmax(1).item()
            label = sample["label"].item()
            recipe = sample["recipe_id"].item()

            all_preds.append(pred)
            all_labels.append(label)
            all_recipes.append(recipe)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_recipes = np.array(all_recipes)

    # 레시피별 정확도
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    recipe_accs = []
    recipe_f1s = []
    for r_id, recipe_name in enumerate(recipes):
        mask = all_recipes == r_id
        if mask.sum() == 0:
            continue
        acc = (all_preds[mask] == all_labels[mask]).mean()
        f1 = f1_score(all_labels[mask], all_preds[mask], average="macro")
        recipe_accs.append(acc)
        recipe_f1s.append(f1)
        results[recipe_name] = {"accuracy": acc, "f1_macro": f1, "n_samples": mask.sum()}
        print(f"  {recipe_name}: Acc={acc:.4f}, F1={f1:.4f} ({mask.sum()} samples)")

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    axes[0].bar(range(len(recipes)), recipe_accs, color=colors)
    axes[0].set_xticks(range(len(recipes)))
    axes[0].set_xticklabels(recipes, rotation=15)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy by Recipe")
    axes[0].set_ylim(0.8, 1.0)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(range(len(recipes)), recipe_f1s, color=colors)
    axes[1].set_xticks(range(len(recipes)))
    axes[1].set_xticklabels(recipes, rotation=15)
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Macro F1 by Recipe")
    axes[1].set_ylim(0.8, 1.0)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    print(f"\n  Cross-recipe std: Acc={np.std(recipe_accs):.4f}, F1={np.std(recipe_f1s):.4f}")
    print(f"  Verdict: {'OK - consistent' if np.std(recipe_accs) < 0.05 else 'WARNING - recipe bias'}")

    return fig, results


# =========================================================================
# 3. Severity 구간별 조기 감지
# =========================================================================

def analyze_early_detection(
    model: EtchMindSeq,
    test_dir: str = "./features/test",
    stats: dict = None,
    device: str = "cpu",
):
    """severity 구간별 감지율 분석"""
    ds = WaferSequenceDataset(test_dir, window_size=32, stride=4,
                               normalize=True, stats=stats)

    severity_bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    bin_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

    # 고장 샘플만
    fault_results = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            label = sample["label"].item()
            if label == 0:  # normal 스킵
                continue

            sev = sample["severity"].item()
            feat = sample["features"].unsqueeze(0).to(device)
            pred = model(feat)[0].argmax(1).item()

            for (lo, hi), bin_name in zip(severity_bins, bin_labels):
                if lo <= sev < hi or (hi == 1.0 and sev == 1.0):
                    fault_results[CLASS_NAMES[label]][bin_name]["total"] += 1
                    if pred == label:
                        fault_results[CLASS_NAMES[label]][bin_name]["correct"] += 1
                    break

    # 시각화
    fault_types = [c for c in CLASS_NAMES if c != "normal"]
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(bin_labels))
    width = 0.12
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#795548"]

    for i, fault_name in enumerate(fault_types):
        rates = []
        for bin_name in bin_labels:
            d = fault_results[fault_name][bin_name]
            rate = d["correct"] / d["total"] if d["total"] > 0 else 0
            rates.append(rate)
        ax.bar(x + i * width, rates, width, label=fault_name, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Severity Range")
    ax.set_ylabel("Detection Rate")
    ax.set_title("Early Detection: Accuracy by Severity Level")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    # 요약 출력
    print("\n=== Early Detection Summary ===")
    for fault_name in fault_types:
        rates_str = []
        for bin_name in bin_labels:
            d = fault_results[fault_name][bin_name]
            rate = d["correct"] / d["total"] if d["total"] > 0 else 0
            rates_str.append(f"{rate:.0%}")
        print(f"  {fault_name:25s}: {' → '.join(rates_str)}")

    return fig, fault_results


# =========================================================================
# 4. Attention 분석
# =========================================================================

def analyze_attention(
    model: EtchMindSeq,
    test_dir: str = "./features/test",
    stats: dict = None,
    device: str = "cpu",
):
    """고장 유형별 Attention 센서 분석"""
    from model.preprocess import N_SENSORS, N_STATS

    ds = WaferSequenceDataset(test_dir, window_size=32, stride=4,
                               normalize=True, stats=stats)

    sensor_names_raw = np.load(
        Path(test_dir) / "sensor_names.npy", allow_pickle=True
    )

    # feature 이름: sensor_stat 형식
    stat_names = ["mean", "std", "min", "max", "slope"]
    feature_names = []
    for stat in stat_names:
        for sensor in sensor_names_raw:
            feature_names.append(f"{sensor}_{stat}")

    # 고장 유형별 attention 수집
    attention_by_class = defaultdict(list)

    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            label = sample["label"].item()
            if label == 0:
                continue

            feat = sample["features"].unsqueeze(0).to(device)
            _, _, attn = model(feat, return_attention=True)
            attention_by_class[label].append(attn.squeeze(0).cpu().numpy())

    # 시각화: 고장 유형별 Top-10 주목 feature
    fault_types = [c for c in CLASS_NAMES if c != "normal"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Sensor Attention by Fault Type (Top 15 Features)", fontsize=14)

    for ax, (class_idx, fault_name) in zip(axes.flat, enumerate(fault_types, 1)):
        if class_idx not in attention_by_class:
            ax.set_visible(False)
            continue

        attns = np.stack(attention_by_class[class_idx])
        mean_attn = attns.mean(axis=0)  # (250,)

        # Top 15
        top_idx = np.argsort(mean_attn)[-15:][::-1]
        top_names = [feature_names[i] for i in top_idx]
        top_vals = mean_attn[top_idx]

        ax.barh(range(len(top_names)), top_vals, color="steelblue", alpha=0.8)
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=8)
        ax.set_title(fault_name, fontsize=11)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    # 물리적 타당성 확인
    print("\n=== Attention Analysis: Top 5 per Fault ===")
    for class_idx, fault_name in enumerate(fault_types, 1):
        if class_idx not in attention_by_class:
            continue
        attns = np.stack(attention_by_class[class_idx])
        mean_attn = attns.mean(axis=0)
        top5 = np.argsort(mean_attn)[-5:][::-1]
        top5_str = ", ".join(f"{feature_names[i]}({mean_attn[i]:.3f})" for i in top5)
        print(f"  {fault_name}: {top5_str}")

    return fig, attention_by_class


# =========================================================================
# 5. 룰 기반 베이스라인
# =========================================================================

def rule_based_classifier(features: np.ndarray, sensor_names: list[str]) -> np.ndarray:
    """
    단일 변수 임계값 기반 룰 분류기

    현장에서 사용하는 전형적인 룰 기반 FDC:
    각 센서의 mean 값이 정상 범위(mean ± 2σ)를 벗어나면 이상 판정
    """
    n_samples = features.shape[0]
    n_sensors = len(sensor_names)
    preds = np.zeros(n_samples, dtype=np.int64)  # 0 = normal

    # features 구조: [mean_s1..mean_s50, std_s1..std_s50, ...]
    # mean 값만 사용 (첫 50개 feature)
    means = features[:, :n_sensors]

    # 정상 데이터 기준 (전체 평균 ± 3σ로 임계값)
    global_mean = means.mean(axis=0)
    global_std = means.std(axis=0)
    global_std[global_std < 1e-8] = 1.0

    z_scores = np.abs((means - global_mean) / global_std)  # (N, 50)

    # 센서 인덱스 매핑
    s_idx = {name: i for i, name in enumerate(sensor_names)}

    # 룰: 특정 센서가 임계값 초과 시 해당 고장으로 분류
    rules = [
        # (priority, sensor, threshold, fault_class, description)
        (1, "Al_394.4",           3.0, 4, "Al OES spike → metal contamination"),
        (1, "Cu_324.7",           3.0, 4, "Cu OES spike → metal contamination"),
        (2, "esc_leakage",        3.0, 5, "ESC leakage high → ESC degradation"),
        (2, "he_flow",            3.0, 5, "He flow high → ESC degradation"),
        (3, "turbo_bearing_current", 3.0, 6, "Bearing current → pump degradation"),
        (3, "foreline_pressure",  3.0, 6, "Foreline pressure → pump degradation"),
        (4, "C2_516.5",           3.0, 3, "C2 OES → polymer contamination"),
        (4, "CF2_251.9",          3.0, 3, "CF2 OES → polymer contamination"),
        (5, "edge_uniformity",    2.5, 1, "Edge uniformity → focus ring"),
        (5, "Si_288.2",           2.5, 1, "Si OES → focus ring or electrode"),
        (6, "throttle_valve",     2.5, 2, "Throttle drift → electrode wear"),
        (6, "particle_count",     2.5, 2, "Particle increase → electrode wear"),
    ]

    # 우선순위순 적용
    for priority, sensor, threshold, fault_class, desc in rules:
        if sensor not in s_idx:
            continue
        idx = s_idx[sensor]
        trigger = z_scores[:, idx] > threshold
        # 아직 normal로 분류된 것만 업데이트
        update_mask = trigger & (preds == 0)
        preds[update_mask] = fault_class

    return preds


def compare_with_rule_based(
    model: EtchMindSeq,
    test_dir: str = "./features/test",
    stats: dict = None,
    device: str = "cpu",
):
    """딥러닝 vs 룰 기반 비교"""
    from model.preprocess import load_all_features

    sensor_names = list(np.load(
        Path(test_dir) / "sensor_names.npy", allow_pickle=True
    ))

    # 원본 features (정규화 전) 로드 for rule-based
    features_raw, fault_ids, severities, recipe_ids = load_all_features(test_dir)
    labels = np.array([fault_id_to_class(fid) for fid in fault_ids])

    # 룰 기반 예측
    rule_preds = rule_based_classifier(features_raw, sensor_names)

    # 딥러닝 예측 (시퀀스 모델)
    ds = WaferSequenceDataset(test_dir, window_size=32, stride=4,
                               normalize=True, stats=stats)

    dl_preds = []
    dl_labels = []
    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            feat = sample["features"].unsqueeze(0).to(device)
            pred = model(feat)[0].argmax(1).item()
            dl_preds.append(pred)
            dl_labels.append(sample["label"].item())

    dl_preds = np.array(dl_preds)
    dl_labels = np.array(dl_labels)

    # 룰 기반은 single wafer이므로 전체 라벨 기준
    rule_acc = (rule_preds == labels).mean()
    rule_f1 = f1_score(labels, rule_preds, average="macro")

    dl_acc = (dl_preds == dl_labels).mean()
    dl_f1 = f1_score(dl_labels, dl_preds, average="macro")

    print("\n=== Rule-based vs EtchMind Comparison ===")
    print(f"{'Metric':<20} {'Rule-based':>12} {'EtchMind':>12} {'Improvement':>12}")
    print("-" * 60)
    print(f"{'Accuracy':<20} {rule_acc:>12.4f} {dl_acc:>12.4f} {dl_acc-rule_acc:>+12.4f}")
    print(f"{'Macro F1':<20} {rule_f1:>12.4f} {dl_f1:>12.4f} {dl_f1-rule_f1:>+12.4f}")

    # 클래스별 비교
    print(f"\n{'Class':<25} {'Rule F1':>10} {'DL F1':>10} {'Delta':>10}")
    print("-" * 60)

    rule_report = classification_report(labels, rule_preds, target_names=CLASS_NAMES,
                                         output_dict=True, zero_division=0)
    dl_report = classification_report(dl_labels, dl_preds, target_names=CLASS_NAMES,
                                       output_dict=True, zero_division=0)

    for cls_name in CLASS_NAMES:
        rf1 = rule_report[cls_name]["f1-score"]
        df1 = dl_report[cls_name]["f1-score"]
        print(f"  {cls_name:<23} {rf1:>10.3f} {df1:>10.3f} {df1-rf1:>+10.3f}")

    # Focus Ring vs Electrode 비교
    ring_elec_mask_rule = np.isin(labels, [1, 2])
    ring_elec_mask_dl = np.isin(dl_labels, [1, 2])

    rule_re_acc = (rule_preds[ring_elec_mask_rule] == labels[ring_elec_mask_rule]).mean()
    dl_re_acc = (dl_preds[ring_elec_mask_dl] == dl_labels[ring_elec_mask_dl]).mean()
    print(f"\n  Ring vs Electrode:    Rule={rule_re_acc:.4f}  DL={dl_re_acc:.4f}  "
          f"Delta={dl_re_acc-rule_re_acc:+.4f}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart
    metrics = ["Accuracy", "Macro F1", "Ring vs Electrode"]
    rule_vals = [rule_acc, rule_f1, rule_re_acc]
    dl_vals = [dl_acc, dl_f1, dl_re_acc]

    x = np.arange(len(metrics))
    axes[0].bar(x - 0.15, rule_vals, 0.3, label="Rule-based", color="#FF9800", alpha=0.8)
    axes[0].bar(x + 0.15, dl_vals, 0.3, label="EtchMind", color="#2196F3", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Rule-based vs EtchMind")
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Per-class F1
    class_rule_f1 = [rule_report[c]["f1-score"] for c in CLASS_NAMES]
    class_dl_f1 = [dl_report[c]["f1-score"] for c in CLASS_NAMES]

    x2 = np.arange(len(CLASS_NAMES))
    axes[1].bar(x2 - 0.15, class_rule_f1, 0.3, label="Rule-based", color="#FF9800", alpha=0.8)
    axes[1].bar(x2 + 0.15, class_dl_f1, 0.3, label="EtchMind", color="#2196F3", alpha=0.8)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([c[:12] for c in CLASS_NAMES], rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("Per-class F1 Comparison")
    axes[1].legend()
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig, {"rule": rule_report, "dl": dl_report}


# =========================================================================
# 전체 실행
# =========================================================================

def run_full_evaluation(
    checkpoint_path: str = "./checkpoints/best_sequence.pt",
    test_dir: str = "./features/test",
    output_dir: str = "./evaluation",
    device: str = "cpu",
):
    """5가지 분석 전체 실행"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_model(checkpoint_path, device)
    stats = ckpt["stats"]

    print("=" * 60)
    print("  EtchMind 종합 평가")
    print("=" * 60)

    # 1. 과적합 분석
    print("\n[1/5] 과적합 분석")
    fig = analyze_overfitting()
    fig.savefig(out / "1_overfitting.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → saved: 1_overfitting.png")

    # 2. Cross-recipe
    print("\n[2/5] Cross-recipe 일반화")
    fig, recipe_results = analyze_cross_recipe(model, test_dir, stats, device)
    fig.savefig(out / "2_cross_recipe.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → saved: 2_cross_recipe.png")

    # 3. 조기 감지
    print("\n[3/5] 조기 감지 능력 분석")
    fig, detection_results = analyze_early_detection(model, test_dir, stats, device)
    fig.savefig(out / "3_early_detection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → saved: 3_early_detection.png")

    # 4. Attention
    print("\n[4/5] Attention 분석")
    fig, attn_results = analyze_attention(model, test_dir, stats, device)
    fig.savefig(out / "4_attention.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → saved: 4_attention.png")

    # 5. 룰 기반 비교
    print("\n[5/5] 룰 기반 베이스라인 비교")
    fig, compare_results = compare_with_rule_based(model, test_dir, stats, device)
    fig.savefig(out / "5_rule_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → saved: 5_rule_comparison.png")

    print(f"\n{'='*60}")
    print(f"  평가 완료! 결과: {out.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_full_evaluation()
