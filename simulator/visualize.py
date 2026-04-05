"""
생성 데이터 검증 및 시각화 (멀티 레시피 대응)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from .config import (
    ALL_SENSOR_NAMES, PROCESS_SENSOR_NAMES, OES_NAMES,
    SENSOR_DEFS, OES_DEFS, RECIPES, FAULT_PARAMS,
    PROCESS_PHASES, SAMPLING_RATE,
)
from .generate import load_chunks

matplotlib.rcParams["font.family"] = "DejaVu Sans"

# 메인 식각 구간 인덱스
_MAIN_START = int(7 * SAMPLING_RATE)
_MAIN_END = int(45 * SAMPLING_RATE)


# =============================================================================
# 1. 레시피 간 비교
# =============================================================================

def plot_recipe_comparison_oes(data_dir: str, timestep: int = 300):
    """
    4개 레시피의 정상 웨이퍼 OES 스펙트럼 비교
    각 레시피에서 1개 웨이퍼를 뽑아 같은 시점의 OES를 겹쳐 그림
    """
    recipes = np.load(Path(data_dir) / "recipe_names.npy", allow_pickle=True)
    oes_start = len(PROCESS_SENSOR_NAMES)

    wavelengths = [OES_DEFS[name]["wavelength"] for name in OES_NAMES]
    species = [OES_DEFS[name]["species"] for name in OES_NAMES]

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(OES_NAMES))
    width = 0.2
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    for i, recipe in enumerate(recipes):
        data, _, _ = load_chunks(data_dir, recipe, "normal")
        # 첫 웨이퍼의 OES
        oes = data[0, timestep, oes_start:]
        ax.bar(x + i * width, oes, width, label=recipe, color=colors[i], alpha=0.8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"{s}\n{w:.0f}nm" for s, w in zip(species, wavelengths)],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(f"OES Spectrum Comparison Across Recipes (t={timestep/SAMPLING_RATE:.1f}s)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def plot_recipe_comparison_process(data_dir: str):
    """
    4개 레시피의 주요 공정 센서 비교 (메인 식각 구간 평균)
    """
    recipes = np.load(Path(data_dir) / "recipe_names.npy", allow_pickle=True)

    key_sensors = [
        "source_rf_fwd", "bias_rf_fwd", "dc_bias",
        "chamber_pressure", "etch_rate", "edge_uniformity",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Process Sensor Comparison Across Recipes (Main Etch Mean)", fontsize=14)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    for ax, sensor_name in zip(axes.flat, key_sensors):
        s_idx = ALL_SENSOR_NAMES.index(sensor_name)
        means = []
        stds = []

        for recipe in recipes:
            data, _, _ = load_chunks(data_dir, recipe, "normal")
            main_vals = data[:, _MAIN_START:_MAIN_END, s_idx].mean(axis=1)
            means.append(main_vals.mean())
            stds.append(main_vals.std())

        ax.bar(range(len(recipes)), means, yerr=stds, color=colors, alpha=0.8, capsize=5)
        ax.set_xticks(range(len(recipes)))
        ax.set_xticklabels(recipes, rotation=15, fontsize=9)
        unit = SENSOR_DEFS[sensor_name]["unit"]
        ax.set_title(f"{sensor_name} ({unit})", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


# =============================================================================
# 2. 고장 시그니처 강도 분석
# =============================================================================

def plot_fault_separability(data_dir: str, recipe: str = "oxide_etch"):
    """
    모든 고장 유형의 영향 센서별 정상-고장 분리도 분석
    SNR(Signal-to-Noise Ratio) = |mean_fault - mean_normal| / std_normal
    SNR이 높을수록 모델이 쉽게 구분 가능
    """
    # 정상 데이터 로드
    normal_data, _, _ = load_chunks(data_dir, recipe, "normal")
    normal_means = normal_data[:, _MAIN_START:_MAIN_END, :].mean(axis=1)  # (N, 50)
    normal_mean = normal_means.mean(axis=0)
    normal_std = normal_means.std(axis=0)
    normal_std[normal_std == 0] = 1e-10  # 0 방지

    fault_names = list(FAULT_PARAMS.keys())
    all_snrs = {}

    for fault_name in fault_names:
        fault_data, sev, _ = load_chunks(data_dir, recipe, fault_name)
        # severity > 0.5인 것만 (심각한 고장)
        mask = sev > 0.5
        if mask.sum() < 10:
            mask = sev > 0.3
        fault_subset = fault_data[mask]
        fault_means = fault_subset[:, _MAIN_START:_MAIN_END, :].mean(axis=1)
        fault_mean = fault_means.mean(axis=0)

        snr = np.abs(fault_mean - normal_mean) / normal_std
        affected = FAULT_PARAMS[fault_name]["affected_sensors"]
        all_snrs[fault_name] = {
            sensor: snr[ALL_SENSOR_NAMES.index(sensor)]
            for sensor in affected
            if sensor in ALL_SENSOR_NAMES
        }

    # 시각화: 고장별 영향 센서의 SNR
    fig, ax = plt.subplots(figsize=(16, 8))

    y_pos = 0
    y_ticks = []
    y_labels = []
    colors_map = {
        "focus_ring_wear": "#2196F3",
        "electrode_wear": "#4CAF50",
        "polymer_contamination": "#FF9800",
        "metal_contamination": "#E91E63",
        "esc_degradation": "#9C27B0",
        "pump_degradation": "#795548",
    }

    for fault_name in fault_names:
        color = colors_map.get(fault_name, "gray")
        for sensor, snr_val in sorted(all_snrs[fault_name].items(),
                                       key=lambda x: -x[1]):
            ax.barh(y_pos, snr_val, color=color, alpha=0.8, height=0.7)
            ax.text(snr_val + 0.1, y_pos, f"{snr_val:.1f}", va="center", fontsize=8)
            y_ticks.append(y_pos)
            y_labels.append(f"{fault_name[:12]}:{sensor}")
            y_pos += 1
        y_pos += 0.5  # 고장 간 간격

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("SNR (|fault_mean - normal_mean| / normal_std)")
    ax.set_title(f"Fault Separability by Sensor — {recipe}\n(SNR > 3: 쉬움, 1~3: 보통, < 1: 어려움)")
    ax.axvline(x=1, color="orange", linestyle="--", alpha=0.5, label="SNR=1")
    ax.axvline(x=3, color="red", linestyle="--", alpha=0.5, label="SNR=3")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def plot_focus_ring_vs_electrode(data_dir: str, recipe: str = "oxide_etch"):
    """
    핵심 챌린지: Focus Ring vs 전극 마모의 센서 분포 비교
    두 고장이 공유하는 센서(dc_bias, Si_288.2 등)에서 분포가 겹치는지 확인
    """
    normal_data, _, _ = load_chunks(data_dir, recipe, "normal")
    ring_data, ring_sev, _ = load_chunks(data_dir, recipe, "focus_ring_wear")
    elec_data, elec_sev, _ = load_chunks(data_dir, recipe, "electrode_wear")

    # severity > 0.5만
    ring_mask = ring_sev > 0.5
    elec_mask = elec_sev > 0.5
    ring_subset = ring_data[ring_mask]
    elec_subset = elec_data[elec_mask]

    # 공유 영향 센서
    shared_sensors = ["dc_bias", "Si_288.2", "match_pos_c1", "etch_rate",
                      "edge_uniformity", "particle_count"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Focus Ring vs Electrode Wear Distribution — {recipe}\n"
                 f"(severity > 0.5, 겹칠수록 모델이 구분하기 어려움)", fontsize=13)

    for ax, sensor in zip(axes.flat, shared_sensors):
        s_idx = ALL_SENSOR_NAMES.index(sensor)

        normal_vals = normal_data[:200, _MAIN_START:_MAIN_END, s_idx].mean(axis=1)
        ring_vals = ring_subset[:, _MAIN_START:_MAIN_END, s_idx].mean(axis=1)
        elec_vals = elec_subset[:, _MAIN_START:_MAIN_END, s_idx].mean(axis=1)

        ax.hist(normal_vals, bins=30, alpha=0.5, color="green", label="Normal", density=True)
        ax.hist(ring_vals, bins=30, alpha=0.5, color="blue", label="Focus Ring", density=True)
        ax.hist(elec_vals, bins=30, alpha=0.5, color="red", label="Electrode", density=True)

        ax.set_title(sensor, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# 기본 시각화 (단일 레시피)
# =============================================================================

def plot_single_wafer_trace(data: np.ndarray, recipe_name: str = "oxide_etch",
                            title: str = None):
    """단일 웨이퍼 주요 센서 trace. data: (600, 50)"""
    time_s = np.arange(data.shape[0]) / SAMPLING_RATE

    key_sensors = [
        ("source_rf_fwd", "Source RF (W)"),
        ("dc_bias", "DC Bias (V)"),
        ("chamber_pressure", "Pressure (mTorr)"),
        ("etch_rate", "Etch Rate (A/min)"),
        ("he_flow", "He Flow (sccm)"),
        ("turbo_bearing_current", "Bearing Current (A)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(title or f"Wafer Trace — {recipe_name}", fontsize=14)

    colors = ["#e8f5e9", "#fff3e0", "#e3f2fd", "#fce4ec", "#f3e5f5"]
    for ax, (sensor_name, label) in zip(axes.flat, key_sensors):
        idx = ALL_SENSOR_NAMES.index(sensor_name)
        ax.plot(time_s, data[:, idx], linewidth=0.8)
        ax.set_ylabel(label)
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

        for (phase_name, (start, end, desc)), c in zip(PROCESS_PHASES.items(), colors):
            ax.axvspan(start, end, alpha=0.2, color=c)

    plt.tight_layout()
    return fig


def plot_oes_spectrum(data: np.ndarray, timestep: int = 300, title: str = "OES Spectrum"):
    """특정 시점 OES 스펙트럼. data: (600, 50)"""
    oes_start = len(PROCESS_SENSOR_NAMES)
    oes_data = data[timestep, oes_start:]

    wavelengths = [OES_DEFS[name]["wavelength"] for name in OES_NAMES]
    species = [OES_DEFS[name]["species"] for name in OES_NAMES]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(OES_NAMES)), oes_data, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(OES_NAMES)))
    ax.set_xticklabels([f"{s}\n{w:.0f}nm" for s, w in zip(species, wavelengths)],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(f"{title} (t={timestep/SAMPLING_RATE:.1f}s)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def plot_normal_vs_fault_trace(normal_trace: np.ndarray, fault_trace: np.ndarray,
                                fault_name: str):
    """정상 vs 고장 웨이퍼 trace 비교"""
    affected = list(FAULT_PARAMS[fault_name]["affected_sensors"].keys())[:6]
    time_s = np.arange(normal_trace.shape[0]) / SAMPLING_RATE

    n_sensors = len(affected)
    n_cols = min(3, n_sensors)
    n_rows = (n_sensors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    if n_sensors == 1:
        axes = np.array([axes])
    axes = axes.flat

    fig.suptitle(f"Normal vs {fault_name} (severity=1.0)", fontsize=13)

    for ax, sensor_name in zip(axes, affected):
        s_idx = ALL_SENSOR_NAMES.index(sensor_name)
        ax.plot(time_s, normal_trace[:, s_idx], "g-", alpha=0.7, linewidth=0.8, label="Normal")
        ax.plot(time_s, fault_trace[:, s_idx], "r-", alpha=0.7, linewidth=0.8, label="Fault")
        ax.set_title(sensor_name, fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in list(axes)[n_sensors:]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig


# =============================================================================
# Sanity Check
# =============================================================================

def run_sanity_check(data_dir: str):
    """멀티 레시피 데이터 검증"""
    path = Path(data_dir)
    recipes = np.load(path / "recipe_names.npy", allow_pickle=True)
    label_map = np.load(path / "label_map.npy", allow_pickle=True).item()

    print(f"레시피: {list(recipes)}")
    print(f"라벨: {label_map}")
    print("=" * 60)

    for recipe in recipes:
        print(f"\n[{recipe}]")
        recipe_dir = path / recipe
        for subdir in sorted(recipe_dir.iterdir()):
            if subdir.is_dir():
                chunks = sorted(subdir.glob("chunk_*.npz"))
                if not chunks:
                    continue
                # 첫 청크만 검증
                d = np.load(chunks[0])
                data = d["data"]
                total = sum(np.load(c)["data"].shape[0] for c in chunks)
                has_nan = np.isnan(data).any()
                has_inf = np.isinf(data).any()
                print(f"  {subdir.name}: {total:,} wafers, "
                      f"shape_per_chunk={data.shape}, NaN={has_nan}, Inf={has_inf}")
                d.close()


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data_test"
    run_sanity_check(data_dir)
