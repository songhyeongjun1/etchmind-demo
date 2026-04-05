"""
멀티 레시피 데이터 생성 스크립트
4개 레시피 × 각 ~102K = 총 ~408K 웨이퍼
배치별 디스크 직접 저장 → 메모리 ~200MB로 충분

사용법:
    python -m simulator.generate --output_dir ./data
    python -m simulator.generate --test --output_dir ./data_test
"""

import argparse
import time
import numpy as np
from pathlib import Path

from .etch_simulator import EtchSimulator
from .config import (
    GENERATION_CONFIG, FAULT_PARAMS, RECIPES, RECIPE_NAMES,
    ALL_SENSOR_NAMES, TIMESTEPS,
)


def generate_all(
    output_dir: str = "./data",
    batch_size: int = 1000,
    seed: int = 42,
    recipes: list[str] | None = None,
    normal_wafers: int | None = None,
    fault_wafers: dict | None = None,
    compound_wafers: int | None = None,
):
    """
    멀티 레시피 전체 데이터셋 생성

    Args:
        output_dir: 저장 디렉토리
        batch_size: 배치 크기
        seed: 랜덤 시드
        recipes: 생성할 레시피 목록 (None이면 전체)
        normal_wafers: 레시피당 정상 웨이퍼 수
        fault_wafers: 레시피당 고장별 웨이퍼 수
        compound_wafers: 레시피당 복합 고장 웨이퍼 수
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    recipe_list = recipes or RECIPE_NAMES
    cfg = GENERATION_CONFIG["per_recipe"]
    n_normal = normal_wafers or cfg["normal_wafers"]
    n_fault = fault_wafers or cfg["fault_wafers"]
    n_compound = compound_wafers or cfg["compound_wafers"]

    total_start = time.time()

    # 메타데이터 저장
    np.save(output_path / "sensor_names.npy", np.array(ALL_SENSOR_NAMES))
    print(f"센서 수: {len(ALL_SENSOR_NAMES)}개, Timesteps: {TIMESTEPS}")
    print(f"레시피: {recipe_list}")
    per_recipe = n_normal + sum(n_fault.values()) + n_compound
    print(f"레시피당 {per_recipe:,}개 × {len(recipe_list)}개 = 총 {per_recipe * len(recipe_list):,}개")
    print(f"출력: {output_path.absolute()}")
    print("=" * 60)

    fault_type_names = list(n_fault.keys())
    fault_type_ids = {name: idx for idx, name in enumerate(fault_type_names)}

    compound_combos = [
        (["focus_ring_wear", "polymer_contamination"], "ring+polymer"),
        (["electrode_wear", "polymer_contamination"], "electrode+polymer"),
        (["focus_ring_wear", "esc_degradation"], "ring+esc"),
        (["pump_degradation", "polymer_contamination"], "pump+polymer"),
        (["electrode_wear", "esc_degradation"], "electrode+esc"),
    ]

    # 레시피별 생성
    for r_idx, recipe_name in enumerate(recipe_list):
        recipe_desc = RECIPES[recipe_name]["description"]
        print(f"\n{'='*60}")
        print(f"[레시피 {r_idx+1}/{len(recipe_list)}] {recipe_name}: {recipe_desc}")
        print(f"{'='*60}")

        # 레시피마다 시드를 다르게 (seed + recipe_index)
        recipe_seed = seed + r_idx * 1000
        sim = EtchSimulator(recipe_name=recipe_name, seed=recipe_seed)

        recipe_dir = output_path / recipe_name

        # 1. 정상
        print(f"\n  [1/3] 정상 데이터: {n_normal:,}개")
        _generate_normal_chunked(sim, recipe_dir, n_normal, batch_size)

        # 2. 단일 고장
        print(f"\n  [2/3] 고장 데이터: {len(n_fault)}개 유형")
        for fault_name, n_w in n_fault.items():
            fault_id = fault_type_ids[fault_name]
            print(f"    → {fault_name} ({n_w:,}개)")
            _generate_fault_chunked(sim, recipe_dir, fault_name, fault_id,
                                    n_w, batch_size)

        # 3. 복합 고장
        per_combo = n_compound // len(compound_combos)
        print(f"\n  [3/3] 복합 고장: {n_compound:,}개")
        for combo_idx, (faults, combo_name) in enumerate(compound_combos):
            compound_id = 100 + combo_idx
            print(f"    → {combo_name} ({per_combo:,}개)")
            _generate_compound_chunked(sim, recipe_dir, faults, combo_name,
                                       compound_id, per_combo, batch_size)

    # 라벨/메타 저장
    label_map = {-1: "normal"}
    for name, idx in fault_type_ids.items():
        label_map[idx] = name
    for i, (faults, combo_name) in enumerate(compound_combos):
        label_map[100 + i] = combo_name

    np.save(output_path / "label_map.npy", label_map)
    np.save(output_path / "recipe_names.npy", np.array(recipe_list))
    _save_manifest(output_path, recipe_list)

    elapsed = time.time() - total_start
    total_wafers = per_recipe * len(recipe_list)
    print(f"\n{'='*60}")
    print(f"완료! 총 {total_wafers:,}개 웨이퍼, {elapsed/60:.1f}분 소요")
    _print_disk_usage(output_path)

    return output_path


# =============================================================================
# 청크 단위 생성 함수들
# =============================================================================

def _generate_normal_chunked(sim, recipe_dir, n_total, batch_size):
    chunk_dir = recipe_dir / "normal"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    n_batches = (n_total + batch_size - 1) // batch_size
    total_saved = 0

    for b in range(n_batches):
        bs = min(batch_size, n_total - b * batch_size)
        data = sim.generate_normal_batch(bs)
        severities = np.zeros(bs, dtype=np.float32)
        fault_ids = np.full(bs, -1, dtype=np.int8)

        np.savez_compressed(
            chunk_dir / f"chunk_{b:04d}.npz",
            data=data, severity=severities, fault_id=fault_ids,
        )
        total_saved += bs
        del data

        if (b + 1) % 20 == 0 or b == n_batches - 1:
            print(f"      {total_saved:,}/{n_total:,} ({100*total_saved/n_total:.0f}%)")


def _generate_fault_chunked(sim, recipe_dir, fault_name, fault_id,
                            n_total, batch_size):
    chunk_dir = recipe_dir / fault_name
    chunk_dir.mkdir(parents=True, exist_ok=True)

    lifetime = FAULT_PARAMS[fault_name]["lifetime_wafers"]
    n_cycles = max(1, n_total // lifetime)
    wafers_per_cycle = n_total // n_cycles
    remainder = n_total - wafers_per_cycle * n_cycles

    chunk_idx = 0
    total_saved = 0

    for cycle in range(n_cycles):
        n_this_cycle = wafers_per_cycle + (1 if cycle < remainder else 0)

        sev_start = sim.rng.uniform(0.0, 0.3)
        sev_end = sim.rng.uniform(0.7, 1.0)
        severities_full = np.linspace(sev_start, sev_end, n_this_cycle)

        for offset in range(0, n_this_cycle, batch_size):
            bs = min(batch_size, n_this_cycle - offset)
            sevs = severities_full[offset:offset + bs]

            data, _ = sim.generate_fault_sequence(
                fault_name, bs,
                severity_start=sevs[0], severity_end=sevs[-1],
            )
            fault_ids = np.full(bs, fault_id, dtype=np.int8)

            np.savez_compressed(
                chunk_dir / f"chunk_{chunk_idx:04d}.npz",
                data=data, severity=sevs.astype(np.float32),
                fault_id=fault_ids,
            )
            chunk_idx += 1
            total_saved += bs
            del data

    print(f"      {total_saved:,}/{n_total:,} (100%)")


def _generate_compound_chunked(sim, recipe_dir, fault_types, combo_name,
                                compound_id, n_total, batch_size):
    chunk_dir = recipe_dir / f"compound_{combo_name}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    sev_pairs = [
        (sim.rng.uniform(0.1, 0.4), sim.rng.uniform(0.6, 1.0))
        for _ in fault_types
    ]

    chunk_idx = 0
    total_saved = 0

    for offset in range(0, n_total, batch_size):
        bs = min(batch_size, n_total - offset)

        t_start = offset / max(n_total, 1)
        t_end = (offset + bs) / max(n_total, 1)
        batch_sev_pairs = [
            (s + (e - s) * t_start, s + (e - s) * t_end)
            for s, e in sev_pairs
        ]

        data, sev_dict = sim.generate_compound_fault(fault_types, batch_sev_pairs, bs)
        max_sev = np.max([sev_dict[f] for f in fault_types], axis=0)
        fault_ids = np.full(bs, compound_id, dtype=np.int8)

        np.savez_compressed(
            chunk_dir / f"chunk_{chunk_idx:04d}.npz",
            data=data, severity=max_sev.astype(np.float32),
            fault_id=fault_ids,
        )
        chunk_idx += 1
        total_saved += bs
        del data

    print(f"      {total_saved:,}/{n_total:,} (100%)")


# =============================================================================
# 유틸리티
# =============================================================================

def _save_manifest(output_path, recipe_list):
    manifest = {}
    for recipe in recipe_list:
        recipe_dir = output_path / recipe
        manifest[recipe] = {}
        for subdir in sorted(recipe_dir.iterdir()):
            if subdir.is_dir():
                chunks = sorted(subdir.glob("chunk_*.npz"))
                if chunks:
                    total = sum(np.load(c)["data"].shape[0] for c in chunks)
                    manifest[recipe][subdir.name] = {
                        "n_chunks": len(chunks),
                        "n_wafers": total,
                    }
    np.save(output_path / "manifest.npy", manifest)

    # 요약 출력
    for recipe, categories in manifest.items():
        total = sum(c["n_wafers"] for c in categories.values())
        print(f"  {recipe}: {total:,}개 ({len(categories)} categories)")


def _print_disk_usage(output_path):
    total_bytes = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    print(f"총 디스크: {total_bytes / (1024**3):.2f} GB")


def load_chunks(data_dir: str, recipe: str, category: str):
    """청크 로드 (소규모 분석용). 대규모는 PyTorch Dataset 사용."""
    chunk_dir = Path(data_dir) / recipe / category
    chunks = sorted(chunk_dir.glob("chunk_*.npz"))
    all_data, all_sev, all_fid = [], [], []
    for c in chunks:
        d = np.load(c)
        all_data.append(d["data"])
        all_sev.append(d["severity"])
        all_fid.append(d["fault_id"])
        d.close()
    return (np.concatenate(all_data),
            np.concatenate(all_sev),
            np.concatenate(all_fid))


# =============================================================================
# 테스트 / 프리셋
# =============================================================================

def generate_test(output_dir: str = "./data_test"):
    """소규모 테스트 (레시피당 300개, 총 ~1.2K)"""
    return generate_all(
        output_dir=output_dir,
        batch_size=100,
        normal_wafers=200,
        fault_wafers={k: 50 for k in FAULT_PARAMS},
        compound_wafers=50,
    )


def generate_train_and_test(output_dir: str = "./data"):
    """학습 + 테스트 데이터 생성 (다른 시드)"""
    print("=" * 60)
    print("  TRAIN SET 생성 (seed=42)")
    print("=" * 60)
    generate_all(
        output_dir=f"{output_dir}/train",
        seed=GENERATION_CONFIG["train_seed"],
    )

    print("\n\n")
    print("=" * 60)
    print("  TEST SET 생성 (seed=99)")
    print("=" * 60)
    # 테스트는 학습의 20% 규모
    cfg = GENERATION_CONFIG["per_recipe"]
    generate_all(
        output_dir=f"{output_dir}/test",
        seed=GENERATION_CONFIG["test_seed"],
        normal_wafers=cfg["normal_wafers"] // 5,
        fault_wafers={k: v // 5 for k, v in cfg["fault_wafers"].items()},
        compound_wafers=cfg["compound_wafers"] // 5,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EtchMind 멀티 레시피 합성 데이터 생성")
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true", help="소규모 테스트 생성")
    parser.add_argument("--full", action="store_true", help="학습+테스트 전체 생성")
    args = parser.parse_args()

    if args.test:
        generate_test(args.output_dir)
    elif args.full:
        generate_train_and_test(args.output_dir)
    else:
        generate_all(args.output_dir, args.batch_size, args.seed)
