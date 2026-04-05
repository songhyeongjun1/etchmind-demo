"""
전처리: Wafer Trace (600, 50) → Feature Vector (250)

메인 식각 구간(7~45초)에서 센서별 5개 통계량 추출:
  mean, std, min, max, slope

결과: 490K wafers × 250 features → ~500MB (메모리/디스크 모두 가벼움)
"""

import numpy as np
from pathlib import Path
import time

# 메인 식각 구간 인덱스 (7s ~ 45s, 10Hz)
MAIN_START = 70
MAIN_END = 450
N_SENSORS = 50
N_STATS = 5  # mean, std, min, max, slope
N_FEATURES = N_SENSORS * N_STATS  # 250


def extract_features(trace: np.ndarray) -> np.ndarray:
    """
    단일 웨이퍼 trace → feature vector

    Args:
        trace: (600, 50) or (T, S)

    Returns:
        (250,) feature vector
    """
    main = trace[MAIN_START:MAIN_END, :]  # (380, 50)
    T = main.shape[0]

    means = main.mean(axis=0)                           # (50,)
    stds = main.std(axis=0)                             # (50,)
    mins = main.min(axis=0)                             # (50,)
    maxs = main.max(axis=0)                             # (50,)

    # slope: 선형 회귀 기울기 (시간에 대한 drift)
    t = np.arange(T, dtype=np.float32)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()
    slopes = ((t[:, None] - t_mean) * (main - means)).sum(axis=0) / (t_var + 1e-10)

    features = np.concatenate([means, stds, mins, maxs, slopes])  # (250,)
    return features


def extract_features_batch(traces: np.ndarray) -> np.ndarray:
    """
    배치 trace → feature matrix (벡터화)

    Args:
        traces: (N, 600, 50)

    Returns:
        (N, 250)
    """
    main = traces[:, MAIN_START:MAIN_END, :]  # (N, 380, 50)
    N, T, S = main.shape

    means = main.mean(axis=1)                             # (N, 50)
    stds = main.std(axis=1)                               # (N, 50)
    mins = main.min(axis=1)                               # (N, 50)
    maxs = main.max(axis=1)                               # (N, 50)

    t = np.arange(T, dtype=np.float32)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()
    # slopes: (N, 50)
    slopes = ((t[None, :, None] - t_mean) * (main - means[:, None, :])).sum(axis=1) / (t_var + 1e-10)

    features = np.concatenate([means, stds, mins, maxs, slopes], axis=1)  # (N, 250)
    return features.astype(np.float32)


def preprocess_dataset(data_dir: str, output_dir: str):
    """
    전체 데이터셋 전처리 (train/ 또는 test/)

    청크 파일을 읽어서 feature 추출 후 저장.
    디렉토리 구조 유지: recipe/category/chunk_XXXX.npz → features 포함 npz
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    recipes = np.load(data_path / "recipe_names.npy", allow_pickle=True)
    label_map = np.load(data_path / "label_map.npy", allow_pickle=True)

    # 메타데이터 복사
    np.save(out_path / "recipe_names.npy", recipes)
    np.save(out_path / "label_map.npy", label_map)
    np.save(out_path / "sensor_names.npy",
            np.load(data_path / "sensor_names.npy", allow_pickle=True))

    total_start = time.time()
    total_wafers = 0
    recipe_ids = {name: idx for idx, name in enumerate(recipes)}

    for recipe in recipes:
        recipe_dir = data_path / recipe
        recipe_out = out_path / recipe
        r_id = recipe_ids[recipe]

        for category_dir in sorted(recipe_dir.iterdir()):
            if not category_dir.is_dir():
                continue

            cat_out = recipe_out / category_dir.name
            cat_out.mkdir(parents=True, exist_ok=True)

            chunks = sorted(category_dir.glob("chunk_*.npz"))
            for chunk_path in chunks:
                d = np.load(chunk_path)
                traces = d["data"]         # (N, 600, 50)
                severity = d["severity"]   # (N,)
                fault_id = d["fault_id"]   # (N,)
                d.close()

                features = extract_features_batch(traces)  # (N, 250)
                recipe_id = np.full(len(features), r_id, dtype=np.int8)

                np.savez_compressed(
                    cat_out / chunk_path.name,
                    features=features,
                    severity=severity,
                    fault_id=fault_id,
                    recipe_id=recipe_id,
                )
                total_wafers += len(features)

        print(f"  {recipe}: done ({total_wafers:,} wafers so far)")

    elapsed = time.time() - total_start
    print(f"\n전처리 완료: {total_wafers:,} wafers, {elapsed:.1f}초")
    _print_size(out_path)
    return out_path


def _print_size(path):
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    print(f"디스크: {total / (1024**2):.1f} MB")


def load_all_features(feat_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    전처리된 전체 features를 메모리에 로드

    Returns:
        features: (N, 250)
        fault_ids: (N,)
        severities: (N,)
        recipe_ids: (N,)
    """
    path = Path(feat_dir)
    recipes = np.load(path / "recipe_names.npy", allow_pickle=True)

    all_feat, all_fid, all_sev, all_rid = [], [], [], []

    for recipe in recipes:
        recipe_dir = path / recipe
        for cat_dir in sorted(recipe_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            for chunk in sorted(cat_dir.glob("chunk_*.npz")):
                d = np.load(chunk)
                all_feat.append(d["features"])
                all_fid.append(d["fault_id"])
                all_sev.append(d["severity"])
                all_rid.append(d["recipe_id"])
                d.close()

    return (
        np.concatenate(all_feat),
        np.concatenate(all_fid),
        np.concatenate(all_sev),
        np.concatenate(all_rid),
    )


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/train"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./features/train"
    preprocess_dataset(data_dir, output_dir)
