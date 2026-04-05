"""
PyTorch Dataset: 전처리된 features에서 시퀀스 구성

두 가지 모드:
1. SingleWafer: 웨이퍼 1개 단위 (250 features) → 빠른 프로토타이핑
2. WaferSequence: 최근 W개 웨이퍼 시퀀스 (W, 250) → drift 감지
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# 7-class 라벨 매핑 (fault_id → class index)
# fault_id: -1=normal, 0~5=단일고장, 100~104=복합고장
FAULT_ID_TO_CLASS = {
    -1: 0,   # normal
    0: 1,    # focus_ring_wear
    1: 2,    # electrode_wear
    2: 3,    # polymer_contamination
    3: 4,    # metal_contamination
    4: 5,    # esc_degradation
    5: 6,    # pump_degradation
}

# 복합 고장 → 주요 고장으로 매핑 (학습 시 primary fault 기준)
COMPOUND_TO_PRIMARY = {
    100: 1,  # ring+polymer → focus_ring
    101: 2,  # electrode+polymer → electrode
    102: 1,  # ring+esc → focus_ring
    103: 6,  # pump+polymer → pump
    104: 2,  # electrode+esc → electrode
}

CLASS_NAMES = [
    "normal", "focus_ring_wear", "electrode_wear",
    "polymer_contamination", "metal_contamination",
    "esc_degradation", "pump_degradation",
]

N_CLASSES = len(CLASS_NAMES)


def fault_id_to_class(fault_id: int) -> int:
    """fault_id를 7-class index로 변환"""
    if fault_id in FAULT_ID_TO_CLASS:
        return FAULT_ID_TO_CLASS[fault_id]
    elif fault_id in COMPOUND_TO_PRIMARY:
        return COMPOUND_TO_PRIMARY[fault_id]
    else:
        return 0  # unknown → normal


class SingleWaferDataset(Dataset):
    """
    웨이퍼 1개 단위 Dataset (빠른 프로토타이핑용)

    입력: (250,) features
    출력: class_label (0~6), severity (0~1), recipe_id (0~3)
    """

    def __init__(self, feat_dir: str, normalize: bool = True,
                 stats: dict | None = None):
        """
        Args:
            feat_dir: features/train 또는 features/test
            normalize: 정규화 여부
            stats: 외부 정규화 통계 {mean, std}. None이면 자체 계산
        """
        from model.preprocess import load_all_features

        self.features, fault_ids, self.severities, self.recipe_ids = \
            load_all_features(feat_dir)

        # fault_id → 7-class
        self.labels = np.array([fault_id_to_class(fid) for fid in fault_ids],
                               dtype=np.int64)

        # 정규화
        if normalize:
            if stats is None:
                self.mean = self.features.mean(axis=0)
                self.std = self.features.std(axis=0)
                self.std[self.std < 1e-8] = 1.0  # 0-std 방지 (미사용 가스)
            else:
                self.mean = stats["mean"]
                self.std = stats["std"]
            self.features = (self.features - self.mean) / self.std
        else:
            self.mean = np.zeros(self.features.shape[1])
            self.std = np.ones(self.features.shape[1])

    def get_stats(self) -> dict:
        """정규화 통계 반환 (테스트셋에 동일 적용용)"""
        return {"mean": self.mean, "std": self.std}

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "severity": torch.tensor(self.severities[idx], dtype=torch.float32),
            "recipe_id": torch.tensor(self.recipe_ids[idx], dtype=torch.long),
        }


class WaferSequenceDataset(Dataset):
    """
    최근 W개 웨이퍼 시퀀스 Dataset (drift 감지용)

    같은 recipe + category 내에서 연속 W개를 묶음.
    라벨은 시퀀스의 마지막 웨이퍼 기준.

    입력: (W, 250) feature sequence
    출력: class_label, severity, recipe_id
    """

    def __init__(self, feat_dir: str, window_size: int = 32,
                 stride: int = 8, normalize: bool = True,
                 stats: dict | None = None):
        from model.preprocess import load_all_features

        features, fault_ids, severities, recipe_ids = \
            load_all_features(feat_dir)

        labels = np.array([fault_id_to_class(fid) for fid in fault_ids],
                          dtype=np.int64)

        # 정규화
        if normalize:
            if stats is None:
                self.mean = features.mean(axis=0)
                self.std = features.std(axis=0)
                self.std[self.std < 1e-8] = 1.0
            else:
                self.mean = stats["mean"]
                self.std = stats["std"]
            features = (features - self.mean) / self.std
        else:
            self.mean = np.zeros(features.shape[1])
            self.std = np.ones(features.shape[1])

        # 시퀀스 구성: 같은 (recipe, category) 내에서 sliding window
        # 데이터가 이미 recipe/category 순서로 정렬되어 있으므로
        # recipe_id와 label이 바뀌는 지점에서 시퀀스를 끊음
        self.sequences = []     # (W, 250)
        self.seq_labels = []    # int
        self.seq_severities = []  # float
        self.seq_recipe_ids = []  # int

        # 같은 (recipe_id, label) 연속 구간 찾기
        group_key = recipe_ids * 100 + labels  # 간단한 그룹 키
        change_points = np.where(np.diff(group_key) != 0)[0] + 1
        segment_starts = np.concatenate([[0], change_points])
        segment_ends = np.concatenate([change_points, [len(features)]])

        for seg_start, seg_end in zip(segment_starts, segment_ends):
            seg_len = seg_end - seg_start
            if seg_len < window_size:
                continue

            for i in range(0, seg_len - window_size + 1, stride):
                start = seg_start + i
                end = start + window_size
                self.sequences.append(features[start:end])
                self.seq_labels.append(labels[end - 1])
                self.seq_severities.append(severities[end - 1])
                self.seq_recipe_ids.append(recipe_ids[end - 1])

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.seq_labels = np.array(self.seq_labels, dtype=np.int64)
        self.seq_severities = np.array(self.seq_severities, dtype=np.float32)
        self.seq_recipe_ids = np.array(self.seq_recipe_ids, dtype=np.int64)

        print(f"  시퀀스 구성: {len(self.sequences):,}개 "
              f"(window={window_size}, stride={stride})")

    def get_stats(self):
        return {"mean": self.mean, "std": self.std}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.sequences[idx], dtype=torch.float32),
            "label": torch.tensor(self.seq_labels[idx], dtype=torch.long),
            "severity": torch.tensor(self.seq_severities[idx], dtype=torch.float32),
            "recipe_id": torch.tensor(self.seq_recipe_ids[idx], dtype=torch.long),
        }


def create_dataloaders(
    train_dir: str = "./features/train",
    test_dir: str = "./features/test",
    mode: str = "single",   # "single" or "sequence"
    batch_size: int = 256,
    window_size: int = 32,
    stride: int = 8,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    학습/테스트 DataLoader 생성

    Returns:
        train_loader, test_loader, stats
    """
    if mode == "single":
        train_ds = SingleWaferDataset(train_dir, normalize=True)
        stats = train_ds.get_stats()
        test_ds = SingleWaferDataset(test_dir, normalize=True, stats=stats)
    elif mode == "sequence":
        train_ds = WaferSequenceDataset(train_dir, window_size, stride,
                                         normalize=True)
        stats = train_ds.get_stats()
        test_ds = WaferSequenceDataset(test_dir, window_size, stride,
                                        normalize=True, stats=stats)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"\nDataLoader 생성 완료:")
    print(f"  Train: {len(train_ds):,} samples, {len(train_loader)} batches")
    print(f"  Test:  {len(test_ds):,} samples, {len(test_loader)} batches")

    return train_loader, test_loader, stats
