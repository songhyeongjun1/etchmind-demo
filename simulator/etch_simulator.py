"""
EtchSimulator - 멀티 레시피 식각장비 센서 데이터 합성 생성기

물리 법칙 기반으로 정상 공정 Trace 데이터를 생성하고,
고장 시나리오를 주입하여 학습용 데이터를 만든다.
레시피별로 다른 baseline을 적용하여 일반화 가능한 데이터 생성.
"""

import numpy as np
from typing import Optional
from .config import (
    SENSOR_DEFS, OES_DEFS, FAULT_PARAMS, RECIPES, PROCESS_PHASES,
    TIMESTEPS, SAMPLING_RATE, PROCESS_SENSOR_NAMES, OES_NAMES,
    ALL_SENSOR_NAMES, GENERATION_CONFIG,
)


class EtchSimulator:
    """반도체 식각장비 센서 데이터 합성 생성기 (멀티 레시피)"""

    def __init__(self, recipe_name: str = "oxide_etch", seed: int = 42):
        """
        Args:
            recipe_name: RECIPES 키 (oxide_etch, poly_si_etch, metal_etch, harc_etch)
            seed: 랜덤 시드
        """
        if recipe_name not in RECIPES:
            raise ValueError(f"Unknown recipe: {recipe_name}. "
                             f"Available: {list(RECIPES.keys())}")

        self.recipe_name = recipe_name
        self.recipe = RECIPES[recipe_name]
        self.rng = np.random.default_rng(seed)

        self.timesteps = TIMESTEPS
        self.n_process = len(PROCESS_SENSOR_NAMES)
        self.n_oes = len(OES_NAMES)
        self.n_sensors = self.n_process + self.n_oes
        self.sensor_names = ALL_SENSOR_NAMES

        # 공정 단계별 시간 인덱스
        self.phase_indices = {}
        for phase_name, (start_s, end_s, _) in PROCESS_PHASES.items():
            self.phase_indices[phase_name] = (
                int(start_s * SAMPLING_RATE),
                int(end_s * SAMPLING_RATE),
            )

        # 레시피 기반 baseline/noise 배열 구축
        self._build_arrays()

    def _build_arrays(self):
        """레시피에 맞는 baseline, noise, phase_profile 배열 구축"""
        self.baselines = np.zeros(self.n_sensors)
        self.noise_stds = np.zeros(self.n_sensors)

        # 공정 센서
        for i, name in enumerate(PROCESS_SENSOR_NAMES):
            baseline = self.recipe["baselines"][name]
            noise_ratio = SENSOR_DEFS[name]["noise_ratio"]
            self.baselines[i] = baseline
            # baseline이 0인 센서(미사용 가스)는 노이즈도 0
            self.noise_stds[i] = abs(baseline) * noise_ratio if baseline != 0 else 0

        # OES 센서
        for j, name in enumerate(OES_NAMES):
            baseline = self.recipe["oes_baselines"][name]
            noise_ratio = OES_DEFS[name]["noise_ratio"]
            idx = self.n_process + j
            self.baselines[idx] = baseline
            self.noise_stds[idx] = baseline * noise_ratio if baseline > 0.005 else 0.001

        # Phase profile 행렬: (5 phases, n_sensors)
        phase_names = list(PROCESS_PHASES.keys())
        self.phase_profiles = np.zeros((len(phase_names), self.n_sensors))

        for i, name in enumerate(PROCESS_SENSOR_NAMES):
            self.phase_profiles[:, i] = SENSOR_DEFS[name]["phase_profile"]

        # OES: 플라즈마 off 시 0, on 시 1
        oes_phase = [0.0, 0.8, 1.0, 0.9, 0.0]
        for j in range(self.n_oes):
            self.phase_profiles[:, self.n_process + j] = oes_phase

    def _build_time_profile(self) -> np.ndarray:
        """단일 웨이퍼의 시간 프로파일 (600, 50)"""
        profile = np.zeros((self.timesteps, self.n_sensors))
        phase_list = list(PROCESS_PHASES.keys())

        for p_idx, phase_name in enumerate(phase_list):
            start_t, end_t = self.phase_indices[phase_name]
            length = end_t - start_t

            for s_idx in range(self.n_sensors):
                target_val = self.baselines[s_idx] * self.phase_profiles[p_idx, s_idx]

                if p_idx == 0:
                    t = np.linspace(0, 3, length)
                    ramp = target_val * (1 - np.exp(-t))
                    profile[start_t:end_t, s_idx] = ramp
                else:
                    prev_val = profile[start_t - 1, s_idx]
                    t = np.linspace(0, 4, length)
                    transition = prev_val + (target_val - prev_val) * (1 - np.exp(-t))
                    profile[start_t:end_t, s_idx] = transition

        return profile

    def generate_normal_wafer(self, wafer_variation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        정상 웨이퍼 1개의 trace 데이터 생성

        Returns:
            (600, 50) ndarray
        """
        profile = self._build_time_profile()

        if wafer_variation is None:
            w2w = GENERATION_CONFIG["wafer_to_wafer_variation"]
            wafer_variation = self.rng.normal(0, w2w, self.n_sensors)

        noise = self.rng.normal(0, 1, (self.timesteps, self.n_sensors)) * self.noise_stds
        trace = profile * (1 + wafer_variation) + noise

        return trace

    def generate_normal_batch(self, n_wafers: int) -> np.ndarray:
        """정상 웨이퍼 N개 배치. Returns (n_wafers, 600, 50)"""
        batch = np.zeros((n_wafers, self.timesteps, self.n_sensors), dtype=np.float32)
        w2w_std = GENERATION_CONFIG["wafer_to_wafer_variation"]

        for i in range(n_wafers):
            variation = self.rng.normal(0, w2w_std, self.n_sensors)
            batch[i] = self.generate_normal_wafer(variation)

        return batch

    def inject_fault(self, trace: np.ndarray, fault_type: str, severity: float) -> np.ndarray:
        """
        단일 웨이퍼 trace에 고장 효과 주입

        Args:
            trace: (600, 50)
            fault_type: FAULT_PARAMS 키
            severity: 0.0 ~ 1.0

        Returns:
            (600, 50)
        """
        if fault_type not in FAULT_PARAMS:
            raise ValueError(f"Unknown fault type: {fault_type}")

        params = FAULT_PARAMS[fault_type]
        result = trace.copy()

        for sensor_name, (direction, max_drift_ratio) in params["affected_sensors"].items():
            s_idx = ALL_SENSOR_NAMES.index(sensor_name)
            baseline = self.baselines[s_idx]

            # baseline이 0인 센서(미사용 가스)에는 고장 영향 없음 (단, OES 오염 피크는 예외)
            if baseline == 0 and sensor_name not in ("Al_394.4", "Cu_324.7"):
                continue

            # OES 오염 피크: baseline이 작아도 절대값으로 drift
            if sensor_name in ("Al_394.4", "Cu_324.7"):
                drift = max_drift_ratio * direction * (severity ** 1.5) * 0.01
                # metal_etch 레시피에서 Al은 이미 높으므로 상대적 drift
                if sensor_name == "Al_394.4" and baseline > 0.1:
                    drift = baseline * max_drift_ratio * 0.1 * direction * (severity ** 1.5)
            else:
                drift = baseline * max_drift_ratio * direction * (severity ** 1.5)

            # 플라즈마 on/off 마스크
            ign_start, _ = self.phase_indices["ignition"]
            _, over_end = self.phase_indices["over_etch"]
            plasma_mask = np.zeros(self.timesteps)
            plasma_mask[ign_start:over_end] = 1.0

            always_on_sensors = {
                "turbo_rpm", "turbo_bearing_current", "foreline_pressure",
                "base_pressure", "esc_leakage", "he_flow", "he_pressure",
                "particle_count", "edge_uniformity",
            }
            if sensor_name in always_on_sensors:
                plasma_mask[:] = 1.0

            result[:, s_idx] += drift * plasma_mask

            # 노이즈 증가
            noise_mult = params["noise_multiplier"]
            extra_noise_std = self.noise_stds[s_idx] * (noise_mult - 1) * severity
            if extra_noise_std > 0:
                extra_noise = self.rng.normal(0, extra_noise_std, self.timesteps) * plasma_mask
                result[:, s_idx] += extra_noise

        return result

    def generate_fault_sequence(
        self,
        fault_type: str,
        n_wafers: int,
        severity_start: float = 0.0,
        severity_end: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        고장 진행 시퀀스 생성

        Returns:
            data: (n_wafers, 600, 50)
            severities: (n_wafers,)
        """
        severities = np.linspace(severity_start, severity_end, n_wafers)
        data = np.zeros((n_wafers, self.timesteps, self.n_sensors), dtype=np.float32)
        w2w_std = GENERATION_CONFIG["wafer_to_wafer_variation"]

        for i in range(n_wafers):
            variation = self.rng.normal(0, w2w_std, self.n_sensors)
            normal_trace = self.generate_normal_wafer(variation)
            data[i] = self.inject_fault(normal_trace, fault_type, severities[i])

        return data, severities

    def generate_compound_fault(
        self,
        fault_types: list[str],
        severity_pairs: list[tuple[float, float]],
        n_wafers: int,
    ) -> tuple[np.ndarray, dict]:
        """복합 고장 시퀀스 생성"""
        all_severities = {}
        for ft, (s_start, s_end) in zip(fault_types, severity_pairs):
            all_severities[ft] = np.linspace(s_start, s_end, n_wafers)

        data = np.zeros((n_wafers, self.timesteps, self.n_sensors), dtype=np.float32)
        w2w_std = GENERATION_CONFIG["wafer_to_wafer_variation"]

        for i in range(n_wafers):
            variation = self.rng.normal(0, w2w_std, self.n_sensors)
            trace = self.generate_normal_wafer(variation)
            for ft in fault_types:
                trace = self.inject_fault(trace, ft, all_severities[ft][i])
            data[i] = trace

        return data, all_severities

    def get_sensor_info(self) -> dict:
        """센서 메타데이터 반환"""
        info = {}
        for name in PROCESS_SENSOR_NAMES:
            info[name] = {
                "type": "process",
                "unit": SENSOR_DEFS[name]["unit"],
                "baseline": self.baselines[ALL_SENSOR_NAMES.index(name)],
                "description": SENSOR_DEFS[name]["description"],
            }
        for name in OES_NAMES:
            info[name] = {
                "type": "oes",
                "wavelength_nm": OES_DEFS[name]["wavelength"],
                "species": OES_DEFS[name]["species"],
                "baseline": self.baselines[ALL_SENSOR_NAMES.index(name)],
                "role": OES_DEFS[name]["role"],
            }
        return info
