"""
센서 정의, 멀티 레시피 설정, 고장 파라미터
모든 값은 플라즈마 물리 문헌 및 장비 스펙 기반

센서 30개 (공정) + 20개 (OES) = 50개
레시피 4개: SiO2 Etch, Poly-Si Etch, Metal(Al) Etch, HARC Etch
"""

import numpy as np

# =============================================================================
# 공정 단계 정의 (모든 레시피 공통 구조)
# =============================================================================
PROCESS_PHASES = {
    # (시작초, 종료초, 설명)
    "gas_stabilization": (0, 5, "가스 안정화"),
    "ignition":          (5, 7, "플라즈마 점화"),
    "main_etch":         (7, 45, "메인 식각"),
    "over_etch":         (45, 55, "오버 식각"),
    "purge":             (55, 60, "퍼지/펌프다운"),
}

SAMPLING_RATE = 10       # Hz
PROCESS_TIME = 60        # 초
TIMESTEPS = PROCESS_TIME * SAMPLING_RATE  # 600


# =============================================================================
# 센서 구조 정의 (30개 공정 센서)
# =============================================================================
# noise_ratio: baseline 대비 노이즈 비율 (예: 0.01 = 1%)
# phase_profile: 공정 단계별 baseline 비율 [gas_stab, ignition, main, over, purge]
# 실제 baseline 값은 레시피별로 다름

SENSOR_DEFS = {
    # --- RF / 전력 (7개) ---
    "source_rf_fwd":      {"noise_ratio": 0.010, "unit": "W",     "phase_profile": [0.0, 0.9, 1.0, 0.7, 0.0],  "description": "Source RF Forward Power"},
    "source_rf_ref":       {"noise_ratio": 0.40,  "unit": "W",     "phase_profile": [0.0, 0.4, 1.0, 1.0, 0.0],  "description": "Source RF Reflected Power"},
    "bias_rf_fwd":         {"noise_ratio": 0.010, "unit": "W",     "phase_profile": [0.0, 0.8, 1.0, 0.5, 0.0],  "description": "Bias RF Forward Power"},
    "bias_rf_ref":         {"noise_ratio": 0.50,  "unit": "W",     "phase_profile": [0.0, 0.3, 1.0, 1.0, 0.0],  "description": "Bias RF Reflected Power"},
    "dc_bias":             {"noise_ratio": 0.010, "unit": "V",     "phase_profile": [0.0, 0.7, 1.0, 0.5, 0.0],  "description": "DC Self-Bias Voltage"},
    "match_pos_c1":        {"noise_ratio": 0.010, "unit": "%",     "phase_profile": [0.5, 0.8, 1.0, 0.95, 0.5], "description": "Matching Network C1 Position"},
    "match_pos_c2":        {"noise_ratio": 0.011, "unit": "%",     "phase_profile": [0.5, 0.85, 1.0, 0.93, 0.5],"description": "Matching Network C2 Position"},

    # --- 가스 (8개: 다양한 레시피 커버) ---
    "cf4_flow":            {"noise_ratio": 0.010, "unit": "sccm",  "phase_profile": [0.0, 1.0, 1.0, 0.8, 0.0],  "description": "CF4 Flow Rate"},
    "chf3_flow":           {"noise_ratio": 0.010, "unit": "sccm",  "phase_profile": [0.0, 1.0, 1.0, 1.2, 0.0],  "description": "CHF3 Flow Rate"},
    "o2_flow":             {"noise_ratio": 0.013, "unit": "sccm",  "phase_profile": [0.0, 1.0, 1.0, 0.5, 0.0],  "description": "O2 Flow Rate"},
    "ar_flow":             {"noise_ratio": 0.010, "unit": "sccm",  "phase_profile": [0.0, 1.0, 1.0, 1.0, 0.0],  "description": "Ar Flow Rate"},
    "cl2_flow":            {"noise_ratio": 0.010, "unit": "sccm",  "phase_profile": [0.0, 1.0, 1.0, 0.8, 0.0],  "description": "Cl2 Flow Rate"},
    "bcl3_flow":           {"noise_ratio": 0.010, "unit": "sccm",  "phase_profile": [0.0, 1.0, 1.0, 0.9, 0.0],  "description": "BCl3 Flow Rate"},
    "c4f8_flow":           {"noise_ratio": 0.010, "unit": "sccm",  "phase_profile": [0.0, 1.0, 1.0, 1.1, 0.0],  "description": "C4F8 Flow Rate"},
    "n2_flow":             {"noise_ratio": 0.013, "unit": "sccm",  "phase_profile": [0.0, 1.0, 1.0, 1.0, 0.0],  "description": "N2 Flow Rate"},

    # --- 압력 (4개) ---
    "chamber_pressure":    {"noise_ratio": 0.020, "unit": "mTorr", "phase_profile": [0.05, 0.85, 1.0, 0.9, 0.05], "description": "Chamber Pressure"},
    "base_pressure":       {"noise_ratio": 0.067, "unit": "mTorr", "phase_profile": [1.0, 1.0, 1.0, 1.0, 1.0],   "description": "Base Pressure"},
    "foreline_pressure":   {"noise_ratio": 0.067, "unit": "Torr",  "phase_profile": [0.8, 1.0, 1.0, 1.0, 0.8],   "description": "Foreline Pressure"},
    "throttle_valve":      {"noise_ratio": 0.009, "unit": "%",     "phase_profile": [0.1, 0.9, 1.0, 1.05, 0.1],  "description": "Throttle Valve Position"},

    # --- 온도 (2개) ---
    "chuck_temp_center":   {"noise_ratio": 0.008, "unit": "°C",    "phase_profile": [0.95, 0.97, 1.0, 1.01, 1.005], "description": "Chuck Temperature Center"},
    "chuck_temp_edge":     {"noise_ratio": 0.007, "unit": "°C",    "phase_profile": [0.95, 0.97, 1.0, 1.01, 1.005], "description": "Chuck Temperature Edge"},

    # --- ESC (4개) ---
    "esc_voltage":         {"noise_ratio": 0.003, "unit": "V",     "phase_profile": [1.0, 1.0, 1.0, 1.0, 0.0],  "description": "ESC Clamping Voltage"},
    "esc_leakage":         {"noise_ratio": 0.10,  "unit": "mA",    "phase_profile": [1.0, 1.0, 1.0, 1.0, 0.5],  "description": "ESC Leakage Current"},
    "he_pressure":         {"noise_ratio": 0.020, "unit": "Torr",  "phase_profile": [0.0, 1.0, 1.0, 1.0, 0.0],  "description": "He Backside Pressure"},
    "he_flow":             {"noise_ratio": 0.067, "unit": "sccm",  "phase_profile": [0.0, 1.0, 1.0, 1.0, 0.0],  "description": "He Backside Flow"},

    # --- 펌프 (2개) ---
    "turbo_rpm":           {"noise_ratio": 0.0003,"unit": "RPM",   "phase_profile": [1.0, 1.0, 1.0, 1.0, 1.0],  "description": "Turbo Pump RPM"},
    "turbo_bearing_current":{"noise_ratio": 0.025,"unit": "A",     "phase_profile": [1.0, 1.0, 1.0, 1.0, 1.0],  "description": "Turbo Pump Bearing Current"},

    # --- 공정 결과 (3개) ---
    "particle_count":      {"noise_ratio": 0.75,  "unit": "ea",    "phase_profile": [1.0, 1.0, 1.0, 1.0, 1.0],  "description": "Particle Count"},
    "etch_rate":           {"noise_ratio": 0.014, "unit": "Å/min", "phase_profile": [0.0, 0.5, 1.0, 0.3, 0.0],  "description": "Etch Rate"},
    "edge_uniformity":     {"noise_ratio": 0.15,  "unit": "%",     "phase_profile": [0.0, 0.0, 1.0, 1.0, 0.0],  "description": "Edge Etch Uniformity"},
}

PROCESS_SENSOR_NAMES = list(SENSOR_DEFS.keys())  # 30개


# =============================================================================
# OES 파장 정의 (20개) — 레시피 불문 동일 파장, baseline 강도만 레시피별 상이
# =============================================================================
OES_DEFS = {
    "F_703.7":   {"wavelength": 703.7, "noise_ratio": 0.025, "species": "F",   "role": "식각율 지표"},
    "F_685.6":   {"wavelength": 685.6, "noise_ratio": 0.036, "species": "F",   "role": "식각율 보조"},
    "CF2_251.9": {"wavelength": 251.9, "noise_ratio": 0.050, "species": "CF2", "role": "플라즈마 화학"},
    "CF_202.4":  {"wavelength": 202.4, "noise_ratio": 0.080, "species": "CF",  "role": "해리도 지표"},
    "CO_483.5":  {"wavelength": 483.5, "noise_ratio": 0.044, "species": "CO",  "role": "산화막 종점"},
    "CO_519.8":  {"wavelength": 519.8, "noise_ratio": 0.067, "species": "CO",  "role": "산화막 보조"},
    "SiF_440.0": {"wavelength": 440.0, "noise_ratio": 0.057, "species": "SiF", "role": "Si 식각 생성물"},
    "O_777.2":   {"wavelength": 777.2, "noise_ratio": 0.050, "species": "O",   "role": "산화/리크 지표"},
    "O_844.6":   {"wavelength": 844.6, "noise_ratio": 0.075, "species": "O",   "role": "산화 보조"},
    "Ar_750.4":  {"wavelength": 750.4, "noise_ratio": 0.029, "species": "Ar",  "role": "정규화 기준"},
    "Ar_811.5":  {"wavelength": 811.5, "noise_ratio": 0.040, "species": "Ar",  "role": "정규화 보조"},
    "N2_337.1":  {"wavelength": 337.1, "noise_ratio": 0.25,  "species": "N2",  "role": "에어 리크 감지"},
    "OH_308.9":  {"wavelength": 308.9, "noise_ratio": 0.25,  "species": "OH",  "role": "수분 감지"},
    "H_656.3":   {"wavelength": 656.3, "noise_ratio": 0.20,  "species": "H",   "role": "수분/수소 지표"},
    "Al_394.4":  {"wavelength": 394.4, "noise_ratio": 0.30,  "species": "Al",  "role": "금속 오염 감지"},
    "Si_288.2":  {"wavelength": 288.2, "noise_ratio": 0.10,  "species": "Si",  "role": "Ring/전극 마모 감지"},
    "C2_516.5":  {"wavelength": 516.5, "noise_ratio": 0.10,  "species": "C2",  "role": "폴리머 지표"},
    "CN_388.3":  {"wavelength": 388.3, "noise_ratio": 0.167, "species": "CN",  "role": "PR 잔여물"},
    "He_667.8":  {"wavelength": 667.8, "noise_ratio": 0.167, "species": "He",  "role": "He 리크 감지"},
    "Cu_324.7":  {"wavelength": 324.7, "noise_ratio": 0.30,  "species": "Cu",  "role": "Cu 오염 감지"},
}

OES_NAMES = list(OES_DEFS.keys())  # 20개
ALL_SENSOR_NAMES = PROCESS_SENSOR_NAMES + OES_NAMES  # 50개


# =============================================================================
# 레시피 정의 (4개)
# =============================================================================
# 각 레시피: 모든 센서의 baseline 값 + OES baseline 강도
# 사용하지 않는 가스는 baseline=0 (MFC가 있지만 흐르지 않음)

RECIPES = {
    "oxide_etch": {
        "description": "SiO2 Oxide Etch (CF4/CHF3 기반)",
        "process_time_s": 60,
        "baselines": {
            # RF
            "source_rf_fwd": 800.0, "source_rf_ref": 5.0,
            "bias_rf_fwd": 300.0, "bias_rf_ref": 3.0,
            "dc_bias": -200.0,
            "match_pos_c1": 50.0, "match_pos_c2": 45.0,
            # Gas
            "cf4_flow": 80.0, "chf3_flow": 40.0, "o2_flow": 15.0, "ar_flow": 200.0,
            "cl2_flow": 0.0, "bcl3_flow": 0.0, "c4f8_flow": 0.0, "n2_flow": 0.0,
            # Pressure
            "chamber_pressure": 15.0, "base_pressure": 0.3,
            "foreline_pressure": 0.3, "throttle_valve": 55.0,
            # Temp
            "chuck_temp_center": 40.0, "chuck_temp_edge": 42.0,
            # ESC
            "esc_voltage": 1500.0, "esc_leakage": 0.5,
            "he_pressure": 10.0, "he_flow": 1.5,
            # Pump
            "turbo_rpm": 36000.0, "turbo_bearing_current": 0.4,
            # Result
            "particle_count": 2.0, "etch_rate": 3500.0, "edge_uniformity": 2.0,
        },
        "oes_baselines": {
            "F_703.7": 0.80, "F_685.6": 0.55, "CF2_251.9": 0.60, "CF_202.4": 0.25,
            "CO_483.5": 0.45, "CO_519.8": 0.30, "SiF_440.0": 0.35,
            "O_777.2": 0.40, "O_844.6": 0.20,
            "Ar_750.4": 0.70, "Ar_811.5": 0.50,
            "N2_337.1": 0.02, "OH_308.9": 0.02, "H_656.3": 0.05,
            "Al_394.4": 0.01, "Si_288.2": 0.15, "C2_516.5": 0.10,
            "CN_388.3": 0.03, "He_667.8": 0.03, "Cu_324.7": 0.01,
        },
    },

    "poly_si_etch": {
        "description": "Poly-Si Etch (Cl2/O2 기반)",
        "process_time_s": 60,
        "baselines": {
            # RF: 소스 낮고, 바이어스도 낮음 (선택비 확보)
            "source_rf_fwd": 600.0, "source_rf_ref": 4.0,
            "bias_rf_fwd": 150.0, "bias_rf_ref": 2.0,
            "dc_bias": -120.0,
            "match_pos_c1": 48.0, "match_pos_c2": 52.0,
            # Gas: Cl2 기반, fluorocarbon 없음
            "cf4_flow": 0.0, "chf3_flow": 0.0, "o2_flow": 5.0, "ar_flow": 100.0,
            "cl2_flow": 150.0, "bcl3_flow": 0.0, "c4f8_flow": 0.0, "n2_flow": 0.0,
            # Pressure: 더 낮음
            "chamber_pressure": 10.0, "base_pressure": 0.3,
            "foreline_pressure": 0.25, "throttle_valve": 50.0,
            # Temp
            "chuck_temp_center": 50.0, "chuck_temp_edge": 52.0,
            # ESC
            "esc_voltage": 1200.0, "esc_leakage": 0.5,
            "he_pressure": 8.0, "he_flow": 1.2,
            # Pump
            "turbo_rpm": 36000.0, "turbo_bearing_current": 0.4,
            # Result: Si etch는 rate가 높음
            "particle_count": 1.5, "etch_rate": 5000.0, "edge_uniformity": 1.8,
        },
        "oes_baselines": {
            # F 계열 거의 없음, Cl 기반이므로 다른 패턴
            "F_703.7": 0.03, "F_685.6": 0.02, "CF2_251.9": 0.02, "CF_202.4": 0.01,
            "CO_483.5": 0.05, "CO_519.8": 0.03, "SiF_440.0": 0.05,
            "O_777.2": 0.15, "O_844.6": 0.08,
            "Ar_750.4": 0.55, "Ar_811.5": 0.40,
            "N2_337.1": 0.02, "OH_308.9": 0.02, "H_656.3": 0.03,
            "Al_394.4": 0.01, "Si_288.2": 0.45,  # Si 식각이므로 Si 피크 강함
            "C2_516.5": 0.03, "CN_388.3": 0.02, "He_667.8": 0.03, "Cu_324.7": 0.01,
        },
    },

    "metal_etch": {
        "description": "Metal (Al) Etch (BCl3/Cl2 기반)",
        "process_time_s": 60,
        "baselines": {
            # RF
            "source_rf_fwd": 500.0, "source_rf_ref": 4.0,
            "bias_rf_fwd": 200.0, "bias_rf_ref": 2.5,
            "dc_bias": -150.0,
            "match_pos_c1": 42.0, "match_pos_c2": 48.0,
            # Gas: BCl3 + Cl2 + N2
            "cf4_flow": 0.0, "chf3_flow": 0.0, "o2_flow": 0.0, "ar_flow": 50.0,
            "cl2_flow": 100.0, "bcl3_flow": 80.0, "c4f8_flow": 0.0, "n2_flow": 30.0,
            # Pressure: 낮음
            "chamber_pressure": 8.0, "base_pressure": 0.25,
            "foreline_pressure": 0.2, "throttle_valve": 45.0,
            # Temp: 높음
            "chuck_temp_center": 60.0, "chuck_temp_edge": 63.0,
            # ESC
            "esc_voltage": 1800.0, "esc_leakage": 0.4,
            "he_pressure": 12.0, "he_flow": 1.0,
            # Pump
            "turbo_rpm": 36000.0, "turbo_bearing_current": 0.4,
            # Result: Al etch는 rate 높음
            "particle_count": 2.5, "etch_rate": 6000.0, "edge_uniformity": 2.5,
        },
        "oes_baselines": {
            # F 없음, Cl 기반 + Al 피크 있음 (공정 중 Al 식각 생성물)
            "F_703.7": 0.02, "F_685.6": 0.01, "CF2_251.9": 0.01, "CF_202.4": 0.01,
            "CO_483.5": 0.03, "CO_519.8": 0.02, "SiF_440.0": 0.02,
            "O_777.2": 0.08, "O_844.6": 0.04,
            "Ar_750.4": 0.40, "Ar_811.5": 0.30,
            "N2_337.1": 0.25, "OH_308.9": 0.02, "H_656.3": 0.03,  # N2 사용하므로 높음
            "Al_394.4": 0.50,  # Al 식각 생성물 → 정상적으로 높음
            "Si_288.2": 0.10, "C2_516.5": 0.02,
            "CN_388.3": 0.02, "He_667.8": 0.03, "Cu_324.7": 0.01,
        },
    },

    "harc_etch": {
        "description": "High Aspect Ratio Contact Etch (C4F8/O2 기반)",
        "process_time_s": 60,
        "baselines": {
            # RF: 매우 높은 파워 (깊은 식각)
            "source_rf_fwd": 1500.0, "source_rf_ref": 8.0,
            "bias_rf_fwd": 600.0, "bias_rf_ref": 5.0,
            "dc_bias": -400.0,
            "match_pos_c1": 55.0, "match_pos_c2": 40.0,
            # Gas: C4F8 기반 + 높은 Ar
            "cf4_flow": 0.0, "chf3_flow": 0.0, "o2_flow": 20.0, "ar_flow": 300.0,
            "cl2_flow": 0.0, "bcl3_flow": 0.0, "c4f8_flow": 20.0, "n2_flow": 0.0,
            # Pressure: 높음
            "chamber_pressure": 25.0, "base_pressure": 0.3,
            "foreline_pressure": 0.35, "throttle_valve": 65.0,
            # Temp: 낮은 편 (polymer 보호)
            "chuck_temp_center": 30.0, "chuck_temp_edge": 32.0,
            # ESC: 높은 클램핑 (높은 bias power)
            "esc_voltage": 2000.0, "esc_leakage": 0.6,
            "he_pressure": 15.0, "he_flow": 1.8,
            # Pump
            "turbo_rpm": 36000.0, "turbo_bearing_current": 0.4,
            # Result: HARC는 rate 낮지만 깊은 식각
            "particle_count": 3.0, "etch_rate": 2000.0, "edge_uniformity": 3.0,
        },
        "oes_baselines": {
            # C4F8 기반 → CF2, C2 매우 높음, F도 높음
            "F_703.7": 0.65, "F_685.6": 0.45, "CF2_251.9": 0.85, "CF_202.4": 0.35,
            "CO_483.5": 0.55, "CO_519.8": 0.38, "SiF_440.0": 0.25,
            "O_777.2": 0.30, "O_844.6": 0.15,
            "Ar_750.4": 0.85, "Ar_811.5": 0.60,  # 높은 Ar flow
            "N2_337.1": 0.02, "OH_308.9": 0.02, "H_656.3": 0.04,
            "Al_394.4": 0.01, "Si_288.2": 0.10,
            "C2_516.5": 0.25,  # 높은 C4F8 → polymer 많음
            "CN_388.3": 0.03, "He_667.8": 0.03, "Cu_324.7": 0.01,
        },
    },
}

RECIPE_NAMES = list(RECIPES.keys())


# =============================================================================
# 고장 파라미터 (6개) — 레시피 불문, 물리 현상은 동일
# =============================================================================
# affected_sensors: {센서명: (drift_direction, max_drift_ratio)}
#   max_drift_ratio: 수명 끝에서 baseline 대비 최대 변화 비율

FAULT_PARAMS = {
    # =========================================================================
    # 조정 원칙:
    #   - w2w variation 0.5% → 2.5% (5배 증가)
    #   - drift ratio 대폭 감소 → 목표 SNR 2~5 (primary), 0.5~2 (secondary)
    #   - 단일 센서로 if문 분류 불가, 다변수 조합이 필요한 수준
    #   - Focus Ring / 전극: dc_bias 같은 방향(둘 다 면적 변화) → 패턴 차이로만 구분
    # =========================================================================

    "focus_ring_wear": {
        "description": "Focus Ring 마모 (Quartz/Si)",
        "lifetime_wafers": 8000,
        "affected_sensors": {
            # DC Bias: 에지 전극 면적 변화 → 더 음(-)으로 drift
            "dc_bias":           (-1, 0.015),
            # Edge uniformity 악화 (핵심 차별화: Ring만 에지 영향)
            "edge_uniformity":   (+1, 0.30),
            # Match position 미세 drift
            "match_pos_c1":      (+1, 0.012),
            # 식각율: 에지 영향 → 평균 미세 감소
            "etch_rate":         (-1, 0.008),
            # OES: Ring 재질(Si/Quartz) 스퍼터링 → Si 피크 증가
            "Si_288.2":          (+1, 0.15),
            # F 피크 미세 변화
            "F_703.7":           (-1, 0.010),
        },
        "noise_multiplier": 1.15,
    },

    "electrode_wear": {
        "description": "상부 전극 (Showerhead) 마모",
        "lifetime_wafers": 15000,
        "affected_sensors": {
            # DC Bias: Ring과 같은 방향(-) drift (둘 다 전극 면적 변화)
            # 차이점: 전극은 전체적, Ring은 에지 연동
            "dc_bias":           (-1, 0.012),
            # Throttle valve: 샤워헤드 구멍 확대 → 밸브 닫힘
            "throttle_valve":    (-1, 0.020),
            # 식각율: 전체적으로 서서히 증가 (Ring과 반대 방향!)
            "etch_rate":         (+1, 0.010),
            # Match position drift
            "match_pos_c1":      (-1, 0.010),
            "match_pos_c2":      (+1, 0.010),
            # RF reflected 미세 증가
            "source_rf_ref":     (+1, 0.08),
            # OES: 전극 재질 스퍼터링 (Ring보다 약함)
            "Si_288.2":          (+1, 0.08),
            # Particle: 말기에만 증가
            "particle_count":    (+1, 0.35),
        },
        "noise_multiplier": 1.12,
    },

    "polymer_contamination": {
        "description": "챔버 내벽 폴리머(C-F) 축적",
        "lifetime_wafers": 3000,
        "affected_sensors": {
            # DC Bias 미세 변화
            "dc_bias":           (+1, 0.008),
            # 식각율 서서히 저하
            "etch_rate":         (-1, 0.012),
            # OES: CF2, C2 baseline 증가 (핵심 시그니처)
            "CF2_251.9":         (+1, 0.08),
            "C2_516.5":          (+1, 0.18),
            # CO 감소
            "CO_483.5":          (-1, 0.04),
            # Particle 미세 증가
            "particle_count":    (+1, 0.25),
            # 압력 미세 불안정
            "chamber_pressure":  (+1, 0.006),
        },
        "noise_multiplier": 1.18,
    },

    "metal_contamination": {
        "description": "챔버 금속 오염 (Al/Cu 스퍼터링)",
        "lifetime_wafers": 500,
        "affected_sensors": {
            # OES: 금속 피크 출현 (여전히 가장 강한 시그니처지만 적당히)
            "Al_394.4":          (+1, 0.60),
            "Cu_324.7":          (+1, 0.45),
            # Particle 증가
            "particle_count":    (+1, 0.50),
            # DC Bias 미세 변화
            "dc_bias":           (-1, 0.006),
            # 식각율 변화
            "etch_rate":         (-1, 0.008),
        },
        "noise_multiplier": 1.25,
    },

    "esc_degradation": {
        "description": "ESC 정전척 열화",
        "lifetime_wafers": 30000,
        "affected_sensors": {
            # He flow 증가 (핵심)
            "he_flow":           (+1, 0.20),
            # ESC leakage 증가 (핵심)
            "esc_leakage":       (+1, 0.30),
            # He pressure 유지 어려움
            "he_pressure":       (-1, 0.015),
            # 온도 불균일
            "chuck_temp_edge":   (+1, 0.010),
            "chuck_temp_center": (-1, 0.006),
            # Edge uniformity 미세 악화
            "edge_uniformity":   (+1, 0.15),
        },
        "noise_multiplier": 1.15,
    },

    "pump_degradation": {
        "description": "터보펌프 베어링 마모",
        "lifetime_wafers": 50000,
        "affected_sensors": {
            # Bearing current 증가 (핵심)
            "turbo_bearing_current": (+1, 0.12),
            # RPM 미세 불안정
            "turbo_rpm":         (-1, 0.004),
            # Foreline pressure 상승
            "foreline_pressure": (+1, 0.15),
            # Base pressure 상승
            "base_pressure":     (+1, 0.12),
            # Throttle valve 미세 변화
            "throttle_valve":    (+1, 0.008),
        },
        "noise_multiplier": 1.20,
    },
}


# =============================================================================
# 데이터 생성 설정
# =============================================================================
GENERATION_CONFIG = {
    # 레시피별 웨이퍼 수
    "per_recipe": {
        "normal_wafers": 50_000,
        "fault_wafers": {
            "focus_ring_wear":       10_000,
            "electrode_wear":        10_000,
            "polymer_contamination":  8_000,
            "metal_contamination":    5_000,
            "esc_degradation":        8_000,
            "pump_degradation":       8_000,
        },
        "compound_wafers": 3_000,
    },
    # 4 recipes × (50K + 49K + 3K) = 4 × 102K = 408K total
    "wafer_to_wafer_variation": 0.025,  # 2.5% — 현실적 수준
    "train_seed": 42,
    "test_seed": 99,  # 테스트셋은 다른 시드
}
