"""
EtchMind Demo — 식각장비 자기진단 시스템
장비 모형 + 내장 스크린 UI

streamlit run demo.py
"""

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

import streamlit.components.v1 as components

from simulator.etch_simulator import EtchSimulator
from simulator.config import (
    RECIPES, PROCESS_SENSOR_NAMES, OES_NAMES, OES_DEFS, ALL_SENSOR_NAMES,
)
from model.preprocess import extract_features
from model.etchmind import EtchMindSeq
from model.dataset import CLASS_NAMES, N_CLASSES

# =========================================================================
# 상수
# =========================================================================
WINDOW_SIZE = 32
KEY_SENSORS = ["source_rf_fwd", "dc_bias", "chamber_pressure",
               "etch_rate", "he_flow", "edge_uniformity"]
KEY_LABELS = ["RF Power (W)", "DC Bias (V)", "Pressure (mTorr)",
              "Etch Rate (Å/min)", "He Flow (sccm)", "Uniformity (%)"]

FAULT_INFO = {
    "normal":                {"name": "NORMAL",          "color": "#4CAF50", "bg": "#E8F5E9", "action": "—"},
    "focus_ring_wear":       {"name": "Focus Ring Wear", "color": "#1976D2", "bg": "#E3F2FD", "action": "Focus Ring 교체 예약"},
    "electrode_wear":        {"name": "Electrode Wear",  "color": "#00897B", "bg": "#E0F2F1", "action": "Electrode 점검 예약"},
    "polymer_contamination": {"name": "Polymer Buildup", "color": "#F57C00", "bg": "#FFF3E0", "action": "챔버 Wet Clean (PM) 예약"},
    "metal_contamination":   {"name": "Metal Contam.",   "color": "#D32F2F", "bg": "#FFEBEE", "action": "즉시 정지 — 챔버 점검"},
    "esc_degradation":       {"name": "ESC Degradation", "color": "#7B1FA2", "bg": "#F3E5F5", "action": "ESC 교체 계획 수립"},
    "pump_degradation":      {"name": "Pump Degradation","color": "#5D4037", "bg": "#EFEBE9", "action": "터보펌프 정비 예약"},
}

SCENARIOS = [
    ("정상 운전",           None),
    ("Focus Ring 마모",    "focus_ring_wear"),
    ("전극 마모",          "electrode_wear"),
    ("폴리머 오염",        "polymer_contamination"),
    ("금속 오염",          "metal_contamination"),
    ("ESC 열화",           "esc_degradation"),
    ("펌프 열화",          "pump_degradation"),
]

# =========================================================================
# 스타일
# =========================================================================
def inject_css():
    st.markdown("""
    <style>
    /* 다크모드 환경에서도 라이트 테마 강제 — 글씨 안 보이는 문제 방지 */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #F5F7FA !important;
        color: #263238 !important;
    }
    .stApp *, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
    label, .stSelectbox label, .stNumberInput label, .stTextInput label,
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    [data-baseweb="select"] *, [data-baseweb="input"] * {
        color: #263238 !important;
    }
    /* 입력 위젯 배경 */
    [data-baseweb="select"] > div, [data-baseweb="input"] > div,
    .stNumberInput input, .stTextInput input, .stSelectbox div[role="combobox"] {
        background-color: #FFFFFF !important;
        color: #263238 !important;
    }
    header[data-testid="stHeader"] { display: none; }

    .equip-body {
        background: linear-gradient(180deg, #E8ECF0 0%, #D5DAE0 100%);
        border: 2px solid #B0B8C4;
        border-radius: 8px;
        box-shadow: 2px 4px 12px rgba(0,0,0,0.15);
        position: relative;
    }
    .equip-screen {
        background: #FFFFFF;
        border: 2px solid #90A4AE;
        border-radius: 6px;
        padding: 12px;
        box-shadow: inset 0 1px 4px rgba(0,0,0,0.1);
    }
    .screen-header {
        background: #37474F;
        color: white;
        padding: 6px 14px;
        border-radius: 4px 4px 0 0;
        font-family: 'Segoe UI', sans-serif;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .gauge-card {
        background: white;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .gauge-label { color: #78909C; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .gauge-value { color: #263238; font-size: 22px; font-weight: 700; font-family: 'Consolas', monospace; }
    .gauge-unit { color: #90A4AE; font-size: 12px; }
    .gauge-delta { font-size: 11px; font-family: 'Consolas', monospace; }

    .result-card {
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .result-title { font-size: 11px; color: #78909C; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .result-status { font-size: 20px; font-weight: 700; margin: 6px 0; }
    .result-detail { font-size: 12px; color: #607D8B; line-height: 1.8; }

    .summary-box {
        background: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .summary-label { color: #90A4AE; font-size: 12px; font-weight: 600; text-transform: uppercase; }
    .summary-value { font-size: 28px; font-weight: 700; font-family: 'Consolas', monospace; margin: 4px 0; }
    </style>
    """, unsafe_allow_html=True)


# =========================================================================
# 장비 SVG
# =========================================================================
def equipment_html(state: str = "off", severity: float = 0.0, fault_type: str | None = None, wafer_num: int = 0):
    """식각장비 모형 — HTML/CSS 기반, 애니메이션 포함"""

    def part_color(part_name):
        if not fault_type or severity < 0.05:
            return "#90A4AE"
        part_map = {"focus_ring_wear": "ring", "electrode_wear": "electrode",
                    "polymer_contamination": "wall", "metal_contamination": "wall",
                    "esc_degradation": "esc", "pump_degradation": "pump"}
        if part_map.get(fault_type) == part_name:
            if severity < 0.3: return "#FFA726"
            if severity < 0.6: return "#FF7043"
            return "#EF5350"
        return "#90A4AE"

    is_running = state != "off"
    led = "#4CAF50" if state == "running" else ("#F44336" if state == "critical" else ("#FFC107" if state == "warning" else "#9E9E9E"))
    plasma_bg = "linear-gradient(180deg, #64B5F6 0%, #42A5F5 30%, #1E88E5 70%, #1565C0 100%)" if is_running else "#ECEFF1"
    plasma_anim = "plasma-glow 1.5s ease-in-out infinite" if is_running else "none"
    gas_anim = "gas-flow 1s linear infinite" if is_running else "none"

    return f"""
    <style>
    @keyframes plasma-glow {{
        0%, 100% {{ opacity: 0.7; filter: brightness(1); }}
        50% {{ opacity: 1; filter: brightness(1.3); }}
    }}
    @keyframes gas-flow {{
        0% {{ background-position: 0 0; }}
        100% {{ background-position: 0 20px; }}
    }}
    @keyframes led-pulse {{
        0%, 100% {{ box-shadow: 0 0 4px {led}; }}
        50% {{ box-shadow: 0 0 12px {led}, 0 0 20px {led}40; }}
    }}
    @keyframes particle-move {{
        0% {{ transform: translateY(0) translateX(0); opacity: 0.8; }}
        50% {{ transform: translateY(-15px) translateX(5px); opacity: 0.4; }}
        100% {{ transform: translateY(-30px) translateX(-3px); opacity: 0; }}
    }}
    </style>

    <div style="background: linear-gradient(145deg, #CFD8DC, #B0BEC5); border-radius: 10px; padding: 3px;
                box-shadow: 3px 5px 15px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.3);
                max-width: 380px; margin: auto;">

      <!-- 상단 패널 -->
      <div style="background: #78909C; border-radius: 8px 8px 0 0; padding: 8px 16px;
                  display: flex; justify-content: space-between; align-items: center;">
        <div style="display: flex; align-items: center; gap: 10px;">
          <div style="width: 12px; height: 12px; border-radius: 50%; background: {led};
                      animation: led-pulse 1.5s infinite;"></div>
          <span style="color: white; font: bold 12px Consolas;">ICP ETCHER</span>
        </div>
        <span style="color: #CFD8DC; font: 11px Consolas;">W#{wafer_num}</span>
      </div>

      <!-- 챔버 영역 -->
      <div style="background: #ECEFF1; margin: 3px; border-radius: 4px; padding: 8px; position: relative;">

        <!-- 가스 라인 -->
        <div style="display: flex; justify-content: center; margin-bottom: 4px;">
          <div style="width: 40px; height: 16px; border: 2px solid #78909C; border-bottom: none;
                      border-radius: 4px 4px 0 0; position: relative; overflow: hidden;">
            <div style="position: absolute; inset: 0;
                        background: repeating-linear-gradient(180deg, transparent, transparent 4px, {'#42A5F5' if is_running else '#CFD8DC'} 4px, {'#42A5F5' if is_running else '#CFD8DC'} 8px);
                        animation: {gas_anim};"></div>
          </div>
        </div>
        <div style="text-align:center; font: 9px Consolas; color: #90A4AE; margin-bottom: 4px;">
          {'GAS FLOW ▼▼▼' if is_running else 'GAS LINE'}</div>

        <!-- 전극 -->
        <div style="background: {part_color('electrode')}; margin: 0 20px; padding: 6px;
                    border-radius: 4px; text-align: center; position: relative;">
          <span style="color: white; font: bold 11px Consolas;">ELECTRODE (Showerhead)</span>
          {'<div style="position:absolute; bottom:-2px; left:20%; right:20%; height:2px; background:repeating-linear-gradient(90deg, transparent, transparent 8px, white 8px, white 10px);"></div>' if is_running else ''}
        </div>

        <!-- 플라즈마 -->
        <div style="background: {plasma_bg}; margin: 4px 30px; padding: 20px 10px;
                    border-radius: 4px; text-align: center; position: relative;
                    animation: {plasma_anim}; min-height: 60px;">
          {'<span style="color: white; font: bold 13px Consolas; text-shadow: 0 0 10px white;">⚡ PLASMA ⚡</span>' if is_running else '<span style="color: #90A4AE; font: 12px Consolas;">CHAMBER</span>'}
          {f'<div style="position:absolute; top:10px; left:15px; width:4px; height:4px; background:white; border-radius:50%; animation: particle-move 0.8s infinite;"></div><div style="position:absolute; top:15px; right:20px; width:3px; height:3px; background:#E3F2FD; border-radius:50%; animation: particle-move 1.1s infinite 0.3s;"></div><div style="position:absolute; top:8px; left:45%; width:3px; height:3px; background:white; border-radius:50%; animation: particle-move 0.9s infinite 0.5s;"></div>' if is_running else ''}
        </div>

        <!-- Focus Ring + Wafer + ESC -->
        <div style="display: flex; margin: 4px 10px; gap: 3px; align-items: stretch;">
          <div style="background: {part_color('ring')}; width: 30px; border-radius: 3px;
                      display: flex; align-items: center; justify-content: center;">
            <span style="color: white; font: bold 7px Consolas; writing-mode: vertical-lr;">F.RING</span>
          </div>
          <div style="flex: 1;">
            <!-- Wafer -->
            <div style="background: linear-gradient(90deg, #B0BEC5, #CFD8DC, #B0BEC5);
                        border: 1px solid #90A4AE; border-radius: 3px; padding: 4px;
                        text-align: center; margin-bottom: 3px;">
              <span style="font: 10px Consolas; color: #546E7A;">◉ WAFER ◉</span>
            </div>
            <!-- ESC -->
            <div style="background: {part_color('esc')}; border-radius: 3px; padding: 5px;
                        text-align: center;">
              <span style="color: white; font: bold 10px Consolas;">ESC CHUCK</span>
              <span style="color: rgba(255,255,255,0.7); font: 8px Consolas; display: block;">He Backside Cooling</span>
            </div>
          </div>
          <div style="background: {part_color('ring')}; width: 30px; border-radius: 3px;
                      display: flex; align-items: center; justify-content: center;">
            <span style="color: white; font: bold 7px Consolas; writing-mode: vertical-lr;">F.RING</span>
          </div>
        </div>

        <!-- 챔버 벽 표시 -->
        <div style="position: absolute; top: 60px; left: 8px; bottom: 8px; width: 4px;
                    background: {part_color('wall')}; border-radius: 2px;"></div>
        <div style="position: absolute; top: 60px; right: 8px; bottom: 8px; width: 4px;
                    background: {part_color('wall')}; border-radius: 2px;"></div>
      </div>

      <!-- 배기 + 펌프 -->
      <div style="display: flex; justify-content: center; padding: 4px 0;">
        <div style="width: 2px; height: 15px; background: #78909C;"></div>
      </div>
      <div style="display: flex; justify-content: center; padding-bottom: 8px;">
        <div style="background: {part_color('pump')}; padding: 6px 20px; border-radius: 5px;
                    text-align: center;">
          <span style="color: white; font: bold 10px Consolas;">TURBO PUMP</span>
          {'<span style="color: rgba(255,255,255,0.7); font: 9px Consolas; display:block;">RPM ▶▶▶</span>' if is_running else ''}
        </div>
      </div>

      <!-- 하단 다리 -->
      <div style="display: flex; justify-content: space-between; padding: 0 30px 5px;">
        <div style="width: 20px; height: 8px; background: #607D8B; border-radius: 0 0 3px 3px;"></div>
        <div style="width: 20px; height: 8px; background: #607D8B; border-radius: 0 0 3px 3px;"></div>
        <div style="width: 20px; height: 8px; background: #607D8B; border-radius: 0 0 3px 3px;"></div>
        <div style="width: 20px; height: 8px; background: #607D8B; border-radius: 0 0 3px 3px;"></div>
      </div>
    </div>
    """


# =========================================================================
# 모델
# =========================================================================
@st.cache_resource
def load_model():
    ckpt = torch.load("./checkpoints/best_sequence.pt", map_location="cpu", weights_only=False)
    model = EtchMindSeq(n_features=ckpt["config"]["n_features"], n_classes=ckpt["config"]["n_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["stats"]


def rule_predict(features):
    means = features[:len(ALL_SENSOR_NAMES)]
    s = {n: i for i, n in enumerate(ALL_SENSOR_NAMES)}
    for sensor, thr, fault in [
        ("Al_394.4", 0.05, "metal_contamination"), ("Cu_324.7", 0.04, "metal_contamination"),
        ("esc_leakage", 2.0, "esc_degradation"), ("he_flow", 3.0, "esc_degradation"),
        ("turbo_bearing_current", 0.6, "pump_degradation"),
        ("C2_516.5", 0.2, "polymer_contamination"),
        ("edge_uniformity", 3.5, "focus_ring_wear"),
        ("particle_count", 5.0, "electrode_wear"),
    ]:
        if sensor in s and abs(means[s[sensor]]) > thr:
            return fault
    return "normal"


# =========================================================================
# 차트 (밝은 테마)
# =========================================================================
LIGHT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
    font=dict(family="Segoe UI", color="#455A64", size=11),
    margin=dict(t=30, b=25, l=50, r=15),
    xaxis=dict(gridcolor="#ECEFF1", zerolinecolor="#CFD8DC"),
    yaxis=dict(gridcolor="#ECEFF1", zerolinecolor="#CFD8DC"),
)

def make_sensor_chart(hist, fault_start):
    fig = make_subplots(rows=3, cols=2, subplot_titles=KEY_LABELS,
                        vertical_spacing=0.10, horizontal_spacing=0.08)
    for i, s in enumerate(KEY_SENSORS):
        r, c = i // 2 + 1, i % 2 + 1
        fig.add_trace(go.Scattergl(y=hist[s], mode="lines",
                      line=dict(color="#1976D2", width=1.5), showlegend=False), row=r, col=c)
        if fault_start < len(hist[s]):
            fig.add_vline(x=fault_start, line_dash="dash", line_color="#F44336", opacity=0.5, row=r, col=c)
    fig.update_layout(height=400, **LIGHT_LAYOUT)
    fig.update_annotations(font=dict(color="#607D8B", size=11))
    return fig

def make_oes_chart(vals):
    species = [OES_DEFS[n]["species"] for n in OES_NAMES]
    wl = [OES_DEFS[n]["wavelength"] for n in OES_NAMES]
    fig = go.Figure(go.Bar(
        x=[f"{s} {w:.0f}" for s, w in zip(species, wl)], y=vals,
        marker_color="#42A5F5",
    ))
    fig.update_layout(height=250, title="OES Spectrum", **LIGHT_LAYOUT)
    return fig

def make_severity_chart(dl_sev, actual_sev):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=dl_sev, mode="lines", name="EtchMind 예측",
                             line=dict(color="#1976D2", width=2.5)))
    if actual_sev:
        fig.add_trace(go.Scatter(y=actual_sev, mode="lines", name="실제 Severity",
                                 line=dict(color="#F44336", width=2, dash="dash")))
    fig.update_layout(height=220, title="Severity Tracking", yaxis_range=[-0.05, 1.1],
                      legend=dict(x=0.02, y=0.98), **LIGHT_LAYOUT)
    return fig


# =========================================================================
# 메인
# =========================================================================
def main():
    st.set_page_config(page_title="EtchMind Demo", layout="wide", initial_sidebar_state="collapsed")
    inject_css()
    model, stats = load_model()

    # ====== 상단 타이틀 ======
    st.markdown("""
    <div style="background:white; padding:12px 24px; border-radius:10px; margin-bottom:16px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06); display:flex; justify-content:space-between; align-items:center;">
        <div>
            <span style="font-size:22px; font-weight:700; color:#263238;">EtchMind</span>
            <span style="font-size:14px; color:#90A4AE; margin-left:10px;">Semiconductor Etch Equipment Self-Diagnosis System</span>
        </div>
        <div style="font-size:12px; color:#90A4AE;">
            Model: CNN+Transformer (449K params) &nbsp;│&nbsp; Accuracy: 96.1%
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ====== 장비 + 스크린 + 차트 레이아웃 ======
    equip_col, chart_col = st.columns([2, 3])

    with equip_col:
        # 장비 모형 (iframe으로 렌더링)
        svg_ph = st.empty()
        with svg_ph:
            components.html(equipment_html("off"), height=480, scrolling=False)

        # 장비 내장 스크린 (컨트롤)
        st.markdown('<div class="equip-screen">', unsafe_allow_html=True)
        st.markdown('<div class="screen-header">📟 DIAGNOSTIC SCREEN — CONTROL</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            recipe = st.selectbox("Recipe", list(RECIPES.keys()),
                                  format_func=lambda x: RECIPES[x]["description"])
        with c2:
            sc_idx = st.selectbox("Scenario", range(len(SCENARIOS)),
                                  format_func=lambda i: SCENARIOS[i][0])
        fault_type = SCENARIOS[sc_idx][1]

        c3, c4, c5 = st.columns(3)
        with c3:
            n_wafers = st.number_input("Wafers", 50, 300, 120)
        with c4:
            speed = st.number_input("Speed", 1, 30, 12)
        with c5:
            fault_start = st.number_input("Fault @", 10, n_wafers - 10, n_wafers // 4)

        run = st.button("▶  START SIMULATION", type="primary", use_container_width=True)

        # 진단 결과 표시 영역
        diag_ph = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col:
        sensor_chart_ph = st.empty()
        bottom_row = st.columns(2)
        sev_chart_ph = bottom_row[0].empty()
        oes_chart_ph = bottom_row[1].empty()

    # 게이지 바
    gauge_ph = st.empty()

    # 하단: DL vs Rule 비교
    compare_ph = st.empty()

    if not run:
        # 초기 상태
        with diag_ph.container():
            st.markdown("""
            <div class="result-card" style="background:#E8F5E9; border-color:#4CAF50;">
                <div class="result-title">SYSTEM STATUS</div>
                <div class="result-status" style="color:#4CAF50;">● STANDBY</div>
                <div class="result-detail">시나리오를 선택하고 START를 누르세요.</div>
            </div>
            """, unsafe_allow_html=True)
        return

    # ====== 시뮬레이션 ======
    sim = EtchSimulator(recipe_name=recipe, seed=int(time.time()) % 10000)
    mn, sd = stats["mean"], stats["std"].copy()
    sd[sd < 1e-8] = 1.0

    feat_hist = []
    dl_preds, dl_sevs, actual_sevs, rule_preds = [], [], [], []
    sensor_hist = {s: [] for s in KEY_SENSORS}
    baselines = {s: RECIPES[recipe]["baselines"].get(s, 0) for s in KEY_SENSORS}

    progress = st.progress(0)

    for w in range(n_wafers):
        sev = min((w - fault_start) / max(n_wafers - fault_start, 1) * 1.2, 1.0) if fault_type and w >= fault_start else 0.0

        trace = sim.generate_normal_wafer()
        if fault_type and sev > 0:
            trace = sim.inject_fault(trace, fault_type, sev)

        feat = extract_features(trace)
        feat_hist.append(feat)
        actual_sevs.append(sev)

        for s in KEY_SENSORS:
            sensor_hist[s].append(trace[70:450, ALL_SENSOR_NAMES.index(s)].mean())

        # Rule
        rule_preds.append(rule_predict(feat))

        # DL
        dl_pred, dl_conf, dl_sev = "normal", 1.0, 0.0
        if len(feat_hist) >= WINDOW_SIZE:
            win = np.array(feat_hist[-WINDOW_SIZE:], dtype=np.float32)
            x = torch.tensor((win - mn) / sd).unsqueeze(0)
            with torch.no_grad():
                logits, sv = model(x)
                probs = torch.softmax(logits, -1)
                cls = probs.argmax(1).item()
                dl_pred, dl_conf, dl_sev = CLASS_NAMES[cls], probs[0, cls].item(), sv.item()

        dl_preds.append(dl_pred)
        dl_sevs.append(dl_sev)

        # 상태
        state = "running" if dl_pred == "normal" else ("critical" if dl_sev > 0.6 else "warning")

        # UI 업데이트
        if w % 3 == 0 or w == n_wafers - 1:
            with svg_ph:
                components.html(equipment_html(state, sev, fault_type, w + 1), height=480, scrolling=False)

            # 진단 카드
            info = FAULT_INFO.get(dl_pred, FAULT_INFO["normal"])
            with diag_ph.container():
                if dl_pred == "normal":
                    st.markdown(f"""
                    <div class="result-card" style="background:{info['bg']}; border-color:{info['color']};">
                        <div class="result-title">EtchMind AI</div>
                        <div class="result-status" style="color:{info['color']};">● NORMAL</div>
                        <div class="result-detail">Wafer #{w+1} &nbsp;│&nbsp; Confidence: {dl_conf:.0%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    rul = int((1 - dl_sev) * 5000)
                    bar_w = int(dl_sev * 100)
                    st.markdown(f"""
                    <div class="result-card" style="background:{info['bg']}; border-color:{info['color']};">
                        <div class="result-title">EtchMind AI — FAULT DETECTED</div>
                        <div class="result-status" style="color:{info['color']};">▲ {info['name']}</div>
                        <div class="result-detail">
                            Confidence: <b>{dl_conf:.0%}</b> &nbsp;│&nbsp;
                            Severity: <b>{dl_sev:.0%}</b> &nbsp;│&nbsp;
                            잔여수명: <b>~{rul:,} wafers</b>
                        </div>
                        <div style="background:#E0E0E0; border-radius:4px; height:8px; margin:8px 0; overflow:hidden;">
                            <div style="width:{bar_w}%; height:100%; background:{info['color']}; border-radius:4px; transition:width 0.3s;"></div>
                        </div>
                        <div class="result-detail">권장 조치: <b style="color:{info['color']}">{info['action']}</b></div>
                    </div>
                    """, unsafe_allow_html=True)

            # 센서 차트
            sensor_chart_ph.plotly_chart(make_sensor_chart(sensor_hist, fault_start),
                                         use_container_width=True)

            # Severity + OES
            sev_chart_ph.plotly_chart(make_severity_chart(dl_sevs, actual_sevs),
                                      use_container_width=True)
            oes_chart_ph.plotly_chart(make_oes_chart(trace[300, len(PROCESS_SENSOR_NAMES):]),
                                      use_container_width=True)

            # 게이지
            with gauge_ph.container():
                gcols = st.columns(len(KEY_SENSORS))
                for i, (s, label, unit) in enumerate(zip(KEY_SENSORS, KEY_LABELS,
                    ["W", "V", "mTorr", "Å/min", "sccm", "%"])):
                    val = sensor_hist[s][-1]
                    bl = baselines[s]
                    delta = (val - bl) / abs(bl) * 100 if bl else 0
                    dc = "#4CAF50" if abs(delta) < 3 else ("#FF9800" if abs(delta) < 8 else "#F44336")
                    with gcols[i]:
                        st.markdown(f"""
                        <div class="gauge-card">
                            <div class="gauge-label">{label.split('(')[0].strip()}</div>
                            <div class="gauge-value">{val:.1f}</div>
                            <div class="gauge-unit">{unit}</div>
                            <div class="gauge-delta" style="color:{dc}">{delta:+.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

            # DL vs Rule 비교
            rule_info = FAULT_INFO.get(rule_preds[-1], FAULT_INFO["normal"])
            with compare_ph.container():
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.markdown(f"""
                    <div class="summary-box" style="border-top: 4px solid #1976D2;">
                        <div class="summary-label">EtchMind (Deep Learning)</div>
                        <div class="summary-value" style="color:{info['color']}">{info['name']}</div>
                        <div style="color:#90A4AE; font-size:12px;">Severity: {dl_sev:.0%} │ Conf: {dl_conf:.0%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with cc2:
                    st.markdown(f"""
                    <div class="summary-box" style="border-top: 4px solid #FF9800;">
                        <div class="summary-label">Rule-based (Baseline)</div>
                        <div class="summary-value" style="color:{rule_info['color']}">{rule_info['name']}</div>
                        <div style="color:#90A4AE; font-size:12px;">단일 변수 임계값 기반</div>
                    </div>
                    """, unsafe_allow_html=True)

        progress.progress((w + 1) / n_wafers)
        time.sleep(1.0 / speed)

    # ====== 완료 요약 ======
    progress.empty()
    dl_first = next((i for i, p in enumerate(dl_preds) if p != "normal"), n_wafers)
    rule_first = next((i for i, p in enumerate(rule_preds) if p != "normal"), n_wafers)

    st.divider()
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"""
        <div class="summary-box">
            <div class="summary-label">EtchMind 최초 감지</div>
            <div class="summary-value" style="color:#1976D2;">
                {'Wafer #' + str(dl_first) if dl_first < n_wafers else '—'}</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="summary-box">
            <div class="summary-label">Rule-based 최초 감지</div>
            <div class="summary-value" style="color:#FF9800;">
                {'Wafer #' + str(rule_first) if rule_first < n_wafers else 'Not Detected'}</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        adv = rule_first - dl_first if dl_first < rule_first else 0
        st.markdown(f"""
        <div class="summary-box">
            <div class="summary-label">EtchMind 조기 감지</div>
            <div class="summary-value" style="color:#4CAF50;">
                {str(adv) + ' wafers 빠름' if adv > 0 else ('동일' if fault_type else '정상 운전')}</div>
        </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
