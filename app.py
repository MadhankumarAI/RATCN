"""
RA-TCN Interactive Demo
=======================
Streamlit app showcasing Regime-Adaptive TCN vs Vanilla TCN.
Run:  streamlit run app.py
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dataset import generate_regime_data, TimeSeriesDataset
from models import VanillaTCN, RATCN

# ── Page config ──
st.set_page_config(
    page_title="RA-TCN Lab",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme state ──
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

IS_LIGHT = st.session_state["theme"] == "light"
THEME_CLASS = "light" if IS_LIGHT else "dark"

# ── Custom CSS with dual theme support ──
# Theme variables are injected separately to avoid f-string escaping issues with CSS braces
if IS_LIGHT:
    _theme_vars = """
    :root {
        --bg-primary: #f4f1ec;
        --bg-card: #ffffff;
        --bg-card-hover: #f8f6f3;
        --border: #e0dbd4;
        --text-primary: #1a1a1a;
        --text-secondary: #5c5a56;
        --text-muted: #9e9a93;
        --accent-cyan: #0a8f7f;
        --accent-amber: #c48500;
        --accent-red: #c9304a;
        --accent-blue: #3460cc;
        --glow-cyan: rgba(10, 143, 127, 0.1);
        --glow-red: rgba(201, 48, 74, 0.1);
        --toggle-bg: #f0ede8;
        --toggle-border: #d4d0ca;
        --sidebar-text: #5c5a56;
    }
    """
else:
    _theme_vars = """
    :root {
        --bg-primary: #08090d;
        --bg-card: #0f1118;
        --bg-card-hover: #161825;
        --border: #1e2035;
        --text-primary: #e8e6e3;
        --text-secondary: #6b7094;
        --text-muted: #3d4166;
        --accent-cyan: #00e5c7;
        --accent-amber: #ffb224;
        --accent-red: #ff4d6a;
        --accent-blue: #4d7cff;
        --glow-cyan: rgba(0, 229, 199, 0.15);
        --glow-red: rgba(255, 77, 106, 0.15);
        --toggle-bg: #161825;
        --toggle-border: #2a2d52;
        --sidebar-text: #6b7094;
    }
    """

st.markdown("<style>" + _theme_vars + "</style>", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Instrument+Serif:ital@0;1&family=Sora:wght@300;400;600;700&display=swap');

/* Global overrides */
.stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'Sora', sans-serif !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

section[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown label,
section[data-testid="stSidebar"] label {
    color: var(--sidebar-text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* Theme toggle */
.theme-toggle {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 8px 14px;
    background: var(--toggle-bg);
    border: 1px solid var(--toggle-border);
    border-radius: 100px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-secondary);
    letter-spacing: 0.05em;
    margin-bottom: 1.2rem;
}
.theme-toggle .active {
    color: var(--accent-cyan);
    font-weight: 500;
}

/* Hero title */
.hero-title {
    font-family: 'Instrument Serif', serif !important;
    font-size: 3.2rem;
    font-weight: 400;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0;
    padding-top: 1rem;
}

.hero-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: var(--text-secondary);
    letter-spacing: 0.04em;
    margin-top: 0.5rem;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card:hover {
    background: var(--bg-card-hover);
    border-color: #2a2d52;
}
.metric-card.cyan::before { background: var(--accent-cyan); }
.metric-card.red::before { background: var(--accent-red); }
.metric-card.amber::before { background: var(--accent-amber); }

.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Sora', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.2;
}
.metric-value.cyan { color: var(--accent-cyan); }
.metric-value.red { color: var(--accent-red); }
.metric-value.amber { color: var(--accent-amber); }
.metric-delta {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    margin-top: 0.25rem;
}
.metric-delta.positive { color: var(--accent-cyan); }
.metric-delta.negative { color: var(--accent-red); }

/* Section headers */
.section-header {
    font-family: 'Instrument Serif', serif;
    font-size: 1.6rem;
    color: var(--text-primary);
    margin: 2rem 0 0.5rem 0;
    letter-spacing: -0.01em;
}
.section-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent-cyan);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    display: inline-block;
    padding: 2px 8px;
    border: 1px solid rgba(0,229,199,0.25);
    border-radius: 4px;
    margin-bottom: 0.5rem;
}

/* Architecture diagram */
.arch-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
}
.arch-flow {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.arch-node {
    background: #161825;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.7rem 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-secondary);
    text-align: center;
    min-width: 100px;
}
.arch-node.highlight {
    border-color: var(--accent-cyan);
    color: var(--accent-cyan);
    background: rgba(0,229,199,0.05);
    box-shadow: 0 0 20px rgba(0,229,199,0.08);
}
.arch-node.highlight-amber {
    border-color: var(--accent-amber);
    color: var(--accent-amber);
    background: rgba(255,178,36,0.05);
}
.arch-arrow {
    color: var(--text-muted);
    font-size: 1.2rem;
}

/* Status pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 100px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
}
.status-pill.live {
    background: rgba(0,229,199,0.1);
    color: var(--accent-cyan);
    border: 1px solid rgba(0,229,199,0.2);
}
.status-pill .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent-cyan);
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Training log */
.train-log {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-secondary);
    max-height: 200px;
    overflow-y: auto;
    line-height: 1.8;
}
.train-log .epoch { color: var(--text-muted); }
.train-log .loss { color: var(--accent-amber); }
.train-log .tag { color: var(--accent-cyan); }

/* Hide streamlit branding */
#MainMenu, footer, .stDeployButton { display: none !important; }

/* Plotly chart containers */
.stPlotlyChart {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

/* Slider styling */
.stSlider > div > div { color: var(--text-secondary) !important; }

/* Primary action button */
div.stButton > button {
    background: linear-gradient(135deg, var(--accent-cyan), #00b3a0) !important;
    color: #08090d !important;
    border: none !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 2rem !important;
    border-radius: 8px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.3s ease !important;
}
div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px var(--glow-cyan) !important;
}

/* Theme toggle button — subtle ghost style */
div[data-testid="stSidebar"] button[kind="secondary"],
button[key="theme_toggle_btn"] {
    background: var(--toggle-bg) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--toggle-border) !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 400 !important;
    font-size: 0.75rem !important;
    padding: 0.4rem 1rem !important;
    border-radius: 6px !important;
    letter-spacing: 0.05em !important;
}
div[data-testid="stSidebar"] button[key="theme_toggle_btn"]:hover {
    background: var(--bg-card-hover) !important;
    transform: none !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)


# ── Constants ──
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Cached training functions ──

def train_model(model, loader, epochs, progress_cb=None):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    model.train()
    logs = []
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(loader.dataset)
        logs.append((ep, avg))
        if progress_cb:
            progress_cb(ep, epochs, avg)
    return logs


@torch.no_grad()
def predict_rolling(model, series, lookback, horizon):
    model.eval()
    preds = np.full(len(series), np.nan)
    for i in range(0, len(series) - lookback - horizon + 1, horizon):
        x = torch.tensor(series[i:i+lookback], dtype=torch.float32)
        x = x.unsqueeze(0).unsqueeze(0).to(DEVICE)
        yhat = model(x).cpu().numpy().flatten()
        preds[i+lookback : i+lookback+horizon] = yhat
    return preds


def mse_per_regime(preds, truth, regimes):
    results = {}
    for r in sorted(set(regimes)):
        mask = (regimes == r) & ~np.isnan(preds)
        if mask.sum() > 0:
            results[r] = float(np.mean((preds[mask] - truth[mask]) ** 2))
        else:
            results[r] = float("nan")
    return results


# ── Plotly theme (adapts to light/dark) ──
if IS_LIGHT:
    PLOTLY_LAYOUT = dict(
        paper_bgcolor="#f4f1ec",
        plot_bgcolor="#ffffff",
        font=dict(family="DM Mono, monospace", color="#5c5a56", size=11),
        xaxis=dict(gridcolor="#e8e5e0", zerolinecolor="#e0dbd4"),
        yaxis=dict(gridcolor="#e8e5e0", zerolinecolor="#e0dbd4"),
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e0dbd4",
            borderwidth=1,
            font=dict(size=10, color="#3a3835"),
        ),
    )
else:
    PLOTLY_LAYOUT = dict(
        paper_bgcolor="#08090d",
        plot_bgcolor="#0f1118",
        font=dict(family="DM Mono, monospace", color="#6b7094", size=11),
        xaxis=dict(gridcolor="#1e2035", zerolinecolor="#1e2035"),
        yaxis=dict(gridcolor="#1e2035", zerolinecolor="#1e2035"),
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(
            bgcolor="rgba(15,17,24,0.8)",
            bordercolor="#1e2035",
            borderwidth=1,
            font=dict(size=10),
        ),
    )

# ── Accent colors for Plotly (theme-aware) ──
if IS_LIGHT:
    CLR_CYAN = "#0a8f7f"
    CLR_RED = "#c9304a"
    CLR_AMBER = "#c48500"
    CLR_BLUE = "#3460cc"
    CLR_GROUND = "#3a3835"
    CLR_TITLE = "#1a1a1a"
    CLR_ANN = "#c9304a"
    CLR_REGIME_BG = ["rgba(52,96,204,0.07)", "rgba(201,48,74,0.07)", "rgba(10,143,127,0.07)"]
    CLR_REGIME_BORDER = ["#3460cc", "#c9304a", "#0a8f7f"]
    CLR_FILL_RED = "rgba(201,48,74,0.12)"
    CLR_FILL_CYAN = "rgba(10,143,127,0.12)"
else:
    CLR_CYAN = "#00e5c7"
    CLR_RED = "#ff4d6a"
    CLR_AMBER = "#ffb224"
    CLR_BLUE = "#4d7cff"
    CLR_GROUND = "#e8e6e3"
    CLR_TITLE = "#e8e6e3"
    CLR_ANN = "#ff4d6a"
    CLR_REGIME_BG = ["rgba(77,124,255,0.08)", "rgba(255,77,106,0.08)", "rgba(0,229,199,0.08)"]
    CLR_REGIME_BORDER = ["#4d7cff", "#ff4d6a", "#00e5c7"]
    CLR_FILL_RED = "rgba(255,77,106,0.15)"
    CLR_FILL_CYAN = "rgba(0,229,199,0.15)"


# ═══════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════

with st.sidebar:
    st.markdown(f"""
    <div style="margin-bottom:1.5rem">
        <div style="font-family:'Instrument Serif',serif;font-size:1.4rem;color:var(--text-primary)">
            RA-TCN Lab
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:var(--text-muted);
                    letter-spacing:0.1em;text-transform:uppercase;margin-top:4px">
            experiment configuration
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Theme toggle
    _dark_active = "active" if not IS_LIGHT else ""
    _light_active = "active" if IS_LIGHT else ""
    st.markdown(f"""
    <div class="theme-toggle">
        <span class="{_dark_active}">DARK</span>
        <span style="color:var(--text-muted)">/</span>
        <span class="{_light_active}">LIGHT</span>
    </div>
    """, unsafe_allow_html=True)

    def toggle_theme():
        st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"

    st.button(
        "Switch to Light" if not IS_LIGHT else "Switch to Dark",
        on_click=toggle_theme,
        use_container_width=True,
        key="theme_toggle_btn",
    )

    st.markdown('<div class="section-tag">DATA</div>', unsafe_allow_html=True)
    n_total = st.slider("Total timesteps", 3000, 12000, 6000, 500)
    seed = st.slider("Random seed", 1, 100, 42)

    st.markdown('<div class="section-tag">ARCHITECTURE</div>', unsafe_allow_html=True)
    lookback = st.slider("Lookback window", 50, 200, 100, 10)
    horizon = st.slider("Forecast horizon", 5, 30, 10, 5)
    hidden = st.select_slider("Hidden channels", [32, 64, 128], value=64)
    n_layers = st.slider("TCN depth", 2, 6, 4)
    kernel_size = st.select_slider("Kernel size", [3, 5, 7, 9], value=7)

    st.markdown('<div class="section-tag">TRAINING</div>', unsafe_allow_html=True)
    epochs = st.slider("Epochs", 20, 120, 60, 10)

    st.markdown("---")
    run_btn = st.button("Run Experiment", use_container_width=True)


# ═══════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════

# ── Hero ──
st.markdown("""
<div style="padding: 0.5rem 0 0.5rem 0">
    <div class="hero-title">Regime-Adaptive TCN</div>
    <div class="hero-subtitle">
        When distributions shift, standard convolutions break.&nbsp;
        We fix the kernels at inference time.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Architecture Overview (always visible) ──
st.markdown('<div class="section-tag">ARCHITECTURE</div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">How RA-TCN works</div>', unsafe_allow_html=True)

st.markdown("""
<div class="arch-box">
    <div class="arch-flow">
        <div class="arch-node">Input<br/>Window</div>
        <div class="arch-arrow">&rarr;</div>
        <div class="arch-node highlight">Shift<br/>Detector</div>
        <div class="arch-arrow">&rarr;</div>
        <div class="arch-node highlight-amber">Regime<br/>Memory Bank</div>
        <div class="arch-arrow">&rarr;</div>
        <div class="arch-node">Shift<br/>Embedding</div>
    </div>
    <div style="text-align:center;color:#3d4166;margin:1rem 0;font-size:1.5rem">&darr;</div>
    <div class="arch-flow">
        <div class="arch-node">Input<br/>Projection</div>
        <div class="arch-arrow">&rarr;</div>
        <div class="arch-node">Dilated Causal<br/>Conv Block</div>
        <div class="arch-arrow">&rarr;</div>
        <div class="arch-node highlight">FiLM<br/>Modulation</div>
        <div class="arch-arrow">&times; N &rarr;</div>
        <div class="arch-node">Prediction<br/>Head</div>
    </div>
    <div style="margin-top:1.5rem;font-family:'DM Mono',monospace;font-size:0.72rem;color:#3d4166;text-align:center">
        <span style="color:#00e5c7">Shift Detector</span> extracts [mean, std, skew, kurtosis] &rarr;
        <span style="color:#ffb224">Memory Bank</span> retrieves closest regime prototype &rarr;
        <span style="color:#00e5c7">FiLM</span> modulates conv features with &gamma; &middot; h + &beta;
    </div>
</div>
""", unsafe_allow_html=True)

# ── Key insight panel ──
col_i1, col_i2 = st.columns(2)
with col_i1:
    st.markdown("""
    <div class="metric-card red">
        <div class="metric-label">The Problem</div>
        <div style="font-family:'Sora',sans-serif;font-size:0.9rem;color:var(--text-primary);margin-top:0.5rem;line-height:1.6">
            TCN convolutional filters are <span style="color:#ff4d6a;font-weight:600">frozen after training</span>.
            When real-world data undergoes a regime change, predictions collapse because filters
            keep applying patterns from the old distribution.
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_i2:
    st.markdown("""
    <div class="metric-card cyan">
        <div class="metric-label">Our Solution</div>
        <div style="font-family:'Sora',sans-serif;font-size:0.9rem;color:var(--text-primary);margin-top:0.5rem;line-height:1.6">
            RA-TCN uses a <span style="color:#00e5c7;font-weight:600">Shift Detector + FiLM Modulation</span>
            to re-calibrate convolution features on-the-fly. A Regime Memory Bank retrieves
            learned prototypes for instant adaptation to recurring regimes.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════
#  EXPERIMENT
# ═══════════════════════════════════════

if run_btn:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem">
        <div class="section-header" style="margin:0">Experiment Running</div>
        <div class="status-pill live"><div class="dot"></div>LIVE</div>
    </div>
    """, unsafe_allow_html=True)

    # Generate data
    series, regimes = generate_regime_data(n_total=n_total, seed=seed)
    seg_len = n_total // 3

    full_ds = TimeSeriesDataset(series, lookback=lookback, horizon=horizon)
    regime_a_end = seg_len - lookback - horizon
    regime_a_ds = Subset(full_ds, list(range(0, max(1, regime_a_end))))
    regime_a_loader = DataLoader(regime_a_ds, batch_size=64, shuffle=True)
    full_loader = DataLoader(full_ds, batch_size=64, shuffle=True)

    # ── Show raw data ──
    st.markdown('<div class="section-tag">DATA</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Generated Time Series</div>', unsafe_allow_html=True)

    fig_data = go.Figure()
    t = np.arange(len(series))

    # Regime shading
    regime_colors = CLR_REGIME_BG
    regime_borders = CLR_REGIME_BORDER
    regime_labels = ["Regime A: Calm", "Regime B: Shock", "Regime C: Recovery"]
    for i in range(3):
        start = i * seg_len
        end = (i + 1) * seg_len
        fig_data.add_vrect(x0=start, x1=end, fillcolor=regime_colors[i],
                           line_width=0, layer="below")
        fig_data.add_vline(x=start if i > 0 else -1, line_dash="dot",
                           line_color=regime_borders[i], line_width=1, opacity=0.5)
        fig_data.add_annotation(
            x=(start + end) / 2, y=max(series) * 1.05,
            text=regime_labels[i], showarrow=False,
            font=dict(size=11, color=regime_borders[i], family="DM Mono, monospace"),
        )

    fig_data.add_trace(go.Scatter(
        x=t, y=series, mode="lines",
        line=dict(color=CLR_GROUND, width=1),
        name="Time Series",
    ))
    fig_data.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        title=dict(text="Synthetic Regime-Shift Data", font=dict(size=14, color=CLR_TITLE)),
        showlegend=False,
    )
    st.plotly_chart(fig_data, use_container_width=True)

    # ── Training ──
    st.markdown('<div class="section-tag">TRAINING</div>', unsafe_allow_html=True)

    col_t1, col_t2 = st.columns(2)

    # Train Vanilla TCN
    with col_t1:
        st.markdown("""
        <div style="font-family:'Sora',sans-serif;font-size:1rem;color:#ff4d6a;font-weight:600;margin-bottom:0.5rem">
            Vanilla TCN <span style="font-size:0.7rem;color:#6b7094;font-weight:400">(trained on Regime A only)</span>
        </div>
        """, unsafe_allow_html=True)
        prog_v = st.progress(0)
        status_v = st.empty()
        vanilla = VanillaTCN(hidden=hidden, n_layers=n_layers, kernel_size=kernel_size,
                             horizon=horizon).to(DEVICE)

        def cb_v(ep, total, loss):
            prog_v.progress(ep / total)
            status_v.markdown(
                f'<div class="train-log"><span class="epoch">epoch {ep}/{total}</span> '
                f'&nbsp;&nbsp;<span class="loss">loss={loss:.6f}</span></div>',
                unsafe_allow_html=True,
            )

        logs_v = train_model(vanilla, regime_a_loader, epochs, progress_cb=cb_v)
        prog_v.progress(1.0)

    # Train RA-TCN
    with col_t2:
        st.markdown("""
        <div style="font-family:'Sora',sans-serif;font-size:1rem;color:#00e5c7;font-weight:600;margin-bottom:0.5rem">
            RA-TCN <span style="font-size:0.7rem;color:#6b7094;font-weight:400">(trained on ALL regimes)</span>
        </div>
        """, unsafe_allow_html=True)
        prog_r = st.progress(0)
        status_r = st.empty()
        ratcn = RATCN(hidden=hidden, n_layers=n_layers, kernel_size=kernel_size,
                      horizon=horizon).to(DEVICE)

        def cb_r(ep, total, loss):
            prog_r.progress(ep / total)
            status_r.markdown(
                f'<div class="train-log"><span class="epoch">epoch {ep}/{total}</span> '
                f'&nbsp;&nbsp;<span class="loss">loss={loss:.6f}</span></div>',
                unsafe_allow_html=True,
            )

        logs_r = train_model(ratcn, full_loader, epochs, progress_cb=cb_r)
        prog_r.progress(1.0)

    # ── Training curves ──
    fig_loss = make_subplots(rows=1, cols=1)
    fig_loss.add_trace(go.Scatter(
        x=[l[0] for l in logs_v], y=[l[1] for l in logs_v],
        mode="lines", name="Vanilla TCN",
        line=dict(color=CLR_RED, width=2),
    ))
    fig_loss.add_trace(go.Scatter(
        x=[l[0] for l in logs_r], y=[l[1] for l in logs_r],
        mode="lines", name="RA-TCN",
        line=dict(color=CLR_CYAN, width=2),
    ))
    fig_loss.update_layout(
        **PLOTLY_LAYOUT,
        height=250,
        title=dict(text="Training Loss", font=dict(size=13, color=CLR_TITLE)),
        xaxis_title="Epoch",
        yaxis_title="MSE Loss",
    )
    st.plotly_chart(fig_loss, use_container_width=True)

    # ── Predictions ──
    st.markdown('<div class="section-tag">RESULTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Prediction Comparison</div>', unsafe_allow_html=True)

    preds_vanilla = predict_rolling(vanilla, series, lookback, horizon)
    preds_ratcn = predict_rolling(ratcn, series, lookback, horizon)

    # Main comparison chart
    fig_pred = go.Figure()

    for i in range(3):
        start = i * seg_len
        end = (i + 1) * seg_len
        fig_pred.add_vrect(x0=start, x1=end, fillcolor=regime_colors[i], line_width=0, layer="below")
        if i > 0:
            fig_pred.add_vline(x=start, line_dash="dot", line_color=CLR_ANN, line_width=1.5, opacity=0.7)
            fig_pred.add_annotation(
                x=start, y=max(series) * 1.1,
                text="DISTRIBUTION SHIFT",
                showarrow=True, arrowhead=2, arrowcolor=CLR_ANN,
                font=dict(size=9, color=CLR_ANN, family="DM Mono"),
                ay=-30,
            )

    fig_pred.add_trace(go.Scatter(
        x=t, y=series, mode="lines",
        line=dict(color=CLR_GROUND, width=1, dash="dot"),
        name="Ground Truth", opacity=0.4,
    ))
    fig_pred.add_trace(go.Scatter(
        x=t, y=preds_vanilla, mode="lines",
        line=dict(color=CLR_RED, width=2),
        name="Vanilla TCN",
    ))
    fig_pred.add_trace(go.Scatter(
        x=t, y=preds_ratcn, mode="lines",
        line=dict(color=CLR_CYAN, width=2.5),
        name="RA-TCN (ours)",
    ))
    fig_pred.update_layout(
        **PLOTLY_LAYOUT,
        height=400,
        title=dict(text="Rolling Forecast: Vanilla TCN vs RA-TCN", font=dict(size=14, color=CLR_TITLE)),
        xaxis_title="Timestep",
        yaxis_title="Predicted Value",
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # ── Per-regime metrics ──
    mse_v = mse_per_regime(preds_vanilla, series, regimes)
    mse_r = mse_per_regime(preds_ratcn, series, regimes)

    st.markdown('<div class="section-tag">METRICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Per-Regime Performance</div>', unsafe_allow_html=True)

    col_m1, col_m2, col_m3 = st.columns(3)

    for col, r, label, accent in [
        (col_m1, 0, "Regime A: Calm", "cyan"),
        (col_m2, 1, "Regime B: Shock", "red"),
        (col_m3, 2, "Regime C: Recovery", "amber"),
    ]:
        imp = ((mse_v[r] - mse_r[r]) / mse_v[r] * 100) if mse_v[r] > 0 else 0
        delta_class = "positive" if imp > 0 else "negative"
        sign = "+" if imp > 0 else ""

        with col:
            st.markdown(f"""
            <div class="metric-card {accent}">
                <div class="metric-label">{label}</div>
                <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-top:0.5rem">
                    <div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#3d4166;margin-bottom:2px">VANILLA MSE</div>
                        <div class="metric-value red">{mse_v[r]:.4f}</div>
                    </div>
                    <div style="text-align:right">
                        <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#3d4166;margin-bottom:2px">RA-TCN MSE</div>
                        <div class="metric-value cyan">{mse_r[r]:.4f}</div>
                    </div>
                </div>
                <div class="metric-delta {delta_class}" style="margin-top:0.8rem;text-align:center;font-size:1rem">
                    {sign}{imp:.1f}% improvement
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Error over time ──
    st.markdown('<div class="section-header">Error Dynamics</div>', unsafe_allow_html=True)

    err_v = np.abs(preds_vanilla - series)
    err_r = np.abs(preds_ratcn - series)

    def rolling_nanmean(arr, w=80):
        out = np.full_like(arr, np.nan)
        for i in range(w, len(arr)):
            chunk = arr[i-w:i]
            valid = chunk[~np.isnan(chunk)]
            if len(valid) > 0:
                out[i] = np.mean(valid)
        return out

    roll_v = rolling_nanmean(err_v)
    roll_r = rolling_nanmean(err_r)

    fig_err = go.Figure()
    for i in range(3):
        start = i * seg_len
        end = (i + 1) * seg_len
        fig_err.add_vrect(x0=start, x1=end, fillcolor=regime_colors[i], line_width=0, layer="below")

    fig_err.add_trace(go.Scatter(
        x=t, y=roll_v, mode="lines", fill="tozeroy",
        line=dict(color=CLR_RED, width=1.5),
        fillcolor=CLR_FILL_RED,
        name="Vanilla Error",
    ))
    fig_err.add_trace(go.Scatter(
        x=t, y=roll_r, mode="lines", fill="tozeroy",
        line=dict(color=CLR_CYAN, width=1.5),
        fillcolor=CLR_FILL_CYAN,
        name="RA-TCN Error",
    ))
    fig_err.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        title=dict(text="Rolling Mean Absolute Error (window=80)", font=dict(size=13, color=CLR_TITLE)),
        xaxis_title="Timestep",
        yaxis_title="MAE",
    )
    st.plotly_chart(fig_err, use_container_width=True)

    # ── Verdict ──
    total_mse_v = np.nanmean((preds_vanilla - series) ** 2)
    total_mse_r = np.nanmean((preds_ratcn - series) ** 2)
    total_imp = (total_mse_v - total_mse_r) / total_mse_v * 100

    st.markdown(f"""
    <div class="arch-box" style="text-align:center;margin-top:1rem">
        <div class="metric-label">Overall Verdict</div>
        <div style="font-family:'Instrument Serif',serif;font-size:2.5rem;color:#00e5c7;margin:0.5rem 0">
            {total_imp:+.1f}% better
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:0.8rem;color:#6b7094">
            RA-TCN total MSE: <span style="color:#00e5c7">{total_mse_r:.4f}</span>
            &nbsp;&middot;&nbsp;
            Vanilla TCN total MSE: <span style="color:#ff4d6a">{total_mse_v:.4f}</span>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#3d4166;margin-top:1rem">
            The Vanilla TCN collapses at regime boundaries. RA-TCN's FiLM-modulated kernels
            adapt to new distributions in real time.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Save models to session state for testing ──
    st.session_state["vanilla_model"] = vanilla
    st.session_state["ratcn_model"] = ratcn
    st.session_state["lookback"] = lookback
    st.session_state["horizon"] = horizon
    st.session_state["series"] = series
    st.session_state["regimes"] = regimes
    st.session_state["seg_len"] = seg_len

# ═══════════════════════════════════════
#  INTERACTIVE TEST PANEL
# ═══════════════════════════════════════

if "vanilla_model" in st.session_state:
    st.markdown("---")
    st.markdown('<div class="section-tag">PLAYGROUND</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Test With Your Own Input</div>', unsafe_allow_html=True)

    _lookback = st.session_state["lookback"]
    _horizon = st.session_state["horizon"]
    _series = st.session_state["series"]
    _seg_len = st.session_state["seg_len"]

    st.markdown(f"""
    <div class="arch-box" style="padding:1.2rem 1.5rem">
        <div style="display:flex;gap:2rem;flex-wrap:wrap">
            <div>
                <div class="metric-label">Required Input Size</div>
                <div style="font-family:'Sora',sans-serif;font-size:1.5rem;font-weight:700;color:#ffb224">{_lookback} values</div>
            </div>
            <div>
                <div class="metric-label">Output Size</div>
                <div style="font-family:'Sora',sans-serif;font-size:1.5rem;font-weight:700;color:#ffb224">{_horizon} values</div>
            </div>
            <div>
                <div class="metric-label">Format</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.85rem;color:#6b7094;margin-top:0.3rem">
                    Comma-separated floats, e.g. 0.2, 0.5, 0.8, ...
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input method selector ──
    input_method = st.radio(
        "How do you want to provide input?",
        ["Pick a slice from the generated data", "Type custom values"],
        horizontal=True,
        label_visibility="collapsed",
    )

    test_values = None
    ground_truth = None

    if input_method == "Pick a slice from the generated data":
        # Let user pick a start index from the time series
        max_start = len(_series) - _lookback - _horizon
        col_sl1, col_sl2 = st.columns([3, 1])
        with col_sl1:
            start_idx = st.slider(
                "Start position in time series",
                0, max_start, _seg_len + 500,  # default to middle of Regime B
                help=f"Regime A: 0–{_seg_len-1}  |  Regime B: {_seg_len}–{2*_seg_len-1}  |  Regime C: {2*_seg_len}–{3*_seg_len-1}",
            )
        with col_sl2:
            # Show which regime this falls in
            if start_idx < _seg_len:
                regime_name, regime_color = "Regime A (Calm)", CLR_BLUE
            elif start_idx < 2 * _seg_len:
                regime_name, regime_color = "Regime B (Shock)", CLR_RED
            else:
                regime_name, regime_color = "Regime C (Recovery)", CLR_CYAN
            st.markdown(f"""
            <div style="margin-top:1.2rem">
                <div class="metric-label">Regime</div>
                <div style="font-family:'Sora',sans-serif;font-size:1rem;font-weight:600;color:{regime_color}">
                    {regime_name}
                </div>
            </div>
            """, unsafe_allow_html=True)

        test_values = _series[start_idx : start_idx + _lookback].tolist()
        ground_truth = _series[start_idx + _lookback : start_idx + _lookback + _horizon].tolist()

    else:
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#3d4166;margin-bottom:0.5rem">
            Enter exactly {_lookback} comma-separated values below.
            Try values near 0 (calm), near 3-5 (shock), or near -1 (recovery).
        </div>
        """, unsafe_allow_html=True)

        # Generate example strings for each regime
        example_a = ", ".join([f"{v:.2f}" for v in _series[100:100+_lookback]])
        example_b = ", ".join([f"{v:.2f}" for v in _series[_seg_len+100:_seg_len+100+_lookback]])

        custom_input = st.text_area(
            "Input values",
            height=120,
            placeholder=f"e.g. {example_a[:80]}...",
            label_visibility="collapsed",
        )

        if custom_input.strip():
            try:
                parsed = [float(v.strip()) for v in custom_input.strip().split(",") if v.strip()]
                if len(parsed) == _lookback:
                    test_values = parsed
                elif len(parsed) > _lookback:
                    st.warning(f"You entered {len(parsed)} values — using the first {_lookback}.")
                    test_values = parsed[:_lookback]
                else:
                    st.error(f"Need exactly {_lookback} values, got {len(parsed)}. Add {_lookback - len(parsed)} more.")
            except ValueError:
                st.error("Could not parse input. Use comma-separated numbers like: 0.2, 0.5, 0.8, ...")

    # ── Run prediction ──
    if test_values is not None:
        predict_btn = st.button("Predict", use_container_width=True, key="predict_btn")

        if predict_btn:
            _vanilla = st.session_state["vanilla_model"]
            _ratcn = st.session_state["ratcn_model"]

            # Prepare input tensor
            x_tensor = torch.tensor(test_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

            # Predict
            _vanilla.eval()
            _ratcn.eval()
            with torch.no_grad():
                pred_v = _vanilla(x_tensor).cpu().numpy().flatten()
                pred_r = _ratcn(x_tensor).cpu().numpy().flatten()

            # ── Show input stats (what ShiftDetector sees) ──
            arr = np.array(test_values)
            st.markdown('<div class="section-tag">SHIFT DETECTOR VIEW</div>', unsafe_allow_html=True)
            det_cols = st.columns(4)
            stat_names = ["Mean", "Std", "Skew", "Kurtosis"]
            stat_vals = [
                np.mean(arr),
                np.std(arr),
                float(np.mean(((arr - np.mean(arr)) / (np.std(arr) + 1e-6)) ** 3)),
                float(np.mean(((arr - np.mean(arr)) / (np.std(arr) + 1e-6)) ** 4) - 3.0),
            ]
            stat_colors = [CLR_BLUE, CLR_AMBER, CLR_CYAN, CLR_RED]
            for i, (col, name, val, clr) in enumerate(zip(det_cols, stat_names, stat_vals, stat_colors)):
                with col:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:center">
                        <div class="metric-label">{name}</div>
                        <div style="font-family:'Sora',sans-serif;font-size:1.8rem;font-weight:700;color:{clr}">
                            {val:+.3f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Input + Prediction chart ──
            st.markdown('<div class="section-tag">PREDICTIONS</div>', unsafe_allow_html=True)

            fig_test = go.Figure()

            # Input window
            t_input = list(range(_lookback))
            fig_test.add_trace(go.Scatter(
                x=t_input, y=test_values, mode="lines",
                line=dict(color=CLR_GROUND, width=1.5),
                name="Input Window",
            ))

            # Prediction zone
            t_pred = list(range(_lookback, _lookback + _horizon))

            # Ground truth (if available)
            if ground_truth is not None:
                fig_test.add_trace(go.Scatter(
                    x=t_pred, y=ground_truth, mode="lines+markers",
                    line=dict(color=CLR_GROUND, width=2, dash="dot"),
                    marker=dict(size=6, color=CLR_GROUND),
                    name="Ground Truth",
                ))

            # Vanilla prediction
            fig_test.add_trace(go.Scatter(
                x=t_pred, y=pred_v.tolist(), mode="lines+markers",
                line=dict(color=CLR_RED, width=2.5),
                marker=dict(size=7, symbol="x", color=CLR_RED),
                name="Vanilla TCN",
            ))

            # RA-TCN prediction
            fig_test.add_trace(go.Scatter(
                x=t_pred, y=pred_r.tolist(), mode="lines+markers",
                line=dict(color=CLR_CYAN, width=2.5),
                marker=dict(size=7, symbol="diamond", color=CLR_CYAN),
                name="RA-TCN",
            ))

            # Separator line
            fig_test.add_vline(
                x=_lookback - 0.5, line_dash="dash", line_color=CLR_AMBER,
                line_width=2, opacity=0.7,
            )
            fig_test.add_annotation(
                x=_lookback - 0.5, y=max(test_values) * 1.05,
                text="PREDICTION STARTS",
                showarrow=False,
                font=dict(size=10, color=CLR_AMBER, family="DM Mono"),
            )

            fig_test.update_layout(
                **PLOTLY_LAYOUT,
                height=400,
                title=dict(
                    text="Your Input + Model Predictions",
                    font=dict(size=14, color=CLR_TITLE),
                ),
                xaxis_title="Timestep",
                yaxis_title="Value",
            )
            st.plotly_chart(fig_test, use_container_width=True)

            # ── Comparison table ──
            st.markdown('<div class="section-tag">COMPARISON</div>', unsafe_allow_html=True)

            col_pr1, col_pr2 = st.columns(2)

            with col_pr1:
                rows_html = ""
                for i in range(len(pred_v)):
                    gt_val = f"{ground_truth[i]:.4f}" if ground_truth else "—"
                    gt_color = CLR_TITLE if ground_truth else "var(--text-muted)"
                    rows_html += f"""
                    <tr>
                        <td style="color:#6b7094">t+{i+1}</td>
                        <td style="color:{gt_color}">{gt_val}</td>
                        <td style="color:#ff4d6a">{pred_v[i]:.4f}</td>
                        <td style="color:#00e5c7">{pred_r[i]:.4f}</td>
                    </tr>"""

                st.markdown(f"""
                <div class="arch-box" style="padding:1rem;overflow-x:auto">
                    <table style="width:100%;border-collapse:collapse;font-family:'DM Mono',monospace;font-size:0.8rem">
                        <thead>
                            <tr style="border-bottom:1px solid #1e2035">
                                <th style="color:#3d4166;text-align:left;padding:6px 10px">Step</th>
                                <th style="color:#3d4166;text-align:right;padding:6px 10px">Truth</th>
                                <th style="color:#3d4166;text-align:right;padding:6px 10px">Vanilla</th>
                                <th style="color:#3d4166;text-align:right;padding:6px 10px">RA-TCN</th>
                            </tr>
                        </thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </div>
                """, unsafe_allow_html=True)

            with col_pr2:
                if ground_truth is not None:
                    gt_arr = np.array(ground_truth)
                    mse_v_test = float(np.mean((pred_v - gt_arr) ** 2))
                    mse_r_test = float(np.mean((pred_r - gt_arr) ** 2))
                    mae_v_test = float(np.mean(np.abs(pred_v - gt_arr)))
                    mae_r_test = float(np.mean(np.abs(pred_r - gt_arr)))
                    imp_mse = (mse_v_test - mse_r_test) / mse_v_test * 100 if mse_v_test > 0 else 0

                    winner_color = CLR_CYAN if imp_mse > 0 else CLR_RED
                    winner_name = "RA-TCN" if imp_mse > 0 else "Vanilla TCN"

                    st.markdown(f"""
                    <div class="metric-card cyan" style="margin-bottom:1rem">
                        <div class="metric-label">Mean Squared Error</div>
                        <div style="display:flex;justify-content:space-between;margin-top:0.8rem">
                            <div>
                                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#3d4166">VANILLA</div>
                                <div style="font-family:'Sora',sans-serif;font-size:1.4rem;font-weight:700;color:#ff4d6a">{mse_v_test:.4f}</div>
                            </div>
                            <div style="text-align:right">
                                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#3d4166">RA-TCN</div>
                                <div style="font-family:'Sora',sans-serif;font-size:1.4rem;font-weight:700;color:#00e5c7">{mse_r_test:.4f}</div>
                            </div>
                        </div>
                    </div>

                    <div class="metric-card amber" style="margin-bottom:1rem">
                        <div class="metric-label">Mean Absolute Error</div>
                        <div style="display:flex;justify-content:space-between;margin-top:0.8rem">
                            <div>
                                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#3d4166">VANILLA</div>
                                <div style="font-family:'Sora',sans-serif;font-size:1.4rem;font-weight:700;color:#ff4d6a">{mae_v_test:.4f}</div>
                            </div>
                            <div style="text-align:right">
                                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#3d4166">RA-TCN</div>
                                <div style="font-family:'Sora',sans-serif;font-size:1.4rem;font-weight:700;color:#00e5c7">{mae_r_test:.4f}</div>
                            </div>
                        </div>
                    </div>

                    <div class="arch-box" style="text-align:center;padding:1.2rem">
                        <div class="metric-label">Winner</div>
                        <div style="font-family:'Instrument Serif',serif;font-size:1.8rem;color:{winner_color};margin:0.3rem 0">
                            {winner_name}
                        </div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.85rem;color:#6b7094">
                            {abs(imp_mse):.1f}% {"lower" if imp_mse > 0 else "higher"} MSE
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # No ground truth — just show predictions side by side
                    st.markdown(f"""
                    <div class="arch-box" style="text-align:center;padding:1.5rem">
                        <div class="metric-label">No Ground Truth Available</div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.8rem;color:#6b7094;margin-top:0.5rem;line-height:1.8">
                            Custom input has no known future values.<br/>
                            Use "Pick a slice" mode to see error comparison.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Per-step error bar chart (when ground truth exists) ──
            if ground_truth is not None:
                gt_arr = np.array(ground_truth)
                err_v_steps = np.abs(pred_v - gt_arr)
                err_r_steps = np.abs(pred_r - gt_arr)

                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=[f"t+{i+1}" for i in range(_horizon)],
                    y=err_v_steps.tolist(),
                    name="Vanilla Error",
                    marker_color=CLR_RED,
                    opacity=0.85,
                ))
                fig_bar.add_trace(go.Bar(
                    x=[f"t+{i+1}" for i in range(_horizon)],
                    y=err_r_steps.tolist(),
                    name="RA-TCN Error",
                    marker_color=CLR_CYAN,
                    opacity=0.85,
                ))
                fig_bar.update_layout(
                    **PLOTLY_LAYOUT,
                    height=300,
                    barmode="group",
                    title=dict(text="Per-Step Absolute Error", font=dict(size=13, color=CLR_TITLE)),
                    xaxis_title="Forecast Step",
                    yaxis_title="Absolute Error",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

if "vanilla_model" not in st.session_state and not run_btn:
    # ── Landing state ──
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem">
        <div style="font-family:'Instrument Serif',serif;font-size:2rem;color:var(--text-primary);margin-bottom:1rem">
            Configure & Run
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:0.85rem;color:#3d4166;max-width:500px;margin:0 auto;line-height:1.8">
            Adjust the experiment parameters in the sidebar,<br/>
            then hit <span style="color:#00e5c7">Run Experiment</span> to see the regime-adaptive
            advantage in action.
        </div>
    </div>
    """, unsafe_allow_html=True)
