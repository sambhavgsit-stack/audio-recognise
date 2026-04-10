import streamlit as st
import numpy as np
import librosa
import plotly.graph_objects as go
import os
import tempfile
from datetime import datetime
import pandas as pd

st.set_page_config(
    page_title="VoiceIQ · Gender Recognition",
    page_icon="🎙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,700;1,9..144,400;1,9..144,700&family=JetBrains+Mono:wght@300;400;500;600&family=Syne:wght@400;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box !important; }

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    background-color: #010409 !important;
    color: #e6edf3 !important;
}

/* Hide Streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d !important;
    min-width: 220px !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117 !important;
    border-radius: 8px !important;
    border: 1px solid #21262d !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 6px !important;
    color: #a0aab4 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(212,134,10,.2) !important;
    color: #f5c842 !important;
    border: 1px solid rgba(212,134,10,.4) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 1.2rem 0 0 !important; }

/* Buttons */
.stButton > button {
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    border: none !important;
    transition: all .18s !important;
}
.stButton > button[kind="primary"] {
    background: #d4860a !important;
    color: #000 !important;
    padding: 0.75rem 1.5rem !important;
    width: 100% !important;
}
.stButton > button[kind="primary"]:hover {
    background: #f5c842 !important;
    box-shadow: 0 0 24px rgba(212,134,10,.45) !important;
}
.stButton > button[kind="secondary"] {
    background: #161b22 !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
}
.stButton > button[kind="secondary"]:hover { background: #21262d !important; color: #f5c842 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.55rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #a0aab4 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    padding: 0.9rem !important;
    text-align: center !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.5rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #a0aab4 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: #f5c842 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* Divider */
hr { border-color: #21262d !important; margin: 1rem 0 !important; }

/* Audio widget */
audio { width: 100% !important; background: #0d1117 !important; border-radius: 6px !important; }

/* Plotly charts transparent bg */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = "gender_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        m = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return m
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None

model = load_model()
MODEL_LOADED = model is not None

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Audio helpers ─────────────────────────────────────────────────────────────
def clean_audio(y, sr):
    y, _ = librosa.effects.trim(y, top_db=20)
    y = librosa.util.normalize(y)
    return librosa.util.fix_length(y, size=sr * 3)

def extract_features(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
    d = librosa.power_to_db(mel, ref=np.max)
    return (d - d.mean()) / (d.std() + 1e-6)

def make_mel_fig(y, sr, gender=""):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
    d = librosa.power_to_db(mel, ref=np.max)
    if gender == "Male":
        cs = [[0,'#0a0e1a'],[0.3,'#0d3b5e'],[0.6,'#1a7fbf'],[0.85,'#38b8f2'],[1,'#a0e8ff']]
    elif gender == "Female":
        cs = [[0,'#1a0a10'],[0.3,'#6b1a3a'],[0.6,'#c0395a'],[0.85,'#f07090'],[1,'#ffc0d0']]
    else:
        cs = [[0,'#0d0d0d'],[0.3,'#2a1f0a'],[0.6,'#7a4f0e'],[0.85,'#d4860a'],[1,'#f5c842']]
    fig = go.Figure(go.Heatmap(z=d, colorscale=cs, showscale=False))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,t=0,b=0), height=180,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

def make_gauge(conf, gender):
    col = '#38b8f2' if gender == 'Male' else '#f07090'
    s1 = 'rgba(56,184,242,.12)' if gender == 'Male' else 'rgba(240,112,144,.12)'
    s2 = 'rgba(56,184,242,.28)' if gender == 'Male' else 'rgba(240,112,144,.28)'
    s3 = 'rgba(56,184,242,.50)' if gender == 'Male' else 'rgba(240,112,144,.50)'
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=conf * 100,
        number={'suffix': '%', 'font': {'color': col, 'size': 28, 'family': 'Fraunces'}},
        gauge=dict(
            axis=dict(range=[50, 100], tickcolor='#30363d',
                      tickfont=dict(color='#a0aab4', size=8)),
            bar=dict(color=col, thickness=0.22),
            bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)',
            steps=[dict(range=[50,70], color=s1), dict(range=[70,85], color=s2),
                   dict(range=[85,100], color=s3)]
        )
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e6edf3'),
        margin=dict(l=12, r=12, t=20, b=4), height=180
    )
    return fig

# ── Chart helpers ─────────────────────────────────────────────────────────────
CHART_BG = 'rgba(0,0,0,0)'
GRID_COL = 'rgba(255,255,255,0.06)'
TICK_FONT = dict(size=9, color='#a0aab4', family='JetBrains Mono')

def _no_data(h=200):
    fig = go.Figure()
    fig.add_annotation(text="No data yet", xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=12, color='#6e7681', family='JetBrains Mono'))
    fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                      height=h, margin=dict(l=8,r=8,t=8,b=8))
    return fig

def chart_timeline(hist):
    if not hist: return _no_data()
    df = pd.DataFrame(hist)
    cols = ['#38b8f2' if g == 'Male' else '#f07090' for g in df['gender']]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(df)+1)), y=df['confidence'] * 100,
        mode='lines+markers',
        marker=dict(size=7, color=cols, line=dict(width=1.5, color='#0d1117')),
        line=dict(width=2, color='#d4860a'),
        hovertemplate='#%{x} · %{y:.1f}%<extra></extra>'
    ))
    fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, height=220,
        margin=dict(l=8,r=8,t=8,b=8), showlegend=False,
        xaxis=dict(showgrid=True, gridcolor=GRID_COL, zeroline=False, tickfont=TICK_FONT),
        yaxis=dict(range=[40,105], showgrid=True, gridcolor=GRID_COL, zeroline=False,
                   ticksuffix='%', tickfont=TICK_FONT))
    return fig

def chart_gender(hist):
    if not hist: return _no_data()
    df = pd.DataFrame(hist)
    gc = df['gender'].value_counts()
    cols = ['#38b8f2' if g == 'Male' else '#f07090' for g in gc.index]
    fig = go.Figure(go.Pie(
        labels=gc.index, values=gc.values, hole=0.55,
        marker=dict(colors=cols, line=dict(color='#0d1117', width=3)),
        textinfo='label+percent',
        textfont=dict(size=10, family='JetBrains Mono', color='#e6edf3')
    ))
    fig.update_layout(paper_bgcolor=CHART_BG, height=220, margin=dict(l=8,r=8,t=8,b=8),
        legend=dict(font=dict(size=9, color='#a0aab4', family='JetBrains Mono'),
                    bgcolor='rgba(0,0,0,0)'))
    return fig

def chart_dist(hist):
    if not hist: return _no_data()
    df = pd.DataFrame(hist)
    fig = go.Figure(go.Histogram(
        x=df['confidence'] * 100, nbinsx=12,
        marker=dict(color='#d4860a', opacity=0.9, line=dict(color='#0d1117', width=1))
    ))
    fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, height=220,
        margin=dict(l=8,r=8,t=8,b=8),
        xaxis=dict(showgrid=False, zeroline=False, ticksuffix='%', tickfont=TICK_FONT),
        yaxis=dict(showgrid=True, gridcolor=GRID_COL, zeroline=False, tickfont=TICK_FONT))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:1.4rem 1.2rem 1rem; border-bottom:1px solid #21262d; margin:-1rem -1rem 0;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.5rem;letter-spacing:3px;text-transform:uppercase;color:#d4860a;margin-bottom:0.3rem;">AI Voice Analysis</div>
      <div style="font-family:'Fraunces',serif;font-size:1.6rem;font-weight:700;font-style:italic;color:#f0f6fc;line-height:1;letter-spacing:-1px;">Voice<span style="color:#f5c842;font-style:normal;">IQ</span></div>
    </div>
    """, unsafe_allow_html=True)

    if MODEL_LOADED:
        st.markdown("""
        <div style="margin:1rem 0 0.5rem;display:inline-flex;align-items:center;gap:0.35rem;padding:0.25rem 0.55rem;border-radius:100px;background:rgba(57,211,83,.12);border:1px solid rgba(57,211,83,.3);font-family:'JetBrains Mono',monospace;font-size:0.48rem;letter-spacing:1.5px;text-transform:uppercase;color:#56d364;">
          <span style="width:5px;height:5px;border-radius:50%;background:#56d364;display:inline-block;"></span>Model Ready
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin:1rem 0 0.5rem;display:inline-flex;align-items:center;gap:0.35rem;padding:0.25rem 0.55rem;border-radius:100px;background:rgba(240,112,144,.12);border:1px solid rgba(240,112,144,.3);font-family:'JetBrains Mono',monospace;font-size:0.48rem;letter-spacing:1.5px;text-transform:uppercase;color:#f07090;">
          <span style="width:5px;height:5px;border-radius:50%;background:#f07090;display:inline-block;"></span>Model Missing
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0.5rem 0;font-family:'JetBrains Mono',monospace;font-size:0.45rem;letter-spacing:2.5px;text-transform:uppercase;color:#b0bac4;margin-top:0.8rem;">Navigation</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#eaf0f6;padding:0.5rem 0;">
      🎙 Analyzer &nbsp;·&nbsp; 📊 Dashboard &nbsp;·&nbsp; 📜 History &nbsp;·&nbsp; ℹ️ About
    </div>
    <p style="font-size:0.7rem;color:#6e7681;margin-top:0.5rem;">Use the tabs above to navigate between sections.</p>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    h = st.session_state.history
    n = len(h)
    if n:
        df_s = pd.DataFrame(h)
        mc = int((df_s['gender'] == 'Male').sum())
        fc = int((df_s['gender'] == 'Female').sum())
        avg = float(df_s['confidence'].mean() * 100)
    else:
        mc = fc = 0; avg = 0.0

    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.45rem;letter-spacing:2.5px;text-transform:uppercase;color:#b0bac4;margin-bottom:0.5rem;">Session Stats</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;">
      <div style="background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:0.6rem;text-align:center;">
        <div style="font-family:'Fraunces',serif;font-size:1.4rem;font-weight:700;color:#f5c842;">{n}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.42rem;letter-spacing:1px;text-transform:uppercase;color:#a0aab4;">Total</div>
      </div>
      <div style="background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:0.6rem;text-align:center;">
        <div style="font-family:'Fraunces',serif;font-size:1.4rem;font-weight:700;color:#58ccf5;">{mc}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.42rem;letter-spacing:1px;text-transform:uppercase;color:#a0aab4;">Male</div>
      </div>
      <div style="background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:0.6rem;text-align:center;">
        <div style="font-family:'Fraunces',serif;font-size:1.4rem;font-weight:700;color:#f585a0;">{fc}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.42rem;letter-spacing:1px;text-transform:uppercase;color:#a0aab4;">Female</div>
      </div>
      <div style="background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:0.6rem;text-align:center;">
        <div style="font-family:'Fraunces',serif;font-size:1.4rem;font-weight:700;color:#f5c842;">{avg:.1f}%</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.42rem;letter-spacing:1px;text-transform:uppercase;color:#a0aab4;">Avg Conf</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.5rem;font-family:'JetBrains Mono',monospace;font-size:0.45rem;letter-spacing:1.8px;text-transform:uppercase;color:#b0bac4;">
      VoiceIQ · TensorFlow · Librosa
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT — TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["🎙 Analyzer", "📊 Dashboard", "📜 History", "ℹ️ About"])

# ── ANALYZER ──────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("""
    <div style="margin-bottom:1.4rem;padding-bottom:1rem;border-bottom:1px solid #21262d;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.52rem;letter-spacing:3px;text-transform:uppercase;color:#d4860a;margin-bottom:0.4rem;">🎙 Upload &amp; Analyze</div>
      <div style="font-family:'Fraunces',serif;font-size:clamp(1.5rem,3.5vw,2.2rem);font-weight:700;font-style:italic;color:#f0f6fc;line-height:1.05;margin:0 0 0.35rem;">Voice Analyzer</div>
      <div style="font-size:0.88rem;color:#a0aab4;line-height:1.65;">Upload any audio clip for instant gender prediction with confidence score and spectrogram.</div>
    </div>
    """, unsafe_allow_html=True)

    # Supported formats card
    st.markdown("""
    <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:1rem 1.2rem;margin-bottom:1rem;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.52rem;letter-spacing:2.5px;text-transform:uppercase;color:#d4860a;margin-bottom:0.75rem;display:flex;align-items:center;gap:0.5rem;">
        Supported Formats
        <span style="flex:1;height:1px;background:#21262d;display:block;"></span>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:0.45rem;">
        <span style="display:inline-flex;align-items:center;gap:0.3rem;padding:0.26rem 0.6rem;border-radius:100px;background:rgba(212,134,10,.12);border:1px solid rgba(212,134,10,.25);font-family:'JetBrains Mono',monospace;font-size:0.54rem;letter-spacing:1px;text-transform:uppercase;color:#f0a030;"><span style="width:4px;height:4px;border-radius:50%;background:#d4860a;display:inline-block;"></span>WAV</span>
        <span style="display:inline-flex;align-items:center;gap:0.3rem;padding:0.26rem 0.6rem;border-radius:100px;background:rgba(212,134,10,.12);border:1px solid rgba(212,134,10,.25);font-family:'JetBrains Mono',monospace;font-size:0.54rem;letter-spacing:1px;text-transform:uppercase;color:#f0a030;"><span style="width:4px;height:4px;border-radius:50%;background:#d4860a;display:inline-block;"></span>MP3</span>
        <span style="display:inline-flex;align-items:center;gap:0.3rem;padding:0.26rem 0.6rem;border-radius:100px;background:rgba(212,134,10,.12);border:1px solid rgba(212,134,10,.25);font-family:'JetBrains Mono',monospace;font-size:0.54rem;letter-spacing:1px;text-transform:uppercase;color:#f0a030;"><span style="width:4px;height:4px;border-radius:50%;background:#d4860a;display:inline-block;"></span>OGG</span>
        <span style="display:inline-flex;align-items:center;gap:0.3rem;padding:0.26rem 0.6rem;border-radius:100px;background:rgba(212,134,10,.12);border:1px solid rgba(212,134,10,.25);font-family:'JetBrains Mono',monospace;font-size:0.54rem;letter-spacing:1px;text-transform:uppercase;color:#f0a030;"><span style="width:4px;height:4px;border-radius:50%;background:#d4860a;display:inline-block;"></span>M4A</span>
        <span style="display:inline-flex;align-items:center;gap:0.3rem;padding:0.26rem 0.6rem;border-radius:100px;background:rgba(212,134,10,.12);border:1px solid rgba(212,134,10,.25);font-family:'JetBrains Mono',monospace;font-size:0.54rem;letter-spacing:1px;text-transform:uppercase;color:#f0a030;"><span style="width:4px;height:4px;border-radius:50%;background:#d4860a;display:inline-block;"></span>FLAC</span>
        <span style="display:inline-flex;align-items:center;gap:0.3rem;padding:0.26rem 0.6rem;border-radius:100px;background:rgba(212,134,10,.12);border:1px solid rgba(212,134,10,.25);font-family:'JetBrains Mono',monospace;font-size:0.54rem;letter-spacing:1px;text-transform:uppercase;color:#f0a030;"><span style="width:4px;height:4px;border-radius:50%;background:#d4860a;display:inline-block;"></span>3 sec window</span>
        <span style="display:inline-flex;align-items:center;gap:0.3rem;padding:0.26rem 0.6rem;border-radius:100px;background:rgba(212,134,10,.12);border:1px solid rgba(212,134,10,.25);font-family:'JetBrains Mono',monospace;font-size:0.54rem;letter-spacing:1px;text-transform:uppercase;color:#f0a030;"><span style="width:4px;height:4px;border-radius:50%;background:#d4860a;display:inline-block;"></span>16 kHz SR</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    audio_file = st.file_uploader("Voice Recording", type=["wav", "mp3", "ogg", "m4a", "flac"])

    analyze_clicked = st.button("⟶  Analyze Voice", type="primary")

    if analyze_clicked and audio_file is not None:
        if not MODEL_LOADED:
            st.markdown("""
            <div style="background:#0d1117;border:1px solid rgba(240,112,144,.35);border-radius:8px;padding:1.5rem;text-align:center;margin:1rem 0;">
              <div style="font-family:'JetBrains Mono',monospace;font-size:0.5rem;letter-spacing:3px;text-transform:uppercase;color:#6e7681;margin-bottom:0.5rem;">ERROR</div>
              <div style="font-family:'Fraunces',serif;font-size:2.5rem;color:#6e7681;">✗</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:0.67rem;color:#a0aab4;margin-top:0.5rem;">gender_model.h5 not found</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Analyzing voice..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
                        tmp.write(audio_file.read())
                        tmp_path = tmp.name

                    y, sr = librosa.load(tmp_path, sr=16000)
                    os.unlink(tmp_path)
                    y = clean_audio(y, sr)
                    feat = extract_features(y, sr)[np.newaxis, ..., np.newaxis]
                    pred = float(model.predict(feat, verbose=0)[0][0])

                    if pred > 0.5:
                        gender, sym, conf = "Male", "♂", pred
                        box_bg = "linear-gradient(135deg, rgba(56,184,242,.1) 0%, #0d1117 100%)"
                        box_border = "rgba(56,184,242,.35)"
                        rg_col = "#58ccf5"
                        bar_grad = "linear-gradient(90deg, #1a7fbf, #58ccf5)"
                    else:
                        gender, sym, conf = "Female", "♀", 1 - pred
                        box_bg = "linear-gradient(135deg, rgba(240,112,144,.1) 0%, #0d1117 100%)"
                        box_border = "rgba(240,112,144,.35)"
                        rg_col = "#f585a0"
                        bar_grad = "linear-gradient(90deg, #c0395a, #f585a0)"

                    ts = datetime.now().strftime("%H:%M:%S")
                    st.session_state.history.append({
                        'id': len(st.session_state.history) + 1,
                        'timestamp': ts,
                        'gender': gender,
                        'confidence': conf
                    })

                    col_res, col_gauge = st.columns([3, 2])
                    with col_res:
                        st.markdown(f"""
                        <div style="background:{box_bg};border:1px solid {box_border};border-radius:8px;padding:1.5rem 1.1rem;text-align:center;margin:0.4rem 0;">
                          <div style="font-family:'JetBrains Mono',monospace;font-size:0.5rem;letter-spacing:3px;text-transform:uppercase;color:#6e7681;margin-bottom:0.5rem;">PREDICTION RESULT</div>
                          <div style="font-family:'Fraunces',serif;font-size:clamp(2rem,5vw,3rem);font-weight:700;font-style:italic;line-height:1;letter-spacing:-1px;margin:0.2rem 0;color:{rg_col};text-shadow:0 0 20px {rg_col}40;">{sym} {gender}</div>
                          <div style="font-family:'JetBrains Mono',monospace;font-size:0.67rem;color:#a0aab4;margin-top:0.5rem;letter-spacing:1.5px;">Confidence · {conf*100:.1f}%</div>
                          <div style="height:3px;background:rgba(255,255,255,.08);border-radius:100px;margin:0.8rem 1.2rem 0;overflow:hidden;">
                            <div style="height:100%;border-radius:100px;width:{conf*100:.1f}%;background:{bar_grad};"></div>
                          </div>
                          <div style="font-family:'JetBrains Mono',monospace;font-size:0.52rem;color:#6e7681;margin-top:0.8rem;padding-top:0.6rem;border-top:1px solid #21262d;">⏱ {ts}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_gauge:
                        st.plotly_chart(make_gauge(conf, gender), key="gauge_chart", config={"displayModeBar": False}, use_container_width=True)

                    st.markdown("""
                    <div style="display:flex;align-items:center;gap:0.7rem;margin:1.1rem 0 0.75rem;">
                      <div style="flex:1;height:1px;background:#21262d;"></div>
                      <div style="font-family:'JetBrains Mono',monospace;font-size:0.48rem;letter-spacing:2.5px;text-transform:uppercase;color:#6e7681;white-space:nowrap;">Mel Spectrogram</div>
                      <div style="flex:1;height:1px;background:#21262d;"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(make_mel_fig(y, sr, gender), key="mel_chart", config={"displayModeBar": False}, use_container_width=True)

                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    elif not analyze_clicked:
        st.markdown("""
        <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:2rem 1.1rem;text-align:center;margin:1rem 0;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.5rem;letter-spacing:3px;text-transform:uppercase;color:#6e7681;margin-bottom:0.5rem;">AWAITING INPUT</div>
          <div style="font-family:'Fraunces',serif;font-size:2.5rem;color:#6e7681;margin:0.5rem 0;">🎙</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.67rem;color:#a0aab4;margin-top:0.5rem;">Upload audio and tap Analyze</div>
        </div>
        """, unsafe_allow_html=True)
    elif analyze_clicked and audio_file is None:
        st.warning("Please upload an audio file first.")

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div style="margin-bottom:1.4rem;padding-bottom:1rem;border-bottom:1px solid #21262d;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.52rem;letter-spacing:3px;text-transform:uppercase;color:#d4860a;margin-bottom:0.4rem;">📊 Session Analytics</div>
      <div style="font-family:'Fraunces',serif;font-size:clamp(1.5rem,3.5vw,2.2rem);font-weight:700;font-style:italic;color:#f0f6fc;line-height:1.05;margin:0 0 0.35rem;">Dashboard</div>
      <div style="font-size:0.88rem;color:#a0aab4;line-height:1.65;">Real-time analytics across all predictions made this session.</div>
    </div>
    """, unsafe_allow_html=True)

    hist = st.session_state.history
    n = len(hist)
    if n:
        df_d = pd.DataFrame(hist)
        mc = int((df_d['gender'] == 'Male').sum())
        fc = int((df_d['gender'] == 'Female').sum())
        avg = float(df_d['confidence'].mean() * 100)
    else:
        mc = fc = 0; avg = 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total", n)
    with k2:
        st.metric("Male", mc)
    with k3:
        st.metric("Female", fc)
    with k4:
        st.metric("Avg Conf", f"{avg:.1f}%")

    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.52rem;letter-spacing:2.5px;text-transform:uppercase;color:#d4860a;margin-bottom:0.5rem;display:flex;align-items:center;gap:0.5rem;">
          Confidence Timeline<span style="flex:1;height:1px;background:#21262d;display:block;"></span>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(chart_timeline(hist), key="timeline_chart", config={"displayModeBar": False}, use_container_width=True)
    with c2:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.52rem;letter-spacing:2.5px;text-transform:uppercase;color:#d4860a;margin-bottom:0.5rem;display:flex;align-items:center;gap:0.5rem;">
          Gender Split<span style="flex:1;height:1px;background:#21262d;display:block;"></span>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(chart_gender(hist), key="gender_chart", config={"displayModeBar": False}, use_container_width=True)

    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.52rem;letter-spacing:2.5px;text-transform:uppercase;color:#d4860a;margin-bottom:0.5rem;display:flex;align-items:center;gap:0.5rem;">
      Confidence Distribution<span style="flex:1;height:1px;background:#21262d;display:block;"></span>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(chart_dist(hist), key="dist_chart", config={"displayModeBar": False}, use_container_width=True)

# ── HISTORY ───────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div style="margin-bottom:1.4rem;padding-bottom:1rem;border-bottom:1px solid #21262d;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.52rem;letter-spacing:3px;text-transform:uppercase;color:#d4860a;margin-bottom:0.4rem;">📜 Prediction Log</div>
      <div style="font-family:'Fraunces',serif;font-size:clamp(1.5rem,3.5vw,2.2rem);font-weight:700;font-style:italic;color:#f0f6fc;line-height:1.05;margin:0 0 0.35rem;">History</div>
      <div style="font-size:0.88rem;color:#a0aab4;line-height:1.65;">All predictions from this session with timestamps and confidence scores.</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑  Clear History", type="secondary"):
        st.session_state.history = []
        st.rerun()

    if st.session_state.history:
        df_h = pd.DataFrame(st.session_state.history)
        df_h['Conf%'] = (df_h['confidence'] * 100).round(1)
        df_show = df_h[['id', 'timestamp', 'gender', 'Conf%']].rename(
            columns={'id': 'ID', 'timestamp': 'Time', 'gender': 'Gender'}
        ).sort_values('ID', ascending=False)
        st.dataframe(df_show, use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:2rem;text-align:center;margin-top:1rem;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.67rem;color:#6e7681;">No predictions yet. Go to the Analyzer tab to get started.</div>
        </div>
        """, unsafe_allow_html=True)

# ── ABOUT ──────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div style="margin-bottom:1.4rem;padding-bottom:1rem;border-bottom:1px solid #21262d;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.52rem;letter-spacing:3px;text-transform:uppercase;color:#d4860a;margin-bottom:0.4rem;">ℹ️ Information</div>
      <div style="font-family:'Fraunces',serif;font-size:clamp(1.5rem,3.5vw,2.2rem);font-weight:700;font-style:italic;color:#f0f6fc;line-height:1.05;margin:0 0 0.35rem;">About VoiceIQ</div>
      <div style="font-size:0.88rem;color:#a0aab4;line-height:1.65;">Technical details and usage instructions.</div>
    </div>

    <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:1.3rem;margin-bottom:0.9rem;">
      <h3 style="font-family:'Fraunces',serif;font-size:1.05rem;font-weight:700;font-style:italic;color:#f5c842;margin:0 0 0.6rem;">🧠 What is VoiceIQ?</h3>
      <p style="font-size:0.84rem;color:#c9d1d9;line-height:1.7;margin:0;">VoiceIQ is a deep learning application that identifies the gender of a speaker from audio input using a CNN trained on mel spectrogram features.</p>
    </div>

    <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:1.3rem;margin-bottom:0.9rem;">
      <h3 style="font-family:'Fraunces',serif;font-size:1.05rem;font-weight:700;font-style:italic;color:#f5c842;margin:0 0 0.6rem;">🔬 Technical Details</h3>
      <ul style="list-style:none;padding:0;margin:0;">
        <li style="padding:0.35rem 0 0.35rem 1.1rem;position:relative;font-size:0.82rem;color:#c9d1d9;border-bottom:1px solid #21262d;line-height:1.6;">▸ <strong style="color:#f0f6fc;">Model:</strong> Convolutional Neural Network (CNN)</li>
        <li style="padding:0.35rem 0 0.35rem 1.1rem;position:relative;font-size:0.82rem;color:#c9d1d9;border-bottom:1px solid #21262d;line-height:1.6;">▸ <strong style="color:#f0f6fc;">Input:</strong> Mel Spectrogram — 128 frequency bands</li>
        <li style="padding:0.35rem 0 0.35rem 1.1rem;position:relative;font-size:0.82rem;color:#c9d1d9;border-bottom:1px solid #21262d;line-height:1.6;">▸ <strong style="color:#f0f6fc;">Sample Rate:</strong> 16,000 Hz</li>
        <li style="padding:0.35rem 0 0.35rem 1.1rem;position:relative;font-size:0.82rem;color:#c9d1d9;border-bottom:1px solid #21262d;line-height:1.6;">▸ <strong style="color:#f0f6fc;">Window:</strong> 3 seconds (auto-trimmed &amp; normalized)</li>
        <li style="padding:0.35rem 0 0.35rem 1.1rem;position:relative;font-size:0.82rem;color:#c9d1d9;border-bottom:1px solid #21262d;line-height:1.6;">▸ <strong style="color:#f0f6fc;">Framework:</strong> TensorFlow / Keras</li>
        <li style="padding:0.35rem 0 0.35rem 1.1rem;font-size:0.82rem;color:#c9d1d9;line-height:1.6;">▸ <strong style="color:#f0f6fc;">Output:</strong> Sigmoid → Male (&gt;0.5) / Female (≤0.5)</li>
      </ul>
    </div>

    <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:1.3rem;margin-bottom:0.9rem;">
      <h3 style="font-family:'Fraunces',serif;font-size:1.05rem;font-weight:700;font-style:italic;color:#f5c842;margin:0 0 0.6rem;">📝 How to Use</h3>
      <ul style="list-style:none;padding:0;margin:0;">
        <li style="padding:0.35rem 0 0.35rem 1.1rem;font-size:0.82rem;color:#c9d1d9;border-bottom:1px solid #21262d;line-height:1.6;">▸ Go to the <strong style="color:#f0f6fc;">Analyzer</strong> tab</li>
        <li style="padding:0.35rem 0 0.35rem 1.1rem;font-size:0.82rem;color:#c9d1d9;border-bottom:1px solid #21262d;line-height:1.6;">▸ Upload a WAV, MP3, OGG, M4A, or FLAC file</li>
        <li style="padding:0.35rem 0 0.35rem 1.1rem;font-size:0.82rem;color:#c9d1d9;border-bottom:1px solid #21262d;line-height:1.6;">▸ Click <strong style="color:#f0f6fc;">Analyze Voice</strong> to run inference</li>
        <li style="padding:0.35rem 0 0.35rem 1.1rem;font-size:0.82rem;color:#c9d1d9;border-bottom:1px solid #21262d;line-height:1.6;">▸ View prediction, confidence gauge, and spectrogram</li>
        <li style="padding:0.35rem 0 0.35rem 1.1rem;font-size:0.82rem;color:#c9d1d9;line-height:1.6;">▸ Check <strong style="color:#f0f6fc;">Dashboard</strong> for session analytics</li>
      </ul>
    </div>

    <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:1.3rem;">
      <h3 style="font-family:'Fraunces',serif;font-size:1.05rem;font-weight:700;font-style:italic;color:#f5c842;margin:0 0 0.6rem;">⚠️ Requirements</h3>
      <p style="font-size:0.84rem;color:#c9d1d9;line-height:1.7;margin:0;">Place <code style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;background:#161b22;padding:0.1rem 0.35rem;border-radius:3px;color:#f5c842;">gender_model.h5</code> in the same directory as <code style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;background:#161b22;padding:0.1rem 0.35rem;border-radius:3px;color:#f5c842;">app.py</code>. Accuracy depends on audio quality — use clear, noise-free recordings.</p>
    </div>
    """, unsafe_allow_html=True)