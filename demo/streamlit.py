"""
CFPB Complaint Analysis Dashboard — v2
demo/streamlit.py

Run from project root:
    streamlit run demo/streamlit.py
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CFPB Intelligence · Group 4",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
PRIMARY   = "#00D4FF"
SECONDARY = "#FF8C42"
SUCCESS   = "#00E5A0"
WARNING   = "#FFB800"
DANGER    = "#FF4757"
PURPLE    = "#A78BFA"
TEXT1     = "#F0F8FF"
TEXT2     = "#8BA0B8"
CARD_BG   = "rgba(10,18,35,0.85)"
BORDER    = "rgba(0,212,255,0.15)"

PRODUCT_COLORS = {
    "credit_reporting":    PRIMARY,
    "debt_collection":     DANGER,
    "mortgages_and_loans": SECONDARY,
    "credit_card":         PURPLE,
    "retail_banking":      SUCCESS,
}

RISK_COLORS = {"high": DANGER, "medium": WARNING, "low": SUCCESS}

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [class*="css"] {{
    font-family: 'Outfit', sans-serif !important;
    color: {TEXT2} !important;
}}

.stApp {{
    background: #050B1A;
    background-image:
        radial-gradient(ellipse 80% 50% at 10% 10%, rgba(0,212,255,0.07) 0%, transparent 55%),
        radial-gradient(ellipse 60% 40% at 88% 88%, rgba(255,140,66,0.05) 0%, transparent 55%),
        radial-gradient(ellipse 50% 70% at 50% 50%, rgba(0,60,160,0.03) 0%, transparent 65%);
    background-attachment: fixed;
}}

.main .block-container {{
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1440px !important;
}}

/* ── Sidebar shell ── */
[data-testid="stSidebar"] {{
    background: rgba(3,8,18,0.98) !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] * {{ color: {TEXT2} !important; }}
/* Hide stray "nav" label text Streamlit may render */
[data-testid="stSidebar"] .stRadio > label,
[data-testid="stSidebar"] .stRadio > div > label:not([data-baseweb="radio"]),
[data-testid="stSidebar"] .stRadio + div > p,
.st-key-sidebar_nav > label {{
    display: none !important;
}}

/* ── Nav: hide radio circles ── */
.st-key-sidebar_nav label[data-baseweb="radio"] > div:first-child {{
    display: none !important;
}}

/* ── Nav: base item ── */
.st-key-sidebar_nav div[role="radiogroup"] {{
    gap: 2px !important;
    display: flex !important;
    flex-direction: column !important;
}}
.st-key-sidebar_nav label[data-baseweb="radio"] {{
    display: flex !important;
    align-items: center !important;
    padding: 10px 14px 10px 16px !important;
    border-radius: 9px !important;
    border: 1px solid transparent !important;
    border-left: 2px solid transparent !important;
    cursor: pointer !important;
    transition: background 0.18s, border-color 0.18s, color 0.18s !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: rgba(139,160,184,0.7) !important;
    background: transparent !important;
    width: 100% !important;
    margin: 0 !important;
    letter-spacing: 0.01em !important;
}}
.st-key-sidebar_nav label[data-baseweb="radio"]:hover {{
    background: rgba(0,212,255,0.07) !important;
    border-color: rgba(0,212,255,0.18) !important;
    border-left-color: rgba(0,212,255,0.4) !important;
    color: {TEXT1} !important;
}}
/* ── Nav: active item via :has(input:checked) ── */
.st-key-sidebar_nav label[data-baseweb="radio"]:has(input[type="radio"]:checked) {{
    background: rgba(0,212,255,0.09) !important;
    border-color: rgba(0,212,255,0.15) !important;
    border-left: 2px solid {PRIMARY} !important;
    color: {TEXT1} !important;
    font-weight: 600 !important;
}}

/* ── Headers ── */
h1 {{
    font-family: 'Syne', sans-serif !important;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    color: {TEXT1} !important;
    letter-spacing: -0.035em !important;
    line-height: 1.05 !important;
    margin-bottom: 0.3rem !important;
}}
h2, h3 {{
    font-family: 'Syne', sans-serif !important;
    font-size: 0.6rem !important;
    font-weight: 700 !important;
    color: rgba(0,212,255,0.5) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.18em !important;
    margin-top: 2.5rem !important;
    margin-bottom: 1rem !important;
    border-bottom: 1px solid rgba(0,212,255,0.1) !important;
    padding-bottom: 0.5rem !important;
}}

/* ── Metric cards ── */
[data-testid="metric-container"] {{
    background: {CARD_BG} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 14px !important;
    padding: 1.4rem 1.6rem !important;
    backdrop-filter: blur(12px);
    animation: fadeUp 0.5s ease both;
}}
[data-testid="metric-container"]:hover {{
    border-color: rgba(0,212,255,0.45) !important;
    box-shadow: 0 0 28px rgba(0,212,255,0.12) !important;
}}
[data-testid="stMetricLabel"] > div {{
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
    color: rgba(0,212,255,0.65) !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2rem !important;
    font-weight: 500 !important;
    color: {TEXT1} !important;
    letter-spacing: -0.02em !important;
}}

/* ── Inputs ── */
[data-baseweb="input"] input {{
    background: rgba(10,18,35,0.8) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    color: {TEXT1} !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
}}
[data-baseweb="input"] input:focus {{
    border-color: {PRIMARY} !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}}
[data-baseweb="select"] > div {{
    background: rgba(10,18,35,0.8) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    color: {TEXT1} !important;
    border-radius: 8px !important;
}}

/* ── Radio nav ── */
[data-testid="stSidebar"] .stRadio > div {{ gap: 2px; }}
[data-testid="stSidebar"] .stRadio label {{
    background: transparent !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: rgba(139,160,184,0.75) !important;
    transition: all 0.2s !important;
    display: block !important;
    width: 100% !important;
    border: 1px solid transparent !important;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(0,212,255,0.07) !important;
    color: {TEXT1} !important;
    border-color: rgba(0,212,255,0.15) !important;
}}

/* ── Alerts ── */
[data-testid="stAlert"] {{
    background: rgba(10,18,35,0.8) !important;
    border: 1px solid rgba(255,140,66,0.3) !important;
    border-radius: 10px !important;
    color: {TEXT2} !important;
}}

/* ── DataFrame ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    overflow: hidden;
}}
.dvn-scroller {{ background: rgba(10,18,35,0.9) !important; }}

/* ── Markdown / Caption ── */
.stMarkdown p {{
    color: rgba(139,160,184,0.85) !important;
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
}}
[data-testid="stCaptionContainer"] p, .stCaption {{
    color: rgba(0,212,255,0.5) !important;
    font-size: 0.72rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.03em !important;
}}

/* ── Divider / Scrollbar ── */
hr {{ border-color: rgba(0,212,255,0.08) !important; margin: 2rem 0 !important; }}
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: #050B1A; }}
::-webkit-scrollbar-thumb {{ background: rgba(0,212,255,0.2); border-radius: 2px; }}
::-webkit-scrollbar-thumb:hover {{ background: rgba(0,212,255,0.4); }}

/* ── Animations ── */
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(16px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
</style>
""", unsafe_allow_html=True)


# ── Plotly layout factory ──────────────────────────────────────────────────────
def PL(title="", height=None, **kw):
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,18,35,0.4)",
        font=dict(family="Outfit", color=TEXT2, size=12),
        xaxis=dict(
            gridcolor="rgba(0,212,255,0.06)",
            linecolor="rgba(0,212,255,0.12)",
            tickcolor="rgba(0,212,255,0.12)",
            tickfont=dict(color="rgba(139,160,184,0.7)", size=11),
        ),
        yaxis=dict(
            gridcolor="rgba(0,212,255,0.06)",
            linecolor="rgba(0,212,255,0.12)",
            tickcolor="rgba(0,212,255,0.12)",
            tickfont=dict(color="rgba(139,160,184,0.7)", size=11),
        ),
        margin=dict(l=16, r=16, t=48 if title else 20, b=16),
        title=dict(text=title, font=dict(family="Syne", color=TEXT1, size=13)),
        legend=dict(
            font=dict(color=TEXT2, size=11),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,212,255,0.12)",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="rgba(10,18,35,0.95)",
            bordercolor="rgba(0,212,255,0.3)",
            font=dict(family="Outfit", color=TEXT1, size=12),
        ),
        coloraxis_colorbar=dict(
            tickfont=dict(color=TEXT2),
            title_font=dict(color=TEXT1),
        ),
    )
    if height:
        layout["height"] = height
    layout.update(kw)
    return layout


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(ROOT, "outputs")
DATA    = os.path.join(ROOT, "data")


@st.cache_data(show_spinner=False)
def csv(fname, folder=None):
    folder = folder or OUTPUTS
    p = os.path.join(folder, fname)
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data(show_spinner=False)
def jfile(fname):
    p = os.path.join(OUTPUTS, fname)
    return json.load(open(p)) if os.path.exists(p) else None


@st.cache_data(show_spinner=False)
def train_data():
    p = os.path.join(DATA, "train_data.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data(show_spinner=False)
def topics_sample(n=8000):
    p = os.path.join(OUTPUTS, "complaints_with_topics.csv")
    if not os.path.exists(p):
        return None
    chunks = []
    for chunk in pd.read_csv(p, chunksize=20000):
        chunks.append(chunk.sample(min(n // 8, len(chunk)), random_state=42))
        if sum(len(c) for c in chunks) >= n:
            break
    return pd.concat(chunks).head(n).reset_index(drop=True)


# ── Custom HTML helpers ────────────────────────────────────────────────────────
def stat_card(label, value, sub="", accent=PRIMARY, delay=0.0):
    sub_html = f'<div style="font-size:0.75rem;color:{TEXT2};margin-top:0.35rem;">{sub}</div>' if sub else ""
    return f"""
    <div style="
        background:{CARD_BG}; border:1px solid {accent}28; border-radius:14px;
        padding:1.4rem 1.6rem; animation:fadeUp 0.5s ease {delay}s both;
        transition:border-color 0.25s,box-shadow 0.25s; position:relative; overflow:hidden;
    ">
        <div style="position:absolute;top:0;left:0;right:0;height:2px;
            background:linear-gradient(90deg,transparent,{accent}55,transparent);"></div>
        <div style="font-size:0.62rem;font-weight:700;text-transform:uppercase;
            letter-spacing:0.14em;color:{accent}AA;font-family:'Outfit',sans-serif;
            margin-bottom:0.55rem;">{label}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:500;
            color:{TEXT1};letter-spacing:-0.02em;line-height:1;">{value}</div>
        {sub_html}
    </div>"""


def section_label(text):
    st.markdown(
        f'<div style="font-size:0.6rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.18em;color:rgba(0,212,255,0.45);font-family:\'JetBrains Mono\',monospace;'
        f'border-bottom:1px solid rgba(0,212,255,0.1);padding-bottom:0.45rem;'
        f'margin-top:2.25rem;margin-bottom:0.9rem;">{text}</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:0.9rem 0 1.4rem;border-bottom:1px solid {BORDER};margin-bottom:1.2rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:800;
            color:{TEXT1};letter-spacing:-0.02em;">⚡ CFPB Intelligence</div>
        <div style="font-size:0.62rem;color:rgba(0,212,255,0.45);text-transform:uppercase;
            letter-spacing:0.12em;margin-top:5px;font-family:'JetBrains Mono',monospace;">
            IS450 · Group 4 · Text Mining
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["🏠  Overview", "🏷️  Classification", "⚠️  Risk Queue", "🔍  Topic Explorer"],
        label_visibility="collapsed",
        key="sidebar_nav",
    )

    st.markdown(f"""
    <div style="margin-top:1.6rem;margin-bottom:0.55rem;">
        <span style="font-size:0.58rem;font-weight:700;color:rgba(0,212,255,0.35);
            text-transform:uppercase;letter-spacing:0.18em;font-family:'JetBrains Mono',monospace;">
            Pipeline Status
        </span>
    </div>""", unsafe_allow_html=True)

    pipeline = {
        "Data Split":      os.path.exists(os.path.join(DATA, "train_data.csv")),
        "Topic Modelling": os.path.exists(os.path.join(OUTPUTS, "topic_vectors.csv")),
        "Classification":  os.path.exists(os.path.join(OUTPUTS, "classification_results.csv")),
        "Risk Rating":     os.path.exists(os.path.join(OUTPUTS, "risk_results.csv")),
    }
    for task, done in pipeline.items():
        a = SUCCESS if done else TEXT2
        bg = "rgba(0,229,160,0.08)" if done else "rgba(10,18,35,0.5)"
        br = "rgba(0,229,160,0.22)" if done else "rgba(0,212,255,0.08)"
        ic = "✓" if done else "○"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;padding:7px 12px;'
            f'border-radius:7px;border:1px solid {br};background:{bg};margin-bottom:4px;'
            f'font-size:0.79rem;font-weight:500;color:{a};font-family:\'DM Sans\',sans-serif;">'
            f'<span style="font-size:0.72rem;">{ic}</span> {task}</div>',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
#  PAGE 1 — OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
if "Overview" in page:

    st.title("Consumer Complaint Intelligence")
    st.markdown(
        f'<p style="font-size:1rem;color:{TEXT2};margin:0.2rem 0 2.5rem;">CFPB complaint triage · classification · risk prioritisation · topic discovery</p>',
        unsafe_allow_html=True,
    )

    td        = train_data()
    class_res = csv("classification_results.csv")
    risk_res  = csv("risk_results.csv")
    td_topics = topics_sample(5000)
    tlabels   = jfile("topic_labels.json")

    # ── Headline metrics ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(stat_card("Complaints Processed", "143,962", "full dataset", PRIMARY,   0.00), unsafe_allow_html=True)
    with c2:
        st.markdown(stat_card("Product Categories",   "5",       "normalised classes",  SECONDARY, 0.08), unsafe_allow_html=True)
    with c3:
        st.markdown(stat_card("Topics Discovered",    "10",      "LDA coherence k=10",  PURPLE,    0.16), unsafe_allow_html=True)
    with c4:
        f1 = "—" if class_res is None else "0.8545"
        st.markdown(stat_card("Best Macro F1",        f1,        "logistic regression", SUCCESS,   0.24), unsafe_allow_html=True)

    st.markdown("---")

    # ── Product distribution ──
    if td is not None:
        section_label("Distribution by Product Category")
        col_bar, col_pie = st.columns([1.3, 1])

        prod = td["product"].value_counts().reset_index()
        prod.columns = ["product", "count"]
        prod["label"]  = prod["product"].str.replace("_", " ").str.title()
        prod["color"]  = prod["product"].map(PRODUCT_COLORS).fillna(PRIMARY)
        prod["share"]  = prod["count"] / prod["count"].sum()

        with col_bar:
            fig = go.Figure()
            for _, r in prod.iterrows():
                fig.add_trace(go.Bar(
                    x=[r["count"]], y=[r["label"]], orientation="h",
                    marker_color=r["color"], marker_line_width=0, opacity=0.82,
                    hovertemplate=f"<b>{r['label']}</b><br>{r['count']:,} complaints ({r['share']:.1%})<extra></extra>",
                    showlegend=False, name=r["label"],
                ))
            fig.update_layout(**PL("Train split — 115,169 complaints", height=270), barmode="overlay")
            fig.update_yaxes(tickfont=dict(size=12, color="#C8D8E8"))
            st.plotly_chart(fig, use_container_width=True)

        with col_pie:
            fig2 = go.Figure(go.Pie(
                labels=prod["label"], values=prod["count"],
                hole=0.62,
                marker=dict(colors=list(prod["color"]), line=dict(color="#050B1A", width=2)),
                textinfo="percent",
                textfont=dict(size=11, color=TEXT1),
                pull=[0.04] * len(prod),
                hovertemplate="<b>%{label}</b><br>%{value:,} complaints<br>%{percent}<extra></extra>",
                rotation=100,
            ))
            fig2.add_annotation(
                text="<b>5</b><br><span style='font-size:10px'>classes</span>",
                x=0.5, y=0.5,
                font=dict(size=16, color=TEXT1, family="Syne"),
                showarrow=False,
            )
            fig2.update_layout(**PL(height=270))
            st.plotly_chart(fig2, use_container_width=True)

    # ── Topic overview ──
    if td_topics is not None and "dominant_topic" in td_topics.columns:
        section_label("Topic Distribution Across Complaints")

        tc = td_topics["dominant_topic"].value_counts().reset_index()
        tc.columns = ["topic", "count"]
        if tlabels:
            tc["label"] = tc["topic"].astype(str).map({str(k): v for k, v in tlabels.items()}).fillna("Topic " + tc["topic"].astype(str))
        else:
            tc["label"] = "Topic " + tc["topic"].astype(str)

        n = len(tc)
        grad = px.colors.sample_colorscale(
            [[0, PRIMARY], [0.5, PURPLE], [1.0, SECONDARY]],
            [i / max(n - 1, 1) for i in range(n)],
        )
        fig3 = go.Figure(go.Bar(
            x=tc["count"], y=tc["label"], orientation="h",
            marker=dict(color=grad, line_width=0),
            opacity=0.85,
            hovertemplate="<b>%{y}</b><br>%{x:,} complaints<extra></extra>",
        ))
        fig3.update_layout(**PL("Dominant topic — sampled 5,000 complaints", height=340))
        fig3.update_yaxes(autorange="reversed", tickfont=dict(size=11, color="#C8D8E8"))
        st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
#  PAGE 2 — CLASSIFICATION
# ──────────────────────────────────────────────────────────────────────────────
elif "Classification" in page:

    st.title("Product Classification")
    st.markdown(
        f'<p style="font-size:1rem;color:{TEXT2};margin:0.2rem 0 2.5rem;">Multi-class NLP classification of complaints into five financial product categories.</p>',
        unsafe_allow_html=True,
    )

    class_res = csv("classification_results.csv")

    # ── Model comparison cards (hardcoded from notebook) ──
    section_label("Model Comparison — Macro F1")
    MODEL_RESULTS = [
        ("Naive Bayes",         0.8138, 0.83, PRIMARY),
        ("Logistic Regression", 0.8545, 0.87, SUCCESS),
        ("Neural Network MLP",  0.8533, 0.87, PURPLE),
    ]
    c1, c2, c3 = st.columns(3)
    for col, (model, f1, acc, color) in zip([c1, c2, c3], MODEL_RESULTS):
        best_tag = "  ★ Best" if f1 == max(m[1] for m in MODEL_RESULTS) else ""
        with col:
            st.markdown(
                stat_card(model + best_tag, f"{f1:.4f}", f"Macro F1  ·  Accuracy {acc:.0%}", color),
                unsafe_allow_html=True,
            )

    # ── Per-class F1 bar ──
    section_label("Per-class F1 Score — Logistic Regression (Best Model)")
    PER_CLASS = [
        ("Credit Reporting",    0.90, PRIMARY),
        ("Retail Banking",      0.89, SUCCESS),
        ("Mortgages & Loans",   0.86, SECONDARY),
        ("Credit Card",         0.81, PURPLE),
        ("Debt Collection",     0.81, DANGER),
    ]
    fig = go.Figure()
    for label, f1, color in PER_CLASS:
        fig.add_trace(go.Bar(
            x=[f1], y=[label], orientation="h",
            marker_color=color, marker_line_width=0,
            text=[f"{f1:.2f}"], textposition="inside",
            textfont=dict(color=TEXT1, size=12, family="JetBrains Mono"),
            hovertemplate=f"<b>{label}</b><br>F1: {f1:.4f}<extra></extra>",
            showlegend=False,
        ))
    fig.add_vline(x=0.85, line_dash="dash", line_color=f"{DANGER}80",
                  annotation_text="0.85 target",
                  annotation_font_color=DANGER, annotation_font_size=11,
                  annotation_position="top right")
    fig.update_layout(**PL(height=240), barmode="overlay", showlegend=False,
                      xaxis=dict(range=[0, 1], tickformat=".0%"))
    fig.update_yaxes(tickfont=dict(size=12, color="#C8D8E8"))
    st.plotly_chart(fig, use_container_width=True)

    # ── Model F1 comparison bar ──
    section_label("All Models — Side-by-side F1")
    mdf = pd.DataFrame([{"Model": m, "Macro F1": f1} for m, f1, _, _ in MODEL_RESULTS])
    colors_m = [PRIMARY, SUCCESS, PURPLE]
    fig_m = go.Figure(go.Bar(
        x=mdf["Macro F1"], y=mdf["Model"], orientation="h",
        marker_color=colors_m, marker_line_width=0, opacity=0.85,
        text=[f"{v:.4f}" for v in mdf["Macro F1"]], textposition="inside",
        textfont=dict(color=TEXT1, size=12, family="JetBrains Mono"),
        hovertemplate="<b>%{y}</b><br>Macro F1: %{x:.4f}<extra></extra>",
    ))
    fig_m.add_vline(x=0.85, line_dash="dash", line_color=f"{DANGER}80",
                    annotation_text="0.85 target",
                    annotation_font_color=DANGER, annotation_font_size=11)
    fig_m.update_layout(**PL(height=200), showlegend=False,
                         xaxis=dict(range=[0.78, 0.88], tickformat=".2f"))
    fig_m.update_yaxes(tickfont=dict(size=12, color="#C8D8E8"))
    st.plotly_chart(fig_m, use_container_width=True)

    # ── Confusion matrices image ──
    cm_path = os.path.join(OUTPUTS, "confusion_matrices.png")
    if os.path.exists(cm_path):
        section_label("Confusion Matrices")
        st.image(cm_path, use_container_width=True)

    # ── Full results (only when CSV exists) ──
    if class_res is not None:
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            section_label("Actual Distribution")
            a = class_res["product"].value_counts().reset_index()
            a.columns = ["product", "count"]
            a["label"] = a["product"].str.replace("_", " ").str.title()
            a["color"] = a["product"].map(PRODUCT_COLORS).fillna(PRIMARY)
            fig_a = go.Figure(go.Pie(
                labels=a["label"], values=a["count"], hole=0.52,
                marker=dict(colors=list(a["color"]), line=dict(color="#050B1A", width=2)),
                textinfo="percent", textfont=dict(size=11, color=TEXT1),
                pull=[0.04] * len(a),
                hovertemplate="<b>%{label}</b><br>%{value:,}<br>%{percent}<extra></extra>",
            ))
            fig_a.update_layout(**PL("Actual", height=260))
            st.plotly_chart(fig_a, use_container_width=True)

        with col_b:
            section_label("Predicted Distribution")
            p = class_res["predicted_product"].value_counts().reset_index()
            p.columns = ["product", "count"]
            p["label"] = p["product"].str.replace("_", " ").str.title()
            p["color"] = p["product"].map(PRODUCT_COLORS).fillna(PURPLE)
            fig_p = go.Figure(go.Pie(
                labels=p["label"], values=p["count"], hole=0.52,
                marker=dict(colors=list(p["color"]), line=dict(color="#050B1A", width=2)),
                textinfo="percent", textfont=dict(size=11, color=TEXT1),
                pull=[0.04] * len(p),
                hovertemplate="<b>%{label}</b><br>%{value:,}<br>%{percent}<extra></extra>",
            ))
            fig_p.update_layout(**PL("Predicted", height=260))
            st.plotly_chart(fig_p, use_container_width=True)

        if "correct" in class_res.columns:
            section_label("Accuracy by Product Category")
            acc_df = class_res.groupby("product")["correct"].mean().reset_index()
            acc_df.columns = ["product", "accuracy"]
            acc_df["label"] = acc_df["product"].str.replace("_", " ").str.title()
            acc_df["color"] = acc_df["product"].map(PRODUCT_COLORS).fillna(PRIMARY)
            acc_df = acc_df.sort_values("accuracy")
            fig_acc = go.Figure(go.Bar(
                x=acc_df["accuracy"], y=acc_df["label"], orientation="h",
                marker=dict(color=list(acc_df["color"]), line_width=0), opacity=0.85,
                text=[f"{v:.1%}" for v in acc_df["accuracy"]], textposition="inside",
                textfont=dict(color=TEXT1, family="JetBrains Mono", size=12),
                hovertemplate="<b>%{y}</b><br>Accuracy: %{x:.2%}<extra></extra>",
            ))
            fig_acc.update_layout(**PL(height=230), showlegend=False,
                                   xaxis=dict(range=[0, 1], tickformat=".0%"))
            fig_acc.update_yaxes(tickfont=dict(size=12, color="#C8D8E8"))
            st.plotly_chart(fig_acc, use_container_width=True)

        section_label("Search Complaints")
        q = st.text_input("", placeholder="⌕  Keywords to filter complaint narratives…")
        if q:
            filt = class_res[class_res["narrative"].str.contains(q, case=False, na=False)].head(20)
            st.caption(f"{len(filt):,} results for '{q}'")
            st.dataframe(filt[["narrative", "product", "predicted_product"]],
                         use_container_width=True, hide_index=True)
    else:
        st.info("Export `outputs/classification_results.csv` from `task2_classification/classification.ipynb` to unlock full prediction results here.")


# ──────────────────────────────────────────────────────────────────────────────
#  PAGE 3 — RISK QUEUE
# ──────────────────────────────────────────────────────────────────────────────
elif "Risk" in page:

    st.title("Risk Priority Queue")
    st.markdown(
        f'<p style="font-size:1rem;color:{TEXT2};margin:0.2rem 0 2.5rem;">Complaints triage by predicted risk level — high-risk cases surfaced first for analyst review.</p>',
        unsafe_allow_html=True,
    )

    risk_res   = csv("risk_results.csv")
    annotation = csv("annotation_sample_labelled.csv", folder=DATA)

    # ── Annotation label distribution (always visible) ──
    if annotation is not None and "risk_label" in annotation.columns:
        section_label("Annotation Sample — Label Distribution (n=692)")

        rc = annotation["risk_label"].str.lower().value_counts().reset_index()
        rc.columns = ["risk", "count"]
        total_ann = len(annotation)

        c1, c2, c3 = st.columns(3)
        for col, risk, color in zip([c1, c2, c3],
                                    ["high", "medium", "low"],
                                    [DANGER, WARNING, SUCCESS]):
            n = int(rc.loc[rc["risk"] == risk, "count"].sum())
            pct = n / total_ann * 100
            with col:
                st.markdown(stat_card(f"{risk.title()} Risk", f"{n:,}", f"{pct:.1f}% of sample", color), unsafe_allow_html=True)

        col_donut, col_bar = st.columns([1, 1.4])
        with col_donut:
            section_label("Risk Split — Donut")
            colors_r = [RISK_COLORS.get(r, PRIMARY) for r in rc["risk"]]
            fig_d = go.Figure(go.Pie(
                labels=rc["risk"].str.title(), values=rc["count"],
                hole=0.62,
                marker=dict(colors=colors_r, line=dict(color="#050B1A", width=3)),
                textinfo="percent+label",
                textfont=dict(size=12, color=TEXT1),
                pull=[0.07 if r == "high" else 0.02 for r in rc["risk"]],
                hovertemplate="<b>%{label}</b><br>%{value:,} complaints<br>%{percent}<extra></extra>",
            ))
            fig_d.add_annotation(
                text=f"<b>{total_ann}</b><br>samples",
                x=0.5, y=0.5,
                font=dict(size=15, color=TEXT1, family="Syne"),
                showarrow=False,
            )
            fig_d.update_layout(**PL(height=300))
            st.plotly_chart(fig_d, use_container_width=True)

        with col_bar:
            if "product" in annotation.columns:
                section_label("Risk × Product Heatmap")
                cross = annotation.groupby(["product", "risk_label"]).size().reset_index(name="n")
                tots  = cross.groupby("product")["n"].sum()
                cross["pct"] = cross.apply(lambda r: r["n"] / tots[r["product"]] * 100, axis=1)
                cross["prod_label"] = cross["product"].str.replace("_", " ").str.title()

                fig_s = px.bar(
                    cross, x="pct", y="prod_label",
                    color="risk_label",
                    color_discrete_map={k: v for k, v in RISK_COLORS.items()},
                    orientation="h", barmode="stack",
                    text=cross["pct"].round(0).astype(int).astype(str) + "%",
                    labels={"pct": "Share (%)", "prod_label": "", "risk_label": "Risk"},
                )
                fig_s.update_layout(
                    **PL("Risk label share by product category", height=300),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(range=[0, 100], ticksuffix="%"),
                )
                fig_s.update_yaxes(tickfont=dict(size=11, color="#C8D8E8"))
                fig_s.update_traces(textfont=dict(size=10, color=TEXT1), marker_line_width=0)
                st.plotly_chart(fig_s, use_container_width=True)

    # ── Risk model confusion matrix ──
    cm_risk = os.path.join(OUTPUTS, "risk_rating_confusion_matrix.png")
    if os.path.exists(cm_risk):
        section_label("Risk Rating Model — Confusion Matrix")
        col_img, _ = st.columns([1, 1])
        with col_img:
            st.image(cm_risk, use_container_width=True)

    # ── Full predicted results (when CSV exists) ──
    if risk_res is not None and "predicted_risk" in risk_res.columns:
        st.markdown("---")
        section_label("Predicted Risk — Full Test Set")

        rc2 = risk_res["predicted_risk"].str.lower().value_counts()
        d1, d2, d3 = st.columns(3)
        for col, risk, color in zip([d1, d2, d3],
                                    ["high", "medium", "low"],
                                    [DANGER, WARNING, SUCCESS]):
            n = int(rc2.get(risk, 0))
            with col:
                st.markdown(stat_card(f"{risk.title()} Risk", f"{n:,}",
                                      f"{n / len(risk_res):.1%} of test set", color),
                             unsafe_allow_html=True)

        if "product" in risk_res.columns:
            section_label("Risk % by Product — Heatmap")
            heat = risk_res.groupby(["product", "predicted_risk"]).size().unstack(fill_value=0)
            heat_pct = heat.div(heat.sum(axis=1), axis=0)
            heat_pct.index = heat_pct.index.str.replace("_", " ").str.title()
            fig_h = px.imshow(
                heat_pct,
                color_continuous_scale=[[0, SUCCESS], [0.5, WARNING], [1, DANGER]],
                aspect="auto", text_auto=".0%",
                labels=dict(color="Share"),
            )
            fig_h.update_layout(**PL(height=220))
            fig_h.update_traces(textfont_color=TEXT1, textfont_size=12)
            st.plotly_chart(fig_h, use_container_width=True)

        section_label("High-Risk Complaint Browser")
        high_df = risk_res[risk_res["predicted_risk"] == "high"].copy()
        cf, cs = st.columns([2, 1])
        with cf:
            opts = ["All"] + sorted(risk_res["product"].dropna().unique().tolist()) if "product" in risk_res.columns else ["All"]
            fp = st.selectbox("Filter by product", opts)
        with cs:
            st.markdown(f'<div style="padding-top:30px;font-size:0.8rem;color:{DANGER};font-family:\'JetBrains Mono\',monospace;">{len(high_df):,} high-risk</div>', unsafe_allow_html=True)
        if fp != "All":
            high_df = high_df[high_df["product"] == fp]
        st.dataframe(high_df[["narrative", "product", "predicted_risk"]].head(50),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Export `outputs/risk_results.csv` from `task3_risk_rating/risk_rating.ipynb` to see full predicted risk results here.")


# ──────────────────────────────────────────────────────────────────────────────
#  PAGE 4 — TOPIC EXPLORER
# ──────────────────────────────────────────────────────────────────────────────
elif "Topic" in page:

    st.title("Topic Discovery Explorer")
    st.markdown(
        f'<p style="font-size:1rem;color:{TEXT2};margin:0.2rem 0 2.5rem;">10 latent themes discovered across CFPB complaint narratives via LDA with coherence-optimised k.</p>',
        unsafe_allow_html=True,
    )

    td_topics = topics_sample(10000)
    tlabels   = jfile("topic_labels.json")

    if td_topics is None:
        st.warning("Run `task1_topic_modelling/topic_modelling.ipynb` to generate outputs.")
        st.stop()

    topics_list = sorted(td_topics["dominant_topic"].unique())
    lmap = {i: tlabels.get(str(i), f"Topic {i}") for i in topics_list} if tlabels else {i: f"Topic {i}" for i in topics_list}
    td_topics["topic_label"]   = td_topics["dominant_topic"].map(lmap)
    if "product" in td_topics.columns:
        td_topics["product_label"] = td_topics["product"].str.replace("_", " ").str.title()

    # ── Sunburst: product → topic ──
    if "product" in td_topics.columns:
        section_label("Topic Distribution by Product — Interactive Sunburst")
        sun_data = (
            td_topics.groupby(["product_label", "topic_label"])
            .size().reset_index(name="count")
        )
        product_color_map = {
            "Credit Reporting":    PRIMARY,
            "Debt Collection":     DANGER,
            "Mortgages And Loans": SECONDARY,
            "Credit Card":         PURPLE,
            "Retail Banking":      SUCCESS,
        }
        fig_sun = px.sunburst(
            sun_data,
            path=["product_label", "topic_label"],
            values="count",
            color="product_label",
            color_discrete_map=product_color_map,
        )
        fig_sun.update_traces(
            textfont=dict(size=11, color=TEXT1),
            marker=dict(line=dict(width=1.5, color="#050B1A")),
            hovertemplate="<b>%{label}</b><br>%{value:,} complaints<br>%{percentParent:.1%} of parent<extra></extra>",
        )
        fig_sun.update_layout(**PL(height=480))
        st.plotly_chart(fig_sun, use_container_width=True)

    # ── Topic volume bar ──
    section_label("Complaint Volume by Topic")
    tc = td_topics["topic_label"].value_counts().reset_index()
    tc.columns = ["label", "count"]
    n = len(tc)
    grad = px.colors.sample_colorscale(
        [[0, PRIMARY], [0.45, PURPLE], [1.0, SECONDARY]],
        [i / max(n - 1, 1) for i in range(n)],
    )
    fig_v = go.Figure(go.Bar(
        x=tc["count"], y=tc["label"], orientation="h",
        marker=dict(color=grad, line_width=0), opacity=0.85,
        text=tc["count"], textposition="auto",
        textfont=dict(family="JetBrains Mono", size=11, color=TEXT1),
        hovertemplate="<b>%{y}</b><br>%{x:,} complaints<extra></extra>",
    ))
    fig_v.update_layout(**PL("Sampled 10,000 complaints", height=360))
    fig_v.update_yaxes(autorange="reversed", tickfont=dict(size=11, color="#C8D8E8"))
    st.plotly_chart(fig_v, use_container_width=True)

    # ── Topic keywords image ──
    kw_path = os.path.join(OUTPUTS, "topic_keywords.png")
    if os.path.exists(kw_path):
        section_label("Top Keywords per Topic")
        st.image(kw_path, use_container_width=True)

    # ── Topic browser ──
    section_label("Browse Complaints by Topic")
    selected = st.selectbox("", options=list(lmap.values()), placeholder="Select a topic…")
    if selected:
        sub = td_topics[td_topics["topic_label"] == selected]
        st.caption(f"{len(sub):,} complaints in sample assigned to  ·  {selected}")

        if "product" in sub.columns:
            pib = sub["product_label"].value_counts().reset_index()
            pib.columns = ["product", "count"]
            pib["color"] = pib["product"].map(product_color_map).fillna(PRIMARY)
            fig_t = go.Figure(go.Bar(
                x=pib["count"], y=pib["product"], orientation="h",
                marker=dict(color=list(pib["color"]), line_width=0), opacity=0.85,
                hovertemplate="<b>%{y}</b><br>%{x:,}<extra></extra>",
            ))
            fig_t.update_layout(**PL(f"Product breakdown — {selected}", height=180), showlegend=False)
            fig_t.update_yaxes(tickfont=dict(size=11, color="#C8D8E8"))
            st.plotly_chart(fig_t, use_container_width=True)

        st.dataframe(sub[["narrative", "product_label"]].rename(columns={"product_label": "product"}).head(15),
                     use_container_width=True, hide_index=True)
