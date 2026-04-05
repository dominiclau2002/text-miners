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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif !important;
    color: {TEXT2} !important;
}}

.stApp {{
    background: #0d1117;
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
    font-family: 'DM Sans', sans-serif !important;
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
    font-family: 'DM Sans', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 600 !important;
    color: {TEXT1} !important;
    letter-spacing: -0.02em !important;
    line-height: 1.1 !important;
    margin-bottom: 0.3rem !important;
}}
h2, h3 {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    color: rgba(139,160,184,0.6) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-top: 2rem !important;
    margin-bottom: 0.8rem !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    padding-bottom: 0.4rem !important;
}}

/* ── Metric cards ── */
[data-testid="metric-container"] {{
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    padding: 1.3rem 1.5rem !important;
}}
[data-testid="metric-container"]:hover {{
    border-color: rgba(255,255,255,0.14) !important;
}}
[data-testid="stMetricLabel"] > div {{
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: rgba(139,160,184,0.6) !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'DM Mono', monospace !important;
    font-size: 1.9rem !important;
    font-weight: 400 !important;
    color: {TEXT1} !important;
    letter-spacing: -0.02em !important;
}}

/* ── Inputs ── */
[data-baseweb="input"] input {{
    background: rgba(10,18,35,0.8) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    color: {TEXT1} !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
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
    font-family: 'DM Mono', monospace !important;
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
        font=dict(family="DM Sans", color=TEXT2, size=12),
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
        title=dict(text=title, font=dict(family="DM Sans", color=TEXT1, size=13)),
        legend=dict(
            font=dict(color=TEXT2, size=11),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,212,255,0.12)",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="rgba(10,18,35,0.95)",
            bordercolor="rgba(0,212,255,0.3)",
            font=dict(family="DM Sans", color=TEXT1, size=12),
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
    df = pd.concat(chunks).head(n).reset_index(drop=True)
    # Recompute dominant_topic from the topic vector columns so this works even
    # when topic_vectors.csv does not include a dominant_topic column.
    tv_path = os.path.join(OUTPUTS, "topic_vectors.csv")
    if os.path.exists(tv_path):
        tv = pd.read_csv(tv_path)
        topic_cols = [c for c in tv.columns if c.startswith("topic_")]
        if topic_cols:
            df["dominant_topic"] = tv.loc[df.index, topic_cols].idxmax(axis=1).str.replace("topic_", "", regex=False).astype(int).values
    return df


# ── Custom HTML helpers ────────────────────────────────────────────────────────
def stat_card(label, value, sub="", accent=PRIMARY, delay=0.0):
    sub_html = f'<div style="font-size:0.72rem;color:{TEXT2};margin-top:0.3rem;">{sub}</div>' if sub else ""
    return f"""
    <div style="
        background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
        border-left:3px solid {accent}; border-radius:8px; padding:1.3rem 1.5rem;
    ">
        <div style="font-size:0.7rem;font-weight:500;text-transform:uppercase;
            letter-spacing:0.07em;color:rgba(139,160,184,0.55);font-family:'DM Sans', sans-serif;
            margin-bottom:0.5rem;">{label}</div>
        <div style="font-family:'DM Mono',monospace;font-size:1.9rem;font-weight:400;
            color:{TEXT1};letter-spacing:-0.02em;line-height:1;">{value}</div>
        {sub_html}
    </div>"""


def short_label(s, n=30):
    """Truncate long strings for chart axes; full text still shown on hover."""
    return s if len(s) <= n else s[:n - 1] + "…"


def section_label(text):
    st.markdown(
        f'<div style="font-size:0.7rem;font-weight:500;text-transform:uppercase;'
        f'letter-spacing:0.07em;color:rgba(139,160,184,0.55);font-family:\'DM Mono\',monospace;'
        f'border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:0.4rem;'
        f'margin-top:2rem;margin-bottom:0.85rem;">{text}</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:0.9rem 0 1.4rem;border-bottom:1px solid {BORDER};margin-bottom:1.2rem;">
        <div style="font-family:'DM Sans', sans-serif;font-size:1.05rem;font-weight:600;
            color:{TEXT1};letter-spacing:-0.01em;">⚡ CFPB Intelligence</div>
        <div style="font-size:0.62rem;color:rgba(0,212,255,0.45);text-transform:uppercase;
            letter-spacing:0.12em;margin-top:5px;font-family:'DM Mono',monospace;">
            IS450 · Group 4 · Text Mining
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["🏠  Overview", "🏷️  Classification", "⚠️  Risk Queue", "📊  Risk Clustering", "🔍  Topic Explorer"],
        label_visibility="collapsed",
        key="sidebar_nav",
    )

    st.markdown(f"""
    <div style="margin-top:1.6rem;margin-bottom:0.55rem;">
        <span style="font-size:0.58rem;font-weight:700;color:rgba(0,212,255,0.35);
            text-transform:uppercase;letter-spacing:0.18em;font-family:'DM Mono',monospace;">
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
                font=dict(size=16, color=TEXT1, family="DM Sans"),
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
            tc["full_label"] = tc["topic"].astype(str).map({str(k): v for k, v in tlabels.items()}).fillna("Topic " + tc["topic"].astype(str))
        else:
            tc["full_label"] = "Topic " + tc["topic"].astype(str)
        tc["label"] = tc["full_label"].apply(short_label)

        n = len(tc)
        grad = px.colors.sample_colorscale(
            [[0, PRIMARY], [0.5, PURPLE], [1.0, SECONDARY]],
            [i / max(n - 1, 1) for i in range(n)],
        )
        fig3 = go.Figure(go.Bar(
            x=tc["count"], y=tc["label"], orientation="h",
            marker=dict(color=grad, line_width=0),
            opacity=0.85,
            customdata=tc["full_label"],
            hovertemplate="<b>%{customdata}</b><br>%{x:,} complaints<extra></extra>",
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
            textfont=dict(color=TEXT1, size=12, family="DM Mono"),
            hovertemplate=f"<b>{label}</b><br>F1: {f1:.4f}<extra></extra>",
            showlegend=False,
        ))
    fig.add_vline(x=0.85, line_dash="dash", line_color="rgba(255,71,87,0.5)",
                  annotation_text="0.85 target",
                  annotation_font_color=DANGER, annotation_font_size=11,
                  annotation_position="top right")
    fig.update_layout(**PL(height=240), barmode="overlay", showlegend=False)
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
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
        textfont=dict(color=TEXT1, size=12, family="DM Mono"),
        hovertemplate="<b>%{y}</b><br>Macro F1: %{x:.4f}<extra></extra>",
    ))
    fig_m.add_vline(x=0.85, line_dash="dash", line_color="rgba(255,71,87,0.5)",
                    annotation_text="0.85 target",
                    annotation_font_color=DANGER, annotation_font_size=11)
    fig_m.update_layout(**PL(height=200), showlegend=False)
    fig_m.update_xaxes(range=[0.78, 0.88], tickformat=".2f")
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
                textfont=dict(color=TEXT1, family="DM Mono", size=12),
                hovertemplate="<b>%{y}</b><br>Accuracy: %{x:.2%}<extra></extra>",
            ))
            fig_acc.update_layout(**PL(height=230), showlegend=False)
            fig_acc.update_xaxes(range=[0, 1], tickformat=".0%")
            fig_acc.update_yaxes(tickfont=dict(size=12, color="#C8D8E8"))
            st.plotly_chart(fig_acc, use_container_width=True)

        section_label("Search Complaints")
        q = st.text_input("", placeholder="⌕  Keywords to filter complaint narratives…")
        if q:
            filt = class_res[class_res["narrative"].str.contains(q, case=False, na=False)].head(20)
            st.caption(f"{len(filt):,} results for '{q}'")

            # Custom HTML table with white text for high contrast
            display_df = filt[["narrative", "product", "predicted_product"]].reset_index(drop=True)

            html_table = '<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">'
            html_table += f'<thead><tr style="background:rgba(0,212,255,0.1);border-bottom:1px solid {BORDER};">'
            for col in display_df.columns:
                html_table += f'<th style="padding:10px;text-align:left;color:{TEXT1};font-weight:600;border-right:1px solid {BORDER};">{col.replace("_", " ").title()}</th>'
            html_table += '</tr></thead>'

            html_table += '<tbody>'
            for idx, row in display_df.iterrows():
                bg = "rgba(0,212,255,0.05)" if idx % 2 == 0 else "transparent"
                html_table += f'<tr style="background:{bg};border-bottom:1px solid {BORDER};">'
                for col in display_df.columns:
                    text = str(row[col])
                    if col == "narrative" and len(text) > 120:
                        text = text[:120] + "..."
                    html_table += f'<td style="padding:10px;color:{TEXT1};border-right:1px solid {BORDER};">{text}</td>'
                html_table += '</tr>'
            html_table += '</tbody></table>'

            st.markdown(html_table, unsafe_allow_html=True)
    else:
        st.info("Export `outputs/classification_results.csv` from `task2_classification/classification.ipynb` to unlock full prediction results here.")


# ──────────────────────────────────────────────────────────────────────────────
#  PAGE 3 — RISK QUEUE
# ──────────────────────────────────────────────────────────────────────────────
elif "Risk Queue" in page:

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
                font=dict(size=15, color=TEXT1, family="DM Sans"),
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
                fig_s.update_layout(**PL("Risk label share by product category", height=300))
                fig_s.update_layout(
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

        # Custom HTML table with white text for high contrast
        display_df = high_df[["narrative", "product", "predicted_risk"]].head(50).reset_index(drop=True)

        html_table = '<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">'
        html_table += f'<thead><tr style="background:rgba(255,71,87,0.1);border-bottom:1px solid {BORDER};">'
        for col in display_df.columns:
            html_table += f'<th style="padding:10px;text-align:left;color:{TEXT1};font-weight:600;border-right:1px solid {BORDER};">{col.replace("_", " ").title()}</th>'
        html_table += '</tr></thead>'

        html_table += '<tbody>'
        for idx, row in display_df.iterrows():
            bg = "rgba(255,71,87,0.05)" if idx % 2 == 0 else "transparent"
            html_table += f'<tr style="background:{bg};border-bottom:1px solid {BORDER};">'
            for col in display_df.columns:
                text = str(row[col])
                if col == "narrative" and len(text) > 120:
                    text = text[:120] + "..."
                # Color-code risk level
                if col == "predicted_risk":
                    color = DANGER if text == "high" else (WARNING if text == "medium" else SUCCESS)
                    html_table += f'<td style="padding:10px;color:{color};font-weight:600;border-right:1px solid {BORDER};">{text.upper()}</td>'
                else:
                    html_table += f'<td style="padding:10px;color:{TEXT1};border-right:1px solid {BORDER};">{text}</td>'
            html_table += '</tr>'
        html_table += '</tbody></table>'

        st.markdown(html_table, unsafe_allow_html=True)
    else:
        st.info("Export `outputs/risk_results.csv` from `task3_risk_rating/risk_rating.ipynb` to see full predicted risk results here.")


# ──────────────────────────────────────────────────────────────────────────────
#  PAGE 3b — RISK CLUSTERING
# ──────────────────────────────────────────────────────────────────────────────
elif "Clustering" in page:

    st.title("Risk Clustering — Semantic Analysis")
    st.markdown(
        f'<p style="font-size:1rem;color:{TEXT2};margin:0.2rem 0 2.5rem;">Discover complaint patterns within HIGH-risk tier · root-cause analysis · regulatory insight · emerging issue detection</p>',
        unsafe_allow_html=True,
    )

    # Load clustering data
    clustering_folder = os.path.join(OUTPUTS, "task3_clustering")
    cluster_csv = csv("high_risk_clustered.csv", folder=clustering_folder)

    # Load cluster definitions (custom JSON loader for clustering folder)
    cluster_defs_path = os.path.join(clustering_folder, "cluster_definitions.json")
    cluster_defs = json.load(open(cluster_defs_path)) if os.path.exists(cluster_defs_path) else None

    if cluster_csv is not None:
        # ── Cluster distribution ──
        section_label("Cluster Distribution — HIGH-Risk Complaints")

        cluster_counts = cluster_csv["cluster"].value_counts()
        c1, c2, c3 = st.columns([1.2, 1.3, 1])

        # Bar chart
        with c1:
            fig_dist = go.Figure(go.Bar(
                y=cluster_counts.index,
                x=cluster_counts.values,
                orientation="h",
                marker=dict(color=SECONDARY, line_width=0),
                opacity=0.85,
                text=cluster_counts.values,
                textposition="auto",
                textfont=dict(family="DM Mono", size=11, color=TEXT1),
                hovertemplate="<b>%{y}</b><br>%{x} complaints<extra></extra>",
            ))
            fig_dist.update_layout(**PL(f"n={len(cluster_csv)}", height=320))
            fig_dist.update_yaxes(autorange="reversed", tickfont=dict(size=11, color="#C8D8E8"))
            st.plotly_chart(fig_dist, use_container_width=True)

        # Summary stats
        with c2:
            st.markdown("---")
            st.markdown(f'<div style="padding:1rem;"><span style="font-size:0.85rem;color:{TEXT2};">Cluster Breakdown</span></div>', unsafe_allow_html=True)
            for cluster_name, count in cluster_counts.items():
                pct = 100 * count / len(cluster_csv)
                bar_width = pct / 100
                st.markdown(
                    f'<div style="margin-bottom:12px;">'
                    f'<div style="font-size:0.8rem;margin-bottom:4px;color:{TEXT1};">{cluster_name}</div>'
                    f'<div style="background:rgba(0,212,255,0.1);border-radius:3px;height:6px;overflow:hidden;">'
                    f'<div style="background:{SECONDARY};height:100%;width:{bar_width*100}%;"></div>'
                    f'</div>'
                    f'<div style="font-size:0.7rem;color:{TEXT2};margin-top:2px;">{count} · {pct:.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # Total stats
        with c3:
            st.markdown("---")
            st.metric("Total Complaints", f"{len(cluster_csv)}", delta="HIGH-risk subset", delta_color="off")
            st.metric("Clusters Found", cluster_counts.nunique(), delta="semantic themes", delta_color="off")
            st.metric("Dominant Cluster", cluster_counts.index[0], delta=f"{cluster_counts.iloc[0]/len(cluster_csv):.1%}", delta_color="off")

        # ── Product × Cluster heatmap ──
        st.markdown("---")
        section_label("Product × Cluster Cross-Tabulation")

        hm_path = os.path.join(clustering_folder, "product_cluster_heatmap.png")
        if os.path.exists(hm_path):
            col_img, col_info = st.columns([1.5, 1])
            with col_img:
                st.image(hm_path, use_container_width=True)
            with col_info:
                st.markdown(f'<div style="padding:1rem;"><span style="font-size:0.85rem;color:{TEXT2};">Insight</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:0.85rem;line-height:1.6;color:{TEXT1};">'
                           f'Product categories show distinct complaint patterns. '
                           f'<b style="color:{SECONDARY};">Credit Reporting</b> dominates HIGH-risk tier, '
                           f'particularly in Identity Theft & Fraud cluster.'
                           f'</div>', unsafe_allow_html=True)

        # ── Cluster browser ──
        st.markdown("---")
        section_label("Browse Complaints by Cluster")

        selected_cluster = st.selectbox(
            "Select cluster to view complaints:",
            options=cluster_counts.index.tolist(),
            placeholder="Choose a cluster…"
        )

        if selected_cluster:
            cluster_subset = cluster_csv[cluster_csv["cluster"] == selected_cluster]

            # Cluster definition
            if cluster_defs and selected_cluster in cluster_defs:
                cluster_info = cluster_defs[selected_cluster]
                st.markdown(
                    f'<div style="padding:12px;border-left:3px solid {SECONDARY};background:rgba(255,140,66,0.05);margin-bottom:1rem;">'
                    f'<b>{selected_cluster}</b><br>'
                    f'<span style="font-size:0.85rem;color:{TEXT2};">{cluster_info.get("description", "")}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Summary
            st.caption(f"**{len(cluster_subset)}** complaints in **{selected_cluster}** cluster")

            # Product breakdown
            if "product" in cluster_subset.columns:
                prod_breakdown = cluster_subset["product"].value_counts()
                fig_prod = go.Figure(go.Bar(
                    x=prod_breakdown.values,
                    y=prod_breakdown.index,
                    orientation="h",
                    marker=dict(color=PRIMARY, line_width=0),
                    opacity=0.85,
                    hovertemplate="<b>%{y}</b><br>%{x}<extra></extra>",
                ))
                fig_prod.update_layout(**PL(f"Product distribution — {selected_cluster}", height=200), showlegend=False)
                fig_prod.update_yaxes(tickfont=dict(size=10, color="#C8D8E8"))
                st.plotly_chart(fig_prod, use_container_width=True)

            # Complaints table — custom HTML for contrast
            display_df = cluster_subset[["narrative", "product", "cluster"]].head(10).reset_index(drop=True)

            # Build HTML table with white text
            html_table = '<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">'
            html_table += f'<thead><tr style="background:rgba(0,212,255,0.1);border-bottom:1px solid {BORDER};">'
            for col in display_df.columns:
                html_table += f'<th style="padding:10px;text-align:left;color:{TEXT1};font-weight:600;border-right:1px solid {BORDER};">{col}</th>'
            html_table += '</tr></thead>'

            html_table += '<tbody>'
            for idx, row in display_df.iterrows():
                bg = "rgba(0,212,255,0.05)" if idx % 2 == 0 else "transparent"
                html_table += f'<tr style="background:{bg};border-bottom:1px solid {BORDER};">'
                for col in display_df.columns:
                    text = str(row[col])
                    if len(text) > 120:
                        text = text[:120] + "..."
                    html_table += f'<td style="padding:10px;color:{TEXT1};border-right:1px solid {BORDER};">{text}</td>'
                html_table += '</tr>'
            html_table += '</tbody></table>'

            st.markdown(html_table, unsafe_allow_html=True)
    else:
        st.info("Run `python task3_risk_rating/task3_risk_clustering.py` to generate risk clustering analysis.")


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
    lmap_short = {i: short_label(v) for i, v in lmap.items()}
    td_topics["topic_label"]       = td_topics["dominant_topic"].map(lmap)
    td_topics["topic_label_short"] = td_topics["dominant_topic"].map(lmap_short)
    if "product" in td_topics.columns:
        td_topics["product_label"] = td_topics["product"].str.replace("_", " ").str.title()

    # ── Sunburst: product → topic ──
    if "product" in td_topics.columns:
        section_label("Topic Distribution by Product — Interactive Sunburst")
        sun_data = (
            td_topics.groupby(["product_label", "topic_label_short"])
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
            path=["product_label", "topic_label_short"],
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
    tc = td_topics[["topic_label", "topic_label_short"]].value_counts().reset_index()
    tc.columns = ["full_label", "label", "count"]
    n = len(tc)
    grad = px.colors.sample_colorscale(
        [[0, PRIMARY], [0.45, PURPLE], [1.0, SECONDARY]],
        [i / max(n - 1, 1) for i in range(n)],
    )
    fig_v = go.Figure(go.Bar(
        x=tc["count"], y=tc["label"], orientation="h",
        marker=dict(color=grad, line_width=0), opacity=0.85,
        text=tc["count"], textposition="auto",
        textfont=dict(family="DM Mono", size=11, color=TEXT1),
        customdata=tc["full_label"],
        hovertemplate="<b>%{customdata}</b><br>%{x:,} complaints<extra></extra>",
    ))
    fig_v.update_layout(**PL("Sampled 10,000 complaints", height=360))
    fig_v.update_yaxes(autorange="reversed", tickfont=dict(size=11, color="#C8D8E8"))
    st.plotly_chart(fig_v, use_container_width=True)

    # ── Topic × Product heatmap image ──
    hm_path = os.path.join(OUTPUTS, "topic_category_heatmap.png")
    if os.path.exists(hm_path):
        section_label("Topic × Product Category Heatmap")
        st.image(hm_path, use_container_width=True)

    # ── Topic keywords image ──
    kw_labeled_path = os.path.join(OUTPUTS, "topic_keywords_labeled.png")
    kw_default_path = os.path.join(OUTPUTS, "topic_keywords.png")
    kw_path = kw_labeled_path if os.path.exists(kw_labeled_path) else kw_default_path
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

        # Custom HTML table with white text for high contrast
        display_df = sub[["narrative", "product_label"]].rename(columns={"product_label": "product"}).head(15).reset_index(drop=True)

        html_table = '<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">'
        html_table += f'<thead><tr style="background:rgba(167,139,250,0.1);border-bottom:1px solid {BORDER};">'
        for col in display_df.columns:
            html_table += f'<th style="padding:10px;text-align:left;color:{TEXT1};font-weight:600;border-right:1px solid {BORDER};">{col.title()}</th>'
        html_table += '</tr></thead>'

        html_table += '<tbody>'
        for idx, row in display_df.iterrows():
            bg = "rgba(167,139,250,0.05)" if idx % 2 == 0 else "transparent"
            html_table += f'<tr style="background:{bg};border-bottom:1px solid {BORDER};">'
            for col in display_df.columns:
                text = str(row[col])
                if col == "narrative" and len(text) > 120:
                    text = text[:120] + "..."
                html_table += f'<td style="padding:10px;color:{TEXT1};border-right:1px solid {BORDER};">{text}</td>'
            html_table += '</tr>'
        html_table += '</tbody></table>'

        st.markdown(html_table, unsafe_allow_html=True)
