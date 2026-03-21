"""
CFPB Complaint Analysis Dashboard
demo/streamlit.py

Run from project root:
    streamlit run demo/streamlit.py
"""

import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="CFPB Complaint Analyser",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Background ── */
.stApp {
    background-color: #111827;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #1F2937 !important;
    border-right: 1px solid #374151 !important;
}

[data-testid="stSidebar"] * {
    color: #F9FAFB !important;
}

/* Radio buttons in sidebar */
[data-testid="stSidebar"] .stRadio > div {
    gap: 4px;
}
[data-testid="stSidebar"] .stRadio label {
    background: transparent;
    border-radius: 6px;
    padding: 8px 12px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: #D1D5DB !important;
    transition: background 0.15s, color 0.15s;
    cursor: pointer;
    display: block;
    width: 100%;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #374151;
    color: #FFFFFF !important;
}
/* selected radio */
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + label,
[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    background: #3B82F6 !important;
    color: #FFFFFF !important;
}

/* ── Page title ── */
h1 {
    font-size: 1.875rem !important;
    font-weight: 700 !important;
    color: #F9FAFB !important;
    letter-spacing: -0.025em !important;
    margin-bottom: 4px !important;
    line-height: 1.25 !important;
}

/* ── Section headers ── */
h2, h3 {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: #9CA3AF !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin-top: 2rem !important;
    margin-bottom: 0.75rem !important;
    border-bottom: 1px solid #374151 !important;
    padding-bottom: 8px !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #1F2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 10px !important;
    padding: 20px 24px !important;
}
[data-testid="metric-container"]:hover {
    border-color: #3B82F6 !important;
}
[data-testid="stMetricLabel"] > div {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #9CA3AF !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 2rem !important;
    font-weight: 500 !important;
    color: #60A5FA !important;
}

/* ── Alerts / warnings ── */
[data-testid="stAlert"] {
    background: #1F2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
    color: #D1D5DB !important;
    font-size: 0.875rem !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
}
.dvn-scroller { background: #1F2937 !important; }

/* ── Text input ── */
[data-baseweb="input"] input {
    background-color: #1F2937 !important;
    border-color: #374151 !important;
    color: #F9FAFB !important;
    border-radius: 8px !important;
}

/* ── Selectbox ── */
[data-baseweb="select"] > div {
    background-color: #1F2937 !important;
    border-color: #374151 !important;
    color: #F9FAFB !important;
}

/* ── Caption / small text ── */
[data-testid="stCaptionContainer"] p,
.stCaption {
    color: #9CA3AF !important;
    font-size: 0.8rem !important;
}

/* ── Markdown text ── */
.stMarkdown p {
    color: #D1D5DB !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
}

/* ── Divider ── */
hr {
    border-color: #374151 !important;
    margin: 1.5rem 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #111827; }
::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4B5563; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#1F2937',
    font=dict(family='Inter', color='#D1D5DB', size=12),
    xaxis=dict(gridcolor='#374151', linecolor='#374151', tickcolor='#374151', tickfont=dict(color='#9CA3AF')),
    yaxis=dict(gridcolor='#374151', linecolor='#374151', tickcolor='#374151', tickfont=dict(color='#9CA3AF')),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(font=dict(color='#D1D5DB'), bgcolor='rgba(0,0,0,0)'),
    title_font=dict(color='#F9FAFB', size=14),
)

OUTPUTS = os.path.join(os.path.dirname(__file__), '..', 'outputs')

def load_json(fn):
    p = os.path.join(OUTPUTS, fn)
    return json.load(open(p)) if os.path.exists(p) else None

def load_csv(fn):
    p = os.path.join(OUTPUTS, fn)
    return pd.read_csv(p) if os.path.exists(p) else None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.75rem 0 1.25rem 0; border-bottom:1px solid #374151; margin-bottom:1.25rem;">
        <div style="font-size:1.1rem; font-weight:700; color:#F9FAFB; letter-spacing:-0.01em;">📋 CFPB Analyser</div>
        <div style="font-size:0.7rem; color:#6B7280; text-transform:uppercase; letter-spacing:0.07em; margin-top:4px;">IS450 · Text Mining · Group 4</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Overview", "🏷️  Classification", "⚠️  Risk Queue", "🔍  Topic Explorer"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top:1.5rem; margin-bottom:0.5rem;'><span style='font-size:0.68rem; font-weight:600; color:#6B7280; text-transform:uppercase; letter-spacing:0.1em;'>Pipeline Status</span></div>", unsafe_allow_html=True)

    tasks = {
        "Data Split":      os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'data', 'train_data.csv')),
        "Topic Modelling": os.path.exists(os.path.join(OUTPUTS, 'topic_vectors.csv')),
        "Classification":  os.path.exists(os.path.join(OUTPUTS, 'classification_results.csv')),
        "Risk Rating":     os.path.exists(os.path.join(OUTPUTS, 'risk_results.csv')),
    }
    for task, done in tasks.items():
        bg    = "#052e16" if done else "#1F2937"
        color = "#4ADE80" if done else "#9CA3AF"
        border= "#166534" if done else "#374151"
        icon  = "✓" if done else "○"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;padding:6px 12px;'
            f'border-radius:6px;border:1px solid {border};background:{bg};'
            f'margin-bottom:4px;font-size:0.8rem;font-weight:500;color:{color};">'
            f'{icon}&nbsp;{task}</div>',
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if "Overview" in page:
    st.title("Consumer Complaint Intelligence")
    st.markdown('<p style="color:#9CA3AF;font-size:0.95rem;margin-bottom:2rem;">Automated triage pipeline for CFPB complaint classification, risk prioritisation and topic discovery.</p>', unsafe_allow_html=True)

    class_results = load_csv('classification_results.csv')
    risk_results  = load_csv('risk_results.csv')
    class_metrics = load_json('classification_metrics.json')

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Complaints Processed", f"{len(class_results):,}" if class_results is not None else "—")
    with c2:
        if class_metrics:
            st.metric("Best Macro F1", f"{max(v['macro_f1'] for v in class_metrics.values()):.3f}")
        else:
            st.metric("Best Macro F1", "—")
    with c3:
        if risk_results is not None and 'predicted_risk' in risk_results.columns:
            st.metric("High-Risk Flagged", f"{(risk_results['predicted_risk']=='high').sum():,}")
        else:
            st.metric("High-Risk Flagged", "—")
    with c4:
        if risk_results is not None and 'predicted_risk' in risk_results.columns:
            pct = (risk_results['predicted_risk'] == 'high').mean() * 100
            st.metric("High-Risk Rate", f"{pct:.1f}%")
        else:
            st.metric("High-Risk Rate", "—")

    if class_results is not None and risk_results is not None:
        st.markdown("---")
        st.subheader("Distribution Overview")
        ca, cb = st.columns(2)
        with ca:
            counts = class_results['product'].value_counts().reset_index()
            counts.columns = ['Product', 'Count']
            fig = px.bar(counts, x='Count', y='Product', orientation='h',
                         color='Count', color_continuous_scale='Blues')
            fig.update_layout(**PLOT, title='Complaints by Product', showlegend=False, coloraxis_showscale=False)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)
        with cb:
            if 'predicted_risk' in risk_results.columns:
                rc = risk_results['predicted_risk'].value_counts().reset_index()
                rc.columns = ['Risk', 'Count']
                cmap = {'high': '#EF4444', 'medium': '#F59E0B', 'low': '#10B981'}
                fig2 = px.pie(rc, names='Risk', values='Count', color='Risk',
                              color_discrete_map=cmap, hole=0.55)
                fig2.update_layout(**PLOT, title='Risk Distribution')
                fig2.update_traces(textfont_color='#F9FAFB', textfont_size=12)
                st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
elif "Classification" in page:
    st.title("Product Classification")
    st.markdown('<p style="color:#9CA3AF;font-size:0.95rem;margin-bottom:2rem;">Multi-class classification of complaints into five financial product categories.</p>', unsafe_allow_html=True)

    class_results = load_csv('classification_results.csv')
    class_metrics = load_json('classification_metrics.json')

    if class_results is None:
        st.warning("Run `task2_classification/classification.ipynb` first to generate outputs.")
        st.stop()

    if class_metrics:
        st.subheader("Model Comparison — Macro F1")
        mdf = pd.DataFrame([{'Model': k, 'Macro F1': v['macro_f1']} for k, v in class_metrics.items()]).sort_values('Macro F1')
        fig = px.bar(mdf, x='Macro F1', y='Model', orientation='h',
                     color='Macro F1', color_continuous_scale='Blues', range_x=[0, 1])
        fig.add_vline(x=0.85, line_dash='dash', line_color='#EF4444',
                      annotation_text='Target 0.85', annotation_font_color='#EF4444',
                      annotation_font_size=11)
        fig.update_layout(**PLOT, coloraxis_showscale=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Predicted vs Actual Distribution")
    c1, c2 = st.columns(2)
    with c1:
        a = class_results['product'].value_counts().reset_index()
        a.columns = ['Product', 'Count']
        fig_a = px.pie(a, names='Product', values='Count',
                       color_discrete_sequence=px.colors.sequential.Blues_r, hole=0.4)
        fig_a.update_layout(**PLOT, title='Actual')
        fig_a.update_traces(textfont_color='#F9FAFB')
        st.plotly_chart(fig_a, use_container_width=True)
    with c2:
        p = class_results['predicted_product'].value_counts().reset_index()
        p.columns = ['Product', 'Count']
        fig_p = px.pie(p, names='Product', values='Count',
                       color_discrete_sequence=px.colors.sequential.Teal_r, hole=0.4)
        fig_p.update_layout(**PLOT, title='Predicted')
        fig_p.update_traces(textfont_color='#F9FAFB')
        st.plotly_chart(fig_p, use_container_width=True)

    st.subheader("Accuracy by Category")
    acc = class_results.groupby('product')['correct'].mean().reset_index()
    acc.columns = ['Product', 'Accuracy']
    fig_acc = px.bar(acc.sort_values('Accuracy'), x='Accuracy', y='Product', orientation='h',
                     color='Accuracy', color_continuous_scale='Greens', range_x=[0, 1])
    fig_acc.update_layout(**PLOT, coloraxis_showscale=False)
    fig_acc.update_traces(marker_line_width=0)
    st.plotly_chart(fig_acc, use_container_width=True)

    st.subheader("Search Complaints")
    query = st.text_input("", placeholder="Type keywords to filter complaint narratives…")
    if query:
        filtered = class_results[class_results['narrative'].str.contains(query, case=False, na=False)].head(20)
        st.caption(f"{len(filtered)} results for '{query}'")
        st.dataframe(filtered[['narrative', 'product', 'predicted_product', 'correct']],
                     use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: RISK QUEUE
# ─────────────────────────────────────────────────────────────────────────────
elif "Risk" in page:
    st.title("Risk Priority Queue")
    st.markdown('<p style="color:#9CA3AF;font-size:0.95rem;margin-bottom:2rem;">Complaints ranked by predicted risk level. High-risk cases are surfaced first for analyst review.</p>', unsafe_allow_html=True)

    risk_results = load_csv('risk_results.csv')
    risk_metrics  = load_json('risk_metrics.json')

    if risk_results is None:
        st.warning("Run `task3_risk_rating/risk_rating.ipynb` first to generate outputs.")
        st.stop()

    rc = risk_results['predicted_risk'].value_counts()
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("High Risk",   f"{rc.get('high',   0):,}")
    with c2: st.metric("Medium Risk", f"{rc.get('medium', 0):,}")
    with c3: st.metric("Low Risk",    f"{rc.get('low',    0):,}")

    if 'product' in risk_results.columns:
        st.subheader("Risk Level by Product")
        heat = risk_results.groupby(['product', 'predicted_risk']).size().unstack(fill_value=0)
        heat_pct = heat.div(heat.sum(axis=1), axis=0)
        fig = px.imshow(heat_pct, color_continuous_scale='RdYlGn_r', aspect='auto', text_auto='.0%')
        fig.update_layout(**PLOT, title='Risk % by Product')
        fig.update_traces(textfont_color='#F9FAFB', textfont_size=11)
        st.plotly_chart(fig, use_container_width=True)

    if risk_metrics:
        st.subheader("Model Performance")
        mdf = pd.DataFrame([
            {'Model': k, 'Macro F1': v['macro_f1'],
             'HIGH Precision': v.get('high_precision', 0),
             'HIGH Recall': v.get('high_recall', 0)}
            for k, v in risk_metrics.items()
        ])
        st.dataframe(mdf, use_container_width=True, hide_index=True)

    st.subheader("High-Risk Complaints")
    high_df = risk_results[risk_results['predicted_risk'] == 'high'].copy()
    cf, cs = st.columns([2, 1])
    with cf:
        fp = st.selectbox("Filter by product",
                          ['All'] + sorted(risk_results['product'].dropna().unique().tolist()))
    with cs:
        st.markdown(f"<div style='padding-top:30px;color:#9CA3AF;font-size:0.8rem;'>{len(high_df):,} total high-risk complaints</div>", unsafe_allow_html=True)
    if fp != 'All':
        high_df = high_df[high_df['product'] == fp]
    st.dataframe(high_df[['narrative', 'product', 'predicted_risk']].head(50),
                 use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: TOPIC EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif "Topic" in page:
    st.title("Topic Discovery Explorer")
    st.markdown('<p style="color:#9CA3AF;font-size:0.95rem;margin-bottom:2rem;">Latent themes discovered across complaint narratives via LDA topic modelling.</p>', unsafe_allow_html=True)

    complaints_with_topics = load_csv('complaints_with_topics.csv')
    topic_labels = load_json('topic_labels.json')

    if complaints_with_topics is None:
        st.warning("Run `task1_topic_modelling/topic_modelling.ipynb` first to generate outputs.")
        st.stop()

    keywords_img = os.path.join(OUTPUTS, 'topic_keywords.png')
    heatmap_img  = os.path.join(OUTPUTS, 'topic_category_heatmap.png')

    if os.path.exists(keywords_img):
        st.subheader("Top Keywords per Topic")
        st.image(keywords_img, use_container_width=True)

    if os.path.exists(heatmap_img):
        st.subheader("Topic to Product Mapping")
        st.image(heatmap_img, use_container_width=True)

    if 'dominant_topic' in complaints_with_topics.columns:
        st.subheader("Complaint Volume by Topic")
        tc = complaints_with_topics['dominant_topic'].value_counts().reset_index()
        tc.columns = ['Topic', 'Count']
        if topic_labels:
            tc['Label'] = tc['Topic'].astype(str).map(
                {str(k): v for k, v in topic_labels.items()}
            ).fillna('Topic ' + tc['Topic'].astype(str))
        else:
            tc['Label'] = 'Topic ' + tc['Topic'].astype(str)

        fig = px.bar(tc.sort_values('Count', ascending=False),
                     x='Label', y='Count', color='Count', color_continuous_scale='Blues')
        fig.update_layout(**PLOT, coloraxis_showscale=False)
        fig.update_traces(marker_line_width=0)
        fig.update_xaxes(tickangle=35, tickfont=dict(size=11))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Browse Complaints by Topic")
        topics = sorted(complaints_with_topics['dominant_topic'].unique())
        label_map = {i: topic_labels.get(str(i), f'Topic {i}') for i in topics} if topic_labels else {i: f'Topic {i}' for i in topics}
        selected = st.selectbox("Select a topic", topics,
                                format_func=lambda x: f"Topic {x}  —  {label_map[x]}")
        sub = complaints_with_topics[complaints_with_topics['dominant_topic'] == selected].head(20)
        st.caption(f"Showing 20 sample complaints for: {label_map[selected]}")
        st.dataframe(sub[['narrative', 'product']].head(20),
                     use_container_width=True, hide_index=True)