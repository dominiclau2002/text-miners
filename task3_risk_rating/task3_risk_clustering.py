#!/usr/bin/env python
# coding: utf-8

"""
Task 3: Risk Clustering — Lightweight Semantic Grouping

Discovers semantic patterns within HIGH-risk complaints by:
1. Loading risk predictions from task3_lr_predictions.pkl
2. Grouping HIGH-risk narratives by keyword themes
3. Exporting clustered results and visualisations

Lightweight alternative to BERTopic — no heavy dependencies, stable execution.

Run from project root:
    python task3_risk_rating/task3_risk_clustering.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import re
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 70)
print("TASK 3: RISK CLUSTERING — LIGHTWEIGHT KEYWORD-BASED GROUPING")
print("=" * 70)

# ============================================================================
# SECTION 1: Load Pre-Labelled Annotation Sample
# ============================================================================

print("\n[1/6] Loading pre-labelled annotation sample...")
annotation_path = 'data/annotation_sample_labelled.csv'
if not os.path.exists(annotation_path):
    raise FileNotFoundError(f'{annotation_path} not found. Run risk_rating2.ipynb first.')

df = pd.read_csv(annotation_path)
df = df.dropna(subset=['narrative', 'risk_label'])
df = df[df['narrative'].str.strip().str.len() > 0].reset_index(drop=True)

print(f"  ✓ Loaded {len(df)} complaints")
print(f"    - HIGH: {(df['risk_label']=='high').sum()}")
print(f"    - MEDIUM: {(df['risk_label']=='medium').sum()}")
print(f"    - LOW: {(df['risk_label']=='low').sum()}")

# ============================================================================
# SECTION 2: Recreate Train/Test Split
# ============================================================================

print("\n[2/6] Recreating 80/20 train/test split...")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

risk_le = LabelEncoder()
y_all = risk_le.fit_transform(df['risk_label'])

train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.2,
    stratify=y_all,
    random_state=RANDOM_STATE
)

X_test_text = df['narrative'].values[test_idx]
y_test_risk = y_all[test_idx]
test_products = df['product'].values[test_idx]

print(f"  ✓ Train: {len(train_idx)} | Test: {len(test_idx)} complaints")

# ============================================================================
# SECTION 3: Load Risk Predictions
# ============================================================================

print("\n[3/6] Loading risk predictions from task3_lr_predictions.pkl...")
lr_pkl_path = 'outputs/task3_lr_predictions.pkl'
if not os.path.exists(lr_pkl_path):
    raise FileNotFoundError(f'{lr_pkl_path} not found. Run risk_rating2.ipynb first.')

with open(lr_pkl_path, 'rb') as f:
    lr_data = pickle.load(f)
    y_pred_lr = lr_data['y_pred']
    classes = lr_data['classes']

assert len(y_pred_lr) == len(X_test_text), "Prediction count mismatch!"
print(f"  ✓ Loaded {len(y_pred_lr)} predictions")

# ============================================================================
# SECTION 4: Segment by Risk Level
# ============================================================================

print("\n[4/6] Segmenting narratives by risk level...")

high_encoded = risk_le.transform(['high'])[0]
medium_encoded = risk_le.transform(['medium'])[0]
low_encoded = risk_le.transform(['low'])[0]

high_idx = np.where(y_pred_lr == high_encoded)[0]
medium_idx = np.where(y_pred_lr == medium_encoded)[0]
low_idx = np.where(y_pred_lr == low_encoded)[0]

high_texts = X_test_text[high_idx]
high_products = test_products[high_idx]

print(f"  ✓ HIGH-risk:   {len(high_texts):3d} complaints")
print(f"  ✓ MEDIUM-risk: {len(X_test_text[medium_idx]):3d} complaints")
print(f"  ✓ LOW-risk:    {len(X_test_text[low_idx]):3d} complaints")

# ============================================================================
# SECTION 5: Keyword-Based Clustering of HIGH-Risk Complaints
# ============================================================================

print("\n[5/6] Clustering HIGH-risk complaints by keyword themes...")

# Define semantic clusters as keyword groups
CLUSTER_DEFS = {
    'Garnishment & Wage Seizure': {
        'keywords': [r'\bgarnish', r'\bwage.*levy', r'\bbank.*levy', r'\bgarni'],
        'description': 'Complaints mentioning wage garnishment, bank levies, or debt seizure'
    },
    'Identity Theft & Fraud': {
        'keywords': [r'\bidentity theft', r'\bfraud', r'\bunauthori[sz]ed.*account', r'\bopened.*account.*name'],
        'description': 'Accounts opened fraudulently or identity theft'
    },
    'Financial Hardship': {
        'keywords': [r'\bcannot (pay|afford)', r'\bunable.*(pay|afford)', r'\brent', r'\beviction', r'\bhomeless'],
        'description': 'Consumer cannot afford basic necessities (rent, food, etc.)'
    },
    'FDCPA Violations': {
        'keywords': [r'\bfdcpa', r'\bfair debt.*collection', r'\bviolat.*collection', r'\billegal.*collect'],
        'description': 'Alleged violations of Fair Debt Collection Practices Act'
    },
    'Legal Action': {
        'keywords': [r'\bsued', r'\blawsuit', r'\bcourt', r'\battorney.*filed', r'\bsum[m]ons'],
        'description': 'Active lawsuits or legal proceedings'
    },
    'Protected Classes': {
        'keywords': [r'\belderly', r'\bsenior', r'\bfixed income', r'\bmilitary', r'\bveteran', r'\bscra'],
        'description': 'Complaints involving elderly, fixed income, or military consumers'
    },
    'Credit Damage': {
        'keywords': [r'\bcredit score', r'\bcredit rating', r'\bdamaged.*credit', r'\baffect.*credit'],
        'description': 'Negative impact on credit score or report'
    }
}

# Assign each HIGH-risk complaint to best-matching cluster
cluster_assignments = []

for idx, narrative in enumerate(high_texts):
    narrative_lower = narrative.lower()
    matched_clusters = []

    for cluster_name, cluster_info in CLUSTER_DEFS.items():
        for pattern in cluster_info['keywords']:
            if re.search(pattern, narrative_lower):
                matched_clusters.append(cluster_name)
                break

    # If multiple clusters match, take the first; if none, mark as 'Other'
    assigned_cluster = matched_clusters[0] if matched_clusters else 'Other'
    cluster_assignments.append(assigned_cluster)

# Create results DataFrame
high_risk_df = pd.DataFrame({
    'narrative': high_texts,
    'product': high_products,
    'cluster': cluster_assignments
})

print(f"\n  Discovered {high_risk_df['cluster'].nunique()} semantic clusters:")
cluster_counts = high_risk_df['cluster'].value_counts()
for cluster, count in cluster_counts.items():
    pct = 100 * count / len(high_risk_df)
    print(f"    • {cluster:35s}: {count:3d} complaints ({pct:5.1f}%)")

# ============================================================================
# SECTION 6: Export Results & Visualisations
# ============================================================================

print("\n[6/6] Exporting results and visualisations...")

os.makedirs('outputs/task3_clustering', exist_ok=True)

# (a) Export full clustered CSV
high_risk_df.to_csv('outputs/task3_clustering/high_risk_clustered.csv', index=False)
print(f"  ✓ Saved: outputs/task3_clustering/high_risk_clustered.csv")

# (b) Cluster summary CSV
cluster_summary = high_risk_df.groupby('cluster').agg({
    'narrative': 'count',
    'product': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A'
}).rename(columns={'narrative': 'count', 'product': 'dominant_product'})
cluster_summary.to_csv('outputs/task3_clustering/cluster_summary.csv')
print(f"  ✓ Saved: outputs/task3_clustering/cluster_summary.csv")

# (c) Product × Cluster heatmap
pivot = high_risk_df.groupby(['product', 'cluster']).size().unstack(fill_value=0)
plt.figure(figsize=(14, 6))
sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'})
plt.title('HIGH-Risk Complaints — Product × Semantic Cluster', fontsize=14, fontweight='bold')
plt.xlabel('Cluster', fontsize=11)
plt.ylabel('Product Category', fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('outputs/task3_clustering/product_cluster_heatmap.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: outputs/task3_clustering/product_cluster_heatmap.png")
plt.close()

# (d) Cluster distribution bar chart
plt.figure(figsize=(12, 5))
cluster_counts.sort_values(ascending=False).plot(kind='barh', color='steelblue')
plt.xlabel('Number of Complaints', fontsize=11)
plt.ylabel('Cluster', fontsize=11)
plt.title('HIGH-Risk Complaint Distribution by Semantic Cluster', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/task3_clustering/cluster_distribution.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: outputs/task3_clustering/cluster_distribution.png")
plt.close()

# (e) Cluster definitions as JSON for reference
cluster_defs_export = {
    name: {
        'count': int(cluster_counts.get(name, 0)),
        'description': info['description'],
        'keywords': info['keywords']
    }
    for name, info in CLUSTER_DEFS.items()
}
cluster_defs_export['Other'] = {
    'count': int(cluster_counts.get('Other', 0)),
    'description': 'Complaints not matching other cluster keywords',
    'keywords': []
}

with open('outputs/task3_clustering/cluster_definitions.json', 'w') as f:
    json.dump(cluster_defs_export, f, indent=2)
print(f"  ✓ Saved: outputs/task3_clustering/cluster_definitions.json")

# ============================================================================
# SECTION 7: Sample Complaints per Cluster
# ============================================================================

print("\n" + "=" * 70)
print("SAMPLE COMPLAINTS BY CLUSTER")
print("=" * 70)

for cluster_name in cluster_counts.index:
    cluster_rows = high_risk_df[high_risk_df['cluster'] == cluster_name]
    print(f"\n{cluster_name.upper()}")
    print(f"  ({len(cluster_rows)} complaints, {100*len(cluster_rows)/len(high_risk_df):.1f}%)")
    print("-" * 70)

    # Show 1 random example
    sample = cluster_rows.sample(1).iloc[0]
    narrative = sample['narrative']
    product = sample['product']

    # Truncate long narrative
    if len(narrative) > 250:
        narrative = narrative[:250] + "..."

    print(f"  Product: {product}")
    print(f"  Example: {narrative}\n")

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 70)
print("CLUSTERING COMPLETE")
print("=" * 70)
print(f"\nOutputs saved to outputs/task3_clustering/:")
print(f"  • high_risk_clustered.csv         — All 34 HIGH-risk complaints with cluster labels")
print(f"  • cluster_summary.csv              — Summary statistics per cluster")
print(f"  • cluster_definitions.json         — Cluster keyword definitions")
print(f"  • product_cluster_heatmap.png      — Product × Cluster breakdown")
print(f"  • cluster_distribution.png         — Complaint counts per cluster")
print(f"\nKey Insight: HIGH-risk complaints are distributed across {high_risk_df['cluster'].nunique()} semantic themes:")
print(f"  → Use these clusters for root-cause analysis and regulatory reporting")
print(f"  → CSV can be imported into Streamlit or analyst dashboards")
print("=" * 70)
