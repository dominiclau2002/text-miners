#!/usr/bin/env python
# coding: utf-8

"""
Task 3: Risk Clustering — BERTopic Semantic Discovery

Discovers semantic sub-themes within HIGH-risk complaints using BERTopic.
Lightweight transformer-based clustering (no jupyter kernel overhead).

Run from project root:
    python task3_risk_rating/task3_risk_clustering_bertopic.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

try:
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("ERROR: BERTopic not installed.")
    print("Install with: pip install bertopic sentence-transformers")
    exit(1)

import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80)
print("TASK 3: RISK CLUSTERING — BERTopic SEMANTIC DISCOVERY")
print("=" * 80)

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

medium_texts = X_test_text[medium_idx]
medium_products = test_products[medium_idx]

print(f"  ✓ HIGH-risk:   {len(high_texts):3d} complaints")
print(f"  ✓ MEDIUM-risk: {len(medium_texts):3d} complaints")
print(f"  ✓ LOW-risk:    {len(X_test_text[low_idx]):3d} complaints")

# ============================================================================
# SECTION 5: BERTopic on HIGH-Risk Complaints
# ============================================================================

print("\n[5/6] Running BERTopic clustering on HIGH-risk complaints...")
print(f"  Clustering {len(high_texts)} narratives with semantic embeddings...\n")

vectorizer_model = CountVectorizer(
    max_df=0.95,
    min_df=1,
    ngram_range=(1, 2),
    stop_words='english',
    max_features=1000
)

topic_model_high = BERTopic(
    embedding_model="all-MiniLM-L6-v2",  # Fast lightweight embeddings
    vectorizer_model=vectorizer_model,
    language="english",
    calculate_probabilities=True,
    min_topic_size=2,          # Low threshold for small dataset
    top_n_words=10,
    verbose=False
)

# Fit and transform
topics_high, probs_high = topic_model_high.fit_transform(high_texts)

print(f"\n  ✓ Discovered {len(set(topics_high))} semantic clusters in HIGH-risk")
print("\n  Cluster Summary:")
print(topic_model_high.get_topic_info())

# ============================================================================
# SECTION 6: BERTopic on MEDIUM-Risk Complaints (Optional)
# ============================================================================

print("\n[5b/6] Running BERTopic on MEDIUM-risk complaints...")

if len(medium_texts) >= 2:
    topic_model_medium = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        vectorizer_model=CountVectorizer(
            max_df=0.95, min_df=1, ngram_range=(1, 2), stop_words='english', max_features=1000
        ),
        calculate_probabilities=True,
        min_topic_size=2,
        top_n_words=10,
        verbose=False
    )

    topics_medium, probs_medium = topic_model_medium.fit_transform(medium_texts)
    print(f"  ✓ Discovered {len(set(topics_medium))} semantic clusters in MEDIUM-risk")
else:
    print(f"  ⚠ Skipping MEDIUM clustering (only {len(medium_texts)} samples)")
    topic_model_medium = None
    topics_medium = None

# ============================================================================
# SECTION 7: Export Results & Visualisations
# ============================================================================

print("\n[6/6] Exporting results and visualisations...\n")

os.makedirs('outputs/task3_bertopic', exist_ok=True)

# (a) Export HIGH-risk clustered CSV
high_risk_df = pd.DataFrame({
    'narrative': high_texts,
    'product': high_products,
    'bert_topic': topics_high,
    'topic_confidence': probs_high.max(axis=1),
})

high_risk_df.to_csv('outputs/task3_bertopic/high_risk_bertopic_clustered.csv', index=False)
print(f"  ✓ Saved: outputs/task3_bertopic/high_risk_bertopic_clustered.csv")

# (b) Export cluster info
cluster_info = topic_model_high.get_topic_info()
cluster_info.to_csv('outputs/task3_bertopic/high_risk_cluster_info.csv', index=False)
print(f"  ✓ Saved: outputs/task3_bertopic/high_risk_cluster_info.csv")

# (c) Product × Cluster heatmap
pivot = high_risk_df.groupby(['product', 'bert_topic']).size().unstack(fill_value=0)
plt.figure(figsize=(14, 6))
sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'})
plt.title('HIGH-Risk Complaints — Product × BERTopic Cluster', fontsize=14, fontweight='bold')
plt.xlabel('BERTopic Cluster', fontsize=11)
plt.ylabel('Product Category', fontsize=11)
plt.tight_layout()
plt.savefig('outputs/task3_bertopic/high_risk_product_cluster_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: outputs/task3_bertopic/high_risk_product_cluster_heatmap.png")

# (d) Cluster distribution
cluster_counts = high_risk_df['bert_topic'].value_counts().sort_values(ascending=False)
plt.figure(figsize=(12, 5))
cluster_counts.plot(kind='barh', color='steelblue')
plt.xlabel('Number of Complaints', fontsize=11)
plt.ylabel('BERTopic Cluster', fontsize=11)
plt.title('HIGH-Risk Complaint Distribution by BERTopic Cluster', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/task3_bertopic/high_risk_cluster_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: outputs/task3_bertopic/high_risk_cluster_distribution.png")

# (e) Save BERTopic model
topic_model_high.save('outputs/task3_bertopic/high_risk_model')
print(f"  ✓ Saved: outputs/task3_bertopic/high_risk_model/")

# (f) Export MEDIUM-risk if available
if topic_model_medium:
    medium_risk_df = pd.DataFrame({
        'narrative': medium_texts,
        'product': medium_products,
        'bert_topic': topics_medium,
        'topic_confidence': probs_medium.max(axis=1),
    })
    medium_risk_df.to_csv('outputs/task3_bertopic/medium_risk_bertopic_clustered.csv', index=False)
    print(f"  ✓ Saved: outputs/task3_bertopic/medium_risk_bertopic_clustered.csv")

    topic_model_medium.save('outputs/task3_bertopic/medium_risk_model')
    print(f"  ✓ Saved: outputs/task3_bertopic/medium_risk_model/")

# ============================================================================
# SECTION 8: Sample Complaints per Cluster
# ============================================================================

print("\n" + "=" * 80)
print("SAMPLE COMPLAINTS BY BERTOPIC CLUSTER")
print("=" * 80)

for cluster_id in sorted(high_risk_df['bert_topic'].unique()):
    cluster_rows = high_risk_df[high_risk_df['bert_topic'] == cluster_id]
    cluster_keywords = topic_model_high.get_topic(cluster_id)

    print(f"\nCLUSTER {cluster_id}")
    print(f"  ({len(cluster_rows)} complaints, {100*len(cluster_rows)/len(high_risk_df):.1f}%)")
    print(f"  Top keywords: {', '.join([w[0] for w in cluster_keywords[:5]])}")
    print("-" * 80)

    # Show 1 random example
    sample = cluster_rows.sample(1).iloc[0]
    narrative = sample['narrative']
    product = sample['product']
    confidence = sample['topic_confidence']

    if len(narrative) > 250:
        narrative = narrative[:250] + "..."

    print(f"  Product: {product}")
    print(f"  Confidence: {confidence:.1%}")
    print(f"  Example: {narrative}\n")

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("BERTOPIC CLUSTERING COMPLETE")
print("=" * 80)
print(f"\nOutputs saved to outputs/task3_bertopic/:")
print(f"  • high_risk_bertopic_clustered.csv      — All HIGH-risk with cluster labels")
print(f"  • high_risk_cluster_info.csv            — Cluster metadata & keywords")
print(f"  • high_risk_product_cluster_heatmap.png — Product × Cluster breakdown")
print(f"  • high_risk_cluster_distribution.png    — Cluster frequency bar chart")
print(f"  • high_risk_model/                      — Saved BERTopic model (reusable)")
if topic_model_medium:
    print(f"  • medium_risk_bertopic_clustered.csv    — MEDIUM-risk with cluster labels")
    print(f"  • medium_risk_model/                    — Saved MEDIUM-risk model")
print(f"\nKey Statistics:")
print(f"  HIGH-risk clusters:   {len(set(topics_high))}")
print(f"  HIGH-risk complaints: {len(high_risk_df)}")
if topic_model_medium:
    print(f"  MEDIUM-risk clusters: {len(set(topics_medium))}")
    print(f"  MEDIUM-risk complaints: {len(medium_risk_df)}")
print(f"\nUse BERTopic clusters for:")
print(f"  → Root-cause analysis: \"What semantic themes define HIGH-risk?\"")
print(f"  → Regulatory reporting: \"What complaint types are emerging?\"")
print(f"  → Analyst triage: \"Show me complaints similar to this one\"")
print("=" * 80)
