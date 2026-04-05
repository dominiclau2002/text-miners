#!/usr/bin/env python
# coding: utf-8

"""
Task 3: Advanced LLM Analysis — Qwen-Powered Complaint Intelligence

Uses Alibaba DashScope (Qwen) API to analyze ALL test complaints:
- Root cause identification
- Consumer harm assessment
- Why each complaint is flagged (brief explanation)

Run from project root:
    python task3_risk_rating/task3_qwen_analysis.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load environment variables
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("DASHSCOPE_API_KEY not found in .env file")

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

MODEL = "qwen-plus"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80)
print("TASK 3: ADVANCED LLM ANALYSIS — QWEN COMPLAINT INTELLIGENCE")
print("=" * 80)

# ============================================================================
# SECTION 1: Load Pre-Labelled Annotation Sample
# ============================================================================

print("\n[1/5] Loading pre-labelled annotation sample...")
annotation_path = 'data/annotation_sample_labelled.csv'
if not os.path.exists(annotation_path):
    raise FileNotFoundError(f'{annotation_path} not found. Run risk_rating2.ipynb first.')

df = pd.read_csv(annotation_path)
df = df.dropna(subset=['narrative', 'risk_label'])
df = df[df['narrative'].str.strip().str.len() > 0].reset_index(drop=True)

print(f"  ✓ Loaded {len(df)} complaints")

# ============================================================================
# SECTION 2: Recreate Train/Test Split
# ============================================================================

print("\n[2/5] Recreating 80/20 train/test split...")

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

print("\n[3/5] Loading risk predictions from task3_lr_predictions.pkl...")
lr_pkl_path = 'outputs/task3_lr_predictions.pkl'
if not os.path.exists(lr_pkl_path):
    raise FileNotFoundError(f'{lr_pkl_path} not found. Run risk_rating2.ipynb first.')

with open(lr_pkl_path, 'rb') as f:
    lr_data = pickle.load(f)
    y_pred_lr = lr_data['y_pred']

assert len(y_pred_lr) == len(X_test_text), "Prediction count mismatch!"
print(f"  ✓ Loaded {len(y_pred_lr)} predictions")

# ============================================================================
# SECTION 4: Convert Predictions to Risk Labels
# ============================================================================

print("\n[4/5] Preparing complaints for analysis...")

pred_risk_labels = risk_le.inverse_transform(y_pred_lr)

complaints_data = pd.DataFrame({
    'narrative': X_test_text,
    'product': test_products,
    'predicted_risk': pred_risk_labels,
})

print(f"  ✓ Analyzing {len(complaints_data)} total complaints")
print(f"    - HIGH: {(complaints_data['predicted_risk']=='high').sum()}")
print(f"    - MEDIUM: {(complaints_data['predicted_risk']=='medium').sum()}")
print(f"    - LOW: {(complaints_data['predicted_risk']=='low').sum()}")

# ============================================================================
# SECTION 5: Qwen LLM Analysis
# ============================================================================

print("\n[5/5] Running Qwen API analysis on all complaints...")
print(f"  Processing {len(complaints_data)} narratives...\n")


def call_llm_with_retry(prompt, max_retries=5):
    """Call Qwen API with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial compliance analyst. Return only valid JSON, no markdown, no backticks.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            if '429' in err_str or 'rate' in err_str.lower():
                wait = 10 * (2 ** attempt)
                print(f"    Rate limited. Waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded for LLM API call")


analyses = []
for idx, (narrative, product, risk_level) in enumerate(zip(
    complaints_data['narrative'],
    complaints_data['product'],
    complaints_data['predicted_risk']
)):
    narrative_clean = narrative[:500] if len(narrative) > 500 else narrative

    prompt = f"""Analyze this consumer financial complaint and return ONLY valid JSON with these fields:
{{
    "root_cause": "2-3 word summary of the core issue (e.g., 'Identity Theft', 'Wage Garnishment', 'Credit Dispute')",
    "consumer_harm": "Type of harm: financial, emotional, time-based, reputational, or operational",
    "severity": "low, medium, or high based on consumer impact",
    "explanation": "1-2 sentences explaining why this complaint is flagged as {risk_level} risk"
}}

Complaint: "{narrative_clean}"
Product Category: {product}
Predicted Risk Level: {risk_level}"""

    try:
        response_text = call_llm_with_retry(prompt)
        response_text = response_text.strip()

        # Strip markdown code blocks if present
        if response_text.startswith("```"):
            parts = response_text.split("```")
            response_text = parts[1] if len(parts) > 1 else response_text
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()

        analysis = json.loads(response_text)

        analyses.append({
            'narrative': narrative[:200],
            'product': product,
            'predicted_risk': risk_level,
            'root_cause': analysis.get('root_cause', 'N/A'),
            'consumer_harm': analysis.get('consumer_harm', 'N/A'),
            'severity': analysis.get('severity', 'N/A'),
            'explanation': analysis.get('explanation', ''),
        })

        progress = (idx + 1) / len(complaints_data)
        status = "✓" if risk_level == 'high' else ("◆" if risk_level == 'medium' else "○")
        print(f"  [{progress*100:5.1f}%] {status} {risk_level.upper():6s} - {analysis.get('root_cause', 'Unknown')[:30]:30s}")

        time.sleep(0.5)  # polite delay — Qwen free tier is generous

    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON parse error on complaint {idx}: {e}")
        analyses.append({
            'narrative': narrative[:200],
            'product': product,
            'predicted_risk': risk_level,
            'root_cause': 'Parse Error',
            'consumer_harm': 'Unknown',
            'severity': 'Unknown',
            'explanation': 'Failed to parse LLM response',
        })
    except Exception as e:
        print(f"  ⚠ API error on complaint {idx}: {e}")
        analyses.append({
            'narrative': narrative[:200],
            'product': product,
            'predicted_risk': risk_level,
            'root_cause': 'API Error',
            'consumer_harm': 'Unknown',
            'severity': 'Unknown',
            'explanation': str(e)[:100],
        })

# ============================================================================
# SECTION 6: Export Results
# ============================================================================

print("\n" + "=" * 80)
print("EXPORTING RESULTS")
print("=" * 80)

os.makedirs('outputs/task3_gemini', exist_ok=True)

results_df = pd.DataFrame(analyses)

results_df.to_csv('outputs/task3_gemini/gemini_complaint_analysis.csv', index=False)
print("\n  ✓ Saved: outputs/task3_gemini/gemini_complaint_analysis.csv")

# Summary statistics
print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)

print("\nRisk Distribution (Predicted):")
for risk in ['high', 'medium', 'low']:
    count = (results_df['predicted_risk'] == risk).sum()
    pct = 100 * count / len(results_df)
    print(f"  • {risk.upper():6s}: {count:3d} ({pct:5.1f}%)")

successful = len(results_df) - results_df['root_cause'].isin(['API Error', 'Parse Error']).sum()
print(f"\n  Successful analyses: {successful}/{len(results_df)}")

print("\nRoot Causes Identified:")
for cause, count in results_df['root_cause'].value_counts().head(10).items():
    pct = 100 * count / len(results_df)
    print(f"  • {cause:35s}: {count:3d} ({pct:5.1f}%)")

print("\nConsumer Harm Types:")
for harm, count in results_df['consumer_harm'].value_counts().items():
    pct = 100 * count / len(results_df)
    print(f"  • {harm:35s}: {count:3d} ({pct:5.1f}%)")

summary = {
    'total_analyzed': len(results_df),
    'successful': int(successful),
    'risk_distribution': results_df['predicted_risk'].value_counts().to_dict(),
    'root_causes': results_df['root_cause'].value_counts().to_dict(),
    'harm_types': results_df['consumer_harm'].value_counts().to_dict(),
    'severity_distribution': results_df['severity'].value_counts().to_dict(),
}

with open('outputs/task3_gemini/analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\n  ✓ Saved: outputs/task3_gemini/analysis_summary.json")

# ============================================================================
# SAMPLE COMPLAINTS BY RISK LEVEL
# ============================================================================

print("\n" + "=" * 80)
print("SAMPLE COMPLAINTS BY RISK LEVEL")
print("=" * 80)

for risk_level in ['high', 'medium', 'low']:
    risk_rows = results_df[results_df['predicted_risk'] == risk_level]
    if len(risk_rows) == 0:
        continue
    print(f"\n{risk_level.upper()}-RISK ({len(risk_rows)} complaints)")
    print("-" * 80)
    sample = risk_rows.sample(1).iloc[0]
    print(f"  Root Cause: {sample['root_cause']}")
    print(f"  Harm Type:  {sample['consumer_harm']}")
    print(f"  Severity:   {sample['severity']}")
    print(f"  Why Flagged: {sample['explanation'][:150]}")
    print(f"  Complaint:  {sample['narrative'][:100]}...")

print("\n" + "=" * 80)
print("QWEN ANALYSIS COMPLETE")
print("=" * 80)
print(f"""
Outputs saved to outputs/task3_gemini/:
  • gemini_complaint_analysis.csv  — Full LLM analysis for all {len(results_df)} complaints
  • analysis_summary.json           — Summary statistics

Next Steps:
  1. View results in Streamlit dashboard (LLM Analysis tab)
  2. Filter by predicted_risk to focus on HIGH/MEDIUM priority
  3. Review explanation field for triage reasoning
""")
