# Risk Rating Annotation Guide

**Project:** IS450 Text Mining — CFPB Complaint Risk Rating  
**Task:** Assign a risk rating of `low`, `medium`, or `high` to each consumer complaint narrative  
**Target sample:** 500–1,000 complaints drawn from `train_data.csv`

---

## 1. Purpose

This guide ensures all annotators apply the same criteria when labelling complaints. The resulting labelled subset will serve as ground truth for training and evaluating the risk rating model in `risk_rating.ipynb`.

---

## 2. Risk Rating Schema

### LOW
The complaint describes a routine administrative issue with no indication of acute financial harm, legal action, or regulatory urgency.

**Criteria (all of the following):**
- No mention of financial hardship (e.g. inability to pay rent, food, utilities)
- No mention of legal proceedings, lawsuits, or attorney involvement
- No evidence of identity theft, fraud, or data breach
- Issue is a billing error, incorrect address, or minor reporting inaccuracy
- Tone is factual or mildly frustrated

**Typical phrases:**
> "incorrect address on my credit report", "wrong balance shown", "account still listed after closure", "duplicate entry", "never received a statement"

---

### MEDIUM
The complaint describes a meaningful financial or procedural harm that has had a material impact on the consumer's finances or access to credit, but does not involve immediate crisis, legal action, or ongoing fraud.

**Criteria (at least one of the following):**
- Denied credit, loan, or housing due to incorrect reporting
- Debt collection harassment (repeated calls, threats) without active legal proceedings
- Unauthorised account opened but issue appears contained
- Significant billing dispute that has not been resolved after multiple attempts
- Evidence of emotional distress or consumer expressing they feel trapped or helpless
- Mentions of late fees, penalty interest, or damaged credit score affecting life decisions

**Typical phrases:**
> "denied mortgage because of error", "called me 10 times a day", "refused to investigate my dispute", "my credit score dropped 100 points", "lost my apartment application"

---

### HIGH
The complaint describes an acute crisis: severe financial hardship, ongoing fraud or identity theft, legal proceedings, potential regulatory breach, or a situation where delayed intervention could cause irreversible harm.

**Criteria (at least one of the following):**
- Consumer reports inability to afford basic necessities (rent, food, medication) as a direct result of the issue
- Active identity theft or widespread fraudulent accounts
- Wage garnishment or bank account levy
- Ongoing or threatened legal action (lawsuit filed, attorney contacted, court summons received)
- Complaint alleges illegal debt collection practices under FDCPA
- Elderly or vulnerable consumer being targeted
- Consumer references military service in context of financial harm (SCRA protections)
- Mentions of bankruptcy triggered by the issue

**Typical phrases:**
> "cannot pay rent", "my wages are being garnished", "filed a lawsuit", "opened 12 accounts in my name", "threatening to sue me", "I am on a fixed income and cannot afford", "deployed overseas and they charged my account"

---

## 3. Annotation Decision Tree

```
Does the complaint mention legal action, lawsuit, garnishment, or identity theft affecting multiple accounts?
  YES → HIGH

Does it mention inability to afford basic needs (rent/food/medication)?
  YES → HIGH

Does it mention debt collection harassment, credit denial, or a significant unresolved dispute affecting major life decisions?
  YES → MEDIUM

Does it describe a minor reporting error, duplicate entry, or billing inaccuracy with no downstream harm stated?
  YES → LOW

If unclear → escalate to team lead or use MEDIUM as a conservative default
```

---

## 4. Annotation Process

### Step 1 — Sample selection
Draw a stratified random sample from `train_data.csv`, proportional to product category:

```python
import pandas as pd

train_df = pd.read_csv('../data/train_data.csv')
sample = train_df.groupby('product', group_keys=False).apply(
    lambda x: x.sample(frac=0.006, random_state=42)
).reset_index(drop=True)
sample['risk_label'] = ''  # column to fill in
sample[['narrative', 'product', 'risk_label']].to_csv(
    '../data/annotation_sample.csv', index=False
)
print(f"Sample size: {len(sample)}")
```

### Step 2 — Fill in labels
Open `data/annotation_sample.csv` and fill the `risk_label` column with `low`, `medium`, or `high` for each row. Read the full narrative, not just the first sentence.

### Step 3 — Inter-annotator check (optional but recommended)
If two or more members annotate an overlapping subset (~100 complaints), compute Cohen's Kappa to measure agreement:

```python
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(annotator1_labels, annotator2_labels)
print(f"Cohen's Kappa: {kappa:.3f}")
# Target: kappa > 0.6 (substantial agreement)
```

Resolve disagreements by discussion and update the guide if a pattern is found.

### Step 4 — Save final labels
Save the completed file as `data/annotation_sample_labelled.csv`. This file is the input to `risk_rating.ipynb`.

---

## 5. Common Edge Cases

| Situation | Guidance |
|-----------|----------|
| Consumer mentions a lawyer but no active lawsuit | MEDIUM |
| Complaint about credit score drop due to an error | MEDIUM |
| Consumer says "this is ruining my life" with no concrete financial harm | LOW–MEDIUM, use MEDIUM |
| Identity theft with only one fraudulent inquiry | MEDIUM |
| Identity theft with multiple accounts opened | HIGH |
| Consumer mentions elderly parent's account | HIGH |
| Military/veteran mentions SCRA | HIGH |
| Complaint is very short (< 10 words) | LOW unless specific HIGH keywords present |
| Consumer references a previous CFPB complaint | No change — rate on current narrative only |

---

## 6. Quality Checks

Before submitting labelled data:
- [ ] No blank `risk_label` cells
- [ ] Only values `low`, `medium`, or `high` in `risk_label` column
- [ ] At least 15% of labels are `high` (if fewer, review HIGH criteria — it may be under-applied)
- [ ] At least 20% of labels are `low`
- [ ] File saved as `data/annotation_sample_labelled.csv`

---

## 7. Label Distribution Target

Based on domain knowledge of CFPB complaint data, aim for approximately:

| Label | Target % |
|-------|----------|
| low | 30–40% |
| medium | 40–50% |
| high | 15–25% |

If your distribution is very different, revisit the criteria before proceeding to model training.