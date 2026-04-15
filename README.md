# Text Miners

This repository contains notebooks and datasets for complaint text preprocessing and downstream NLP tasks.

## Repository structure

```text
text-miners/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── complaints.csv
│   ├── complaints_processed_full.csv
│   ├── complaints_processed_full.csv.zip
│   ├── train_data.csv
│   ├── test_data.csv
│   ├── train_indices.csv
│   ├── test_indices.csv
│   ├── annotation_sample.csv
│   ├── annotation_sample_labelled.csv
│   ├── split_info.txt
│   └── data_info.py
├── preprocessing/
│   ├── data_preprocessing.ipynb
│   ├── data_preprocessing.py
│   └── data_splitting.ipynb
├── task1_topic_modelling/
│   ├── topic_modelling.ipynb
│   └── topic_modelling.py
├── task2_classification/
│   ├── classification.ipynb
│   └── classification.py
├── task3_risk_rating/
│   ├── annotation_guide.md
│   ├── risk_rating.ipynb
│   ├── risk_rating.py
│   ├── risk_rating2.ipynb
│   ├── risk_rating2.py
│   ├── task3_qwen_analysis.py
│   ├── task3_risk_clustering.py
│   └── task3_risk_clustering_bertopic.py
├── correct_annotations.py
├── outputs/
└── demo/
    └── streamlit.py
```

## Data files and purpose

- `data/complaints_processed_full.csv` (also available as `.zip`): cleaned full dataset — the **minimum required starting point** for running the pipeline. Produced from `complaints.csv` via `data_preprocessing.ipynb`; unzip and use directly without re-running preprocessing.
- `data/complaints.csv`: original raw dataset. Only needed if you want to regenerate `complaints_processed_full.csv` from scratch; not required otherwise.
- `data/train_data.csv` / `data/test_data.csv`: stratified 80/20 split of the full dataset (115,169 train / 28,793 test).
- `data/train_indices.csv` / `data/test_indices.csv`: row indices for reproducible split reconstruction.
- `data/annotation_sample.csv`: stratified random sample (~0.6% of train set, ~692 complaints) drawn for manual risk labelling.
- `data/annotation_sample_labelled.csv`: the annotation sample with `risk_label` (high / medium / low) filled in via keyword-based rules; used by all Task 3 models.
- `data/split_info.txt`: metadata about the train/test split (class distribution, split parameters).

## Dependency and environment requirements

The repository currently declares:

- Python `>=3.12` in `requirements.txt`
- pandas `>=1.5.0` in `requirements.txt`

The preprocessing notebooks also import additional libraries. A complete environment for running notebooks is:

- Python `3.12`
- pandas `>=1.5.0`
- numpy `>=1.24.0`
- nltk `>=3.8.1`
- plotly `>=5.0.0`
- scikit-learn `>=1.3.0`
- jupyter `>=1.0.0`

Install example:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "pandas>=1.5.0" "numpy>=1.24.0" "nltk>=3.8.1" "plotly>=5.0.0" "scikit-learn>=1.3.0" "jupyter>=1.0.0"
```

Optional NLTK assets (if missing at runtime):

```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
```

## How `complaints_processed_full.csv` is created from `complaints.csv`

The transformation is implemented in `preprocessing/data_preprocessing.ipynb`.

1. Load source data:
   - `pd.read_csv('../data/complaints.csv')`
2. Keep and rename two columns:
   - `Product` -> `product`
   - `Consumer complaint narrative` -> `narrative`
3. Normalize product labels into grouped categories:
   - `credit_reporting`
   - `debt_collection`
   - `credit_card`
   - `mortgages_and_loans`
   - `retail_banking`
4. Preprocess `narrative` text:
   - tokenize with NLTK
   - lowercase
   - remove stopwords and punctuation
   - remove non alphabetic tokens
   - lemmatize with `WordNetLemmatizer`
   - rejoin tokens into cleaned space separated text
5. Finalize dataset:
   - `drop_duplicates()`
   - keep non null narratives only
6. Export:
   - `df.to_csv('../data/complaints_processed_full.csv', index=False)`

The train/test split is performed separately in `preprocessing/data_splitting.ipynb`, which produces `train_data.csv`, `test_data.csv`, and the corresponding index files.

## Quick start

1. Create and activate a Python 3.12 virtual environment.
2. Install dependencies listed above.
3. Unzip `data/complaints_processed_full.csv.zip` — this is the only data file you need to start.
4. Run `preprocessing/data_splitting.ipynb` to generate `train_data.csv`, `test_data.csv`, and the index files.
5. Run the task notebooks in order: `task1_topic_modelling/`, `task2_classification/`, `task3_risk_rating/`.

> **Note:** `data/complaints.csv` and `preprocessing/data_preprocessing.ipynb` are only needed if you want to regenerate `complaints_processed_full.csv` from the original raw data. Skip them otherwise.

## Demo dashboard

The `demo/streamlit.py` app is a Streamlit dashboard for exploring the complaint analysis pipeline. It reads the exported CSV and JSON artifacts under `outputs/` and presents five tabs in the sidebar:

- `Overview`: summary metrics and dataset distribution
- `Classification`: product category model results and complaint search
- `Risk Queue`: annotated and predicted risk prioritisation
- `LLM Analysis`: Qwen-powered semantic analysis of complaints
- `Topic Explorer`: LDA topic browsing, sunburst, and keyword charts

### Tab descriptions

#### Overview (`🏠 Overview`)

The landing page. Displays four headline stat cards at the top:

| Card | Value |
|------|-------|
| Complaints Processed | 143,962 (full dataset) |
| Product Categories | 5 normalised classes |
| Topics Discovered | 10 (LDA coherence k=10) |
| Best Macro F1 | 0.8545 (Logistic Regression) |

Below the metrics:

- **Distribution by Product Category** — a horizontal bar chart and a donut pie chart side-by-side showing how the 115,169 training complaints are split across the five product classes (Credit Reporting, Debt Collection, Mortgages & Loans, Credit Card, Retail Banking).
- **Topic Distribution Across Complaints** — a horizontal bar chart of dominant topic frequencies sampled from 5,000 complaints, using human-readable topic labels from `outputs/topic_labels.json` when available.

The sidebar also shows a **Pipeline Status** panel with green/grey indicators for each pipeline stage (Data Split, Topic Modelling, Classification, Risk Rating).

---

#### Classification (`🏷️ Classification`)

Shows the results from Task 2: multi-class NLP classification of complaints into five financial product categories.

- **Model Comparison** — three stat cards comparing Naive Bayes (F1 0.8138), Logistic Regression (F1 0.8545 ★ Best), and Neural Network MLP (F1 0.8533), each with macro F1 and accuracy.
- **Per-class F1 Score** — horizontal bar chart showing Logistic Regression's per-category F1 scores (Credit Reporting 0.90, Retail Banking 0.89, Mortgages & Loans 0.86, Credit Card 0.81, Debt Collection 0.81) with a 0.85 target line.
- **All Models Side-by-side F1** — horizontal bar chart comparing the three models on a zoomed axis for easy comparison.
- **Confusion Matrices** — displayed as a static image if `outputs/confusion_matrices.png` exists.

When `outputs/classification_results.csv` is present, additional live views are unlocked:

- **Actual vs Predicted Distribution** — two donut charts side-by-side comparing the true and predicted class distributions on the test set.
- **Accuracy by Product Category** — horizontal bar chart of per-category accuracy.
- **Search Complaints** — keyword search box that filters complaint narratives in real time and displays up to 20 matching rows (narrative, actual product, predicted product).

---

#### Risk Queue (`⚠️ Risk Queue`)

Shows the results from Task 3: triaging complaints by predicted risk level so high-risk cases surface first for analyst review.

Always visible (from `data/annotation_sample_labelled.csv`):

- **Annotation Label Distribution** — three stat cards showing how many of the 692 hand-labelled complaints were rated high / medium / low risk.
- **Risk Split Donut** — donut chart of the annotation sample with high risk pulled out.
- **Risk × Product Heatmap** — stacked horizontal bar chart showing the share of each risk label within each product category for the annotated sample.

If `outputs/risk_rating_confusion_matrix.png` exists:

- **Confusion Matrix** — the risk-rating model's confusion matrix image.

When `outputs/risk_results.csv` is present:

- **Predicted Risk Counts** — three stat cards (high / medium / low) for the full test set.
- **Risk % by Product Heatmap** — colour-coded grid (green → yellow → red) showing predicted risk share per product category.
- **High-Risk Complaint Browser** — filterable table (by product) showing up to 50 high-risk complaints with their narrative, product, and risk label colour-coded in red.

---

#### LLM Analysis (`🤖 LLM Analysis`)

Uses Qwen (via `task3_risk_rating/task3_qwen_analysis.py`) to semantically enrich complaints already classified by Logistic Regression, surfacing root causes, consumer harm types, and severity judgements.

Requires `outputs/task3_gemini/gemini_complaint_analysis.csv` to be generated first by running:

```bash
python task3_risk_rating/task3_qwen_analysis.py
```

This script calls `qwen-plus` via the Alibaba DashScope API (set `DASHSCOPE_API_KEY` in a `.env` file). It loads LR's predicted risk from `outputs/task3_lr_predictions.pkl` and, for each complaint, returns four fields: `root_cause`, `consumer_harm`, `severity`, and `explanation`.

- **Analysis Overview** — four stat cards: total complaints analysed, unique root causes identified, high-risk count, high-severity count.
- **Root Cause Analysis** — horizontal bar chart ranking the most common root causes across all analysed complaints.
- **Severity Breakdown** — inline summary showing high / medium / low severity counts and percentages.
- **Risk Level Distribution** — bar chart of predicted risk levels (high / medium / low) colour-coded red / amber / green.
- **Consumer Harm Types** — donut chart showing the distribution of harm categories identified by the LLM.
- **Browse Analyzed Complaints** — filterable table (by root cause and risk level) showing up to 20 complaints with narrative, product, predicted risk, root cause, consumer harm, severity, and LLM explanation.

---

#### Topic Explorer (`🔍 Topic Explorer`)

Visualises the 10 latent topics discovered by LDA in Task 1.

Requires `outputs/complaints_with_topics.csv` and `outputs/topic_vectors.csv`.

- **Topic Distribution by Product — Interactive Sunburst** — two-level sunburst chart (outer ring = topic, inner ring = product category) that lets you drill into which topics appear in which product lines.
- **Complaint Volume by Topic** — horizontal bar chart showing how many of the 10,000 sampled complaints belong to each topic, with gradient colouring.
- **Topic × Product Category Heatmap** — static heatmap image (`outputs/topic_category_heatmap.png`) if available.
- **Top Keywords per Topic** — static bar/chart image (`outputs/topic_keywords_labeled.png` or `outputs/topic_keywords.png`) showing the highest-weight words for each of the 10 topics.
- **Browse Complaints by Topic** — dropdown to select a topic, followed by a product breakdown bar chart and a table of up to 15 sample complaint narratives assigned to that topic.

### What the demo requires

Run the demo only after the pipeline artifacts have been generated. The dashboard expects these files to exist:

- `data/train_data.csv`
- `outputs/topic_vectors.csv`
- `outputs/topic_labels.json`
- `outputs/complaints_with_topics.csv`
- `outputs/classification_results.csv`
- `outputs/risk_results.csv`
- `outputs/topic_category_heatmap.png`
- `outputs/topic_keywords.png`

The app also uses the following Python packages:

- `streamlit`
- `plotly`
- `pandas`
- `numpy`

The existing `requirements.txt` already includes the core notebook and demo dependencies. If Streamlit is not installed yet, install it in the same environment you use for the notebooks.

### How to start the demo

From the repository root, run:

```bash
streamlit run demo/streamlit.py
```

The app uses the repository root as its working base, so run the command from the top-level `text-miners/` folder rather than from inside `demo/`.

### How it works

The app loads data with small cached helper functions, so repeated navigation is fast. If an expected artifact is missing, the app shows a friendly warning or placeholder message instead of crashing.

The `Topic Explorer` page uses the topic labels from `outputs/topic_labels.json` to display human-readable topic names, and it reuses the topic vectors generated by `task1_topic_modelling/topic_modelling.ipynb`.

The `Classification` and `Risk Queue` views are driven by exported model results from the later notebooks, so you should run those notebooks before opening the dashboard if you want the full experience.

## Git workflow: notebook metadata stripping

This repository uses a pre-commit hook for notebooks:

- Hook config file: `.pre-commit-config.yaml`
- Hook: `nbstripout`
- Purpose: strip notebook outputs and transient metadata before commit

One time setup per clone:

```bash
python3 -m pip install nbstripout pre-commit
python3 -m nbstripout --install
python3 -m pre_commit install
```

Manual verification:

```bash
python3 -m pre_commit run --all-files
```
