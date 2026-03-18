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
│   ├── complaints_processed.csv
│   ├── complaints_processed_full.csv
│   ├── complaints_processed_full.csv.zip
│   ├── credit_card_text.txt
│   ├── credit_card_text_processed.csv
│   ├── credit_reporting_text.txt
│   ├── credit_reporting_text_processed.csv
│   ├── credit_reporting_text_processed.csv.zip
│   ├── debt_collection_text.txt
│   ├── debt_collection_text_processed.csv
│   ├── mortgages_and_loans_text.txt
│   ├── mortgages_and_loans_text_processed.csv
│   ├── retail_banking_text.txt
│   ├── retail_banking_text_processed.csv
│   └── data_info.py
├── preprocessing/
│   ├── 1_EDA-1.ipynb
│   ├── 2_EDA-2.ipynb
│   └── 3_prepping_data.ipynb
├── task1_topic_modelling/
│   └── topic_modelling.ipynb
├── task2_classification/
│   └── classification.ipynb
├── task3_risk_rating/
│   ├── annotation_guide.md
│   └── risk_rating.ipynb
├── outputs/
└── demo/
    └── streamlit.py
```

Notes:

- `task1_topic_modelling/topic_modelling.ipynb`: LDA topic modelling pipeline (see section below).
- `task2_classification/classification.ipynb`: empty notebook scaffold, classification work in progress.
- `outputs/` is present but currently empty.
- `task3_risk_rating/annotation_guide.md` is currently empty.
- `task3_risk_rating/risk_rating.ipynb` is a minimal valid notebook scaffold.
- `demo/streamlit.py` is an empty placeholder for a future Streamlit demo application.

## Data files and purpose

- `data/complaints.csv`: original complaint dataset used as the primary source.
- `data/complaints_processed_full.csv`: cleaned full dataset produced from `complaints.csv` in preprocessing.
- `data/*_text.txt` and `data/*_text_processed.csv`: category level intermediate and processed artifacts.
- `data/complaints_processed.csv`: legacy processed dataset artifact.

## Dependency and environment requirements

The repository currently declares:

- Python `>=3.12` in `requirements.txt`
- pandas `>=1.5.0` in `requirements.txt`
- gensim `>=4.3.0` in `requirements.txt`
- matplotlib `>=3.7.0` in `requirements.txt`

The preprocessing notebooks also import additional libraries. A complete environment for running notebooks is:

- Python `3.12`
- pandas `>=1.5.0`
- numpy `>=1.24.0`
- nltk `>=3.8.1`
- plotly `>=5.0.0`
- scikit-learn `>=1.3.0`
- gensim `>=4.3.0`
- matplotlib `>=3.7.0`
- jupyter `>=1.0.0`

Install example:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "pandas>=1.5.0" "numpy>=1.24.0" "nltk>=3.8.1" "plotly>=5.0.0" "scikit-learn>=1.3.0" "gensim>=4.3.0" "matplotlib>=3.7.0" "jupyter>=1.0.0"
```

Optional NLTK assets (if missing at runtime):

```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
```

## How `complaints_processed_full.csv` is created from `complaints.csv`

The transformation is implemented in `preprocessing/3_prepping_data.ipynb`.

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

Related preprocessing notebooks:

- `preprocessing/1_EDA-1.ipynb`: reads `complaints.csv` and writes category specific processed CSV files.
- `preprocessing/2_EDA-2.ipynb`: performs additional EDA on category processed files.

## Task 1: Topic modelling (`task1_topic_modelling/topic_modelling.ipynb`)

This notebook performs unsupervised LDA topic modelling on `data/complaints_processed_full.csv`.

Steps:

1. Load preprocessed data:
   - `pd.read_csv('../data/complaints_processed_full.csv')`
2. Build gensim corpus:
   - tokenize `narrative` column with `.str.split()`
   - build a `corpora.Dictionary` and convert each document to bag-of-words
3. Select optimal number of topics:
   - evaluate `CoherenceModel` (metric: `c_v`) for `num_topics` in `[5, 10, 15, 20, 25, 30]`
   - plot coherence scores and select the highest
4. Train final LDA model:
   - `gensim.models.LdaModel` with optimal `num_topics`
5. Inspect results:
   - print top-10 words per topic
   - extract per-document topic vectors
   - assign dominant topic to each document

Required additional dependencies: `gensim>=4.3.0`, `matplotlib>=3.7.0`.

## Quick start

1. Create and activate a Python 3.12 virtual environment.
2. Install dependencies listed above.
3. Open notebooks from `preprocessing/` in Jupyter.
4. Run `3_prepping_data.ipynb` to regenerate `data/complaints_processed_full.csv`.

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
