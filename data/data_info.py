import pandas as pd

def get_data_info(file_path):
    df = pd.read_csv(file_path)

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Rows:     {len(df):,}")
    print(f"Columns:  {len(df.columns)}")
    print(f"Columns:  {list(df.columns)}")

    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)
    print(df.dtypes.to_string())

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    missing = df.isnull().sum()
    print(missing.to_string())
    print(f"Total missing: {missing.sum()}")

    print("\n" + "=" * 60)
    print("DUPLICATES")
    print("=" * 60)
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    print("\n" + "=" * 60)
    print("PRODUCT CATEGORY DISTRIBUTION")
    print("=" * 60)
    counts = df["product"].value_counts()
    pct = df["product"].value_counts(normalize=True) * 100
    dist = pd.DataFrame({"count": counts, "pct": pct.round(2)})
    dist.index.name = "product"
    print(dist.to_string())

    print("\n" + "=" * 60)
    print("NARRATIVE LENGTH STATISTICS (characters)")
    print("=" * 60)
    char_len = df["narrative"].str.len()
    print(char_len.describe().round(1).to_string())

    print("\n" + "=" * 60)
    print("NARRATIVE LENGTH STATISTICS (words)")
    print("=" * 60)
    word_len = df["narrative"].str.split().str.len()
    print(word_len.describe().round(1).to_string())

    print("\n" + "=" * 60)
    print("NARRATIVE LENGTH BY PRODUCT (median words)")
    print("=" * 60)
    print(df.groupby("product")["narrative"].apply(
        lambda x: x.str.split().str.len().median()
    ).round(1).sort_values(ascending=False).to_string())

    print("\n" + "=" * 60)
    print("VOCABULARY SIZE (unique tokens across all narratives)")
    print("=" * 60)
    all_tokens = df["narrative"].str.split().explode()
    print(f"Total tokens:  {len(all_tokens):,}")
    print(f"Unique tokens: {all_tokens.nunique():,}")

    print("\n" + "=" * 60)
    print("SAMPLE NARRATIVES (1 per product)")
    print("=" * 60)
    for product, group in df.groupby("product"):
        sample = group["narrative"].iloc[0]
        print(f"\n[{product}]")
        print(sample[:300] + ("..." if len(sample) > 300 else ""))

    print("\n" + "=" * 60)


if __name__ == "__main__":
    get_data_info("complaints_processed_full.csv")
