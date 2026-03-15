import pandas as pd
import os

def get_data_info(file_path):
    data = pd.read_csv(file_path)
    num_rows = len(data)
    num_cols = len(data.columns)
    data_types = data.dtypes
    product_types = data["product"].unique()
    missing_values = data.isnull().sum()
    duplicates = data.duplicated().sum()
    return num_rows, num_cols, data_types, product_types, missing_values, duplicates
    
if __name__ == "__main__":
    file_path = "complaints_processed_full.csv"  # os.path.join() is unnecessary with one arg
    num_rows, num_cols, data_types, product_types, missing_values, duplicates = get_data_info(file_path)
    print(f"Number of rows: {num_rows}")
    print(f"Number of columns: {num_cols}")
    print(f"Data types:\n{data_types}")
    print(f"Product types: {product_types}")
    print(f"\nMissing values:\n{missing_values}")
    print(f"Duplicate rows: {duplicates}")
