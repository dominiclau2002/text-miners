import pandas as pd
import os

def get_data_info(file_path):
    data = pd.read_csv(file_path)
    num_rows = len(data)
    num_cols = len(data.columns)
    data_types = data.dtypes
    product_types = data["product"].unique()
    return num_rows, num_cols, data_types, product_types
    
if __name__ == "__main__":
    file_path = os.path.join("complaints_processed.csv")
    num_rows, num_cols, data_types, product_types = get_data_info(file_path)
    print(f"Number of rows: {num_rows}")
    print(f"Number of columns: {num_cols}")
    print(f"Data types: {data_types}")
    print(f"Product types: {product_types}")
