import pandas as pd
import numpy as np
import os

# Define the source data directory
source_data_dir = "E:/qingzhu/FAB/source_data"

print("=== Examining etching-measurement.xlsx ===")
try:
    df_etching = pd.read_excel(os.path.join(source_data_dir, "etching-measurement.xlsx"))
    print(f"Shape: {df_etching.shape}")
    print(f"Columns: {list(df_etching.columns)}")
    print("\nFirst 10 rows:")
    print(df_etching.head(10))
    print("\nColumn details:")
    print(df_etching.dtypes)
except Exception as e:
    print(f"Error reading etching-measurement.xlsx: {e}")

print("\n" + "="*70 + "\n")

print("=== Examining sampling-wafer-list.xlsx ===")
try:
    df_sampling = pd.read_excel(os.path.join(source_data_dir, "sampling-wafer-list.xlsx"))
    print(f"Shape: {df_sampling.shape}")
    print(f"Columns: {list(df_sampling.columns)}")
    print("\nFirst 10 rows:")
    print(df_sampling.head(10))
    print("\nColumn details:")
    print(df_sampling.dtypes)
except Exception as e:
    print(f"Error reading sampling-wafer-list.xlsx: {e}")

print("\n" + "="*70 + "\n")