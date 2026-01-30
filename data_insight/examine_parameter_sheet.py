import pandas as pd
import os

# Define the source data directory
source_data_dir = "~/FAB/source_data"
wafer_file = "WFR_300MM_20250126_L001.xlsx"

print(f"=== Examining parameter data sheets in {wafer_file} ===")

# Let's examine a few parameter sheets to understand their structure
# I'll check a few different types of parameters

sheets_to_examine = [
    'sheet-1-RF Source Power',      # RF parameter
    'sheet-29-SF6 Gas Flow',        # Gas flow parameter
    'sheet-68-Trench Profile Angle' # Output/measurement parameter
]

for sheet_name in sheets_to_examine:
    print(f"\n{'='*60}")
    print(f"Examining sheet: {sheet_name}")
    print('='*60)

    try:
        df = pd.read_excel(os.path.join(source_data_dir, wafer_file), sheet_name=sheet_name)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nData types:")
        print(df.dtypes)

        # Check for step/time related columns
        step_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['step', 'time', 'seq', 'sequence'])]
        if step_cols:
            print(f"\nStep/Time columns: {step_cols}")

        print(f"\nFirst 10 rows:")
        print(df.head(10))

        print(f"\nLast 5 rows:")
        print(df.tail(5))

        # Check if there are any missing values
        print(f"\nMissing values per column:")
        print(df.isnull().sum())

        # Look for unique values in non-numeric columns
        for col in df.columns:
            if df[col].dtype in ['object', 'string']:
                unique_vals = df[col].unique()
                if len(unique_vals) <= 10:
                    print(f"\nUnique values in '{col}': {unique_vals}")
                else:
                    print(f"\nUnique values in '{col}': {len(unique_vals)} unique values")
                    print(f"Sample values: {unique_vals[:5]}")

        print(f"\nBasic statistics for numeric columns:")
        print(df.describe())

    except Exception as e:
        print(f"Error reading sheet {sheet_name}: {e}")

# Let's also check how many steps are typically in a process
print(f"\n{'='*60}")
print("Analyzing step patterns across multiple parameter sheets")
print('='*60)

# Read metadata to get total steps
metadata = pd.read_excel(os.path.join(source_data_dir, wafer_file), sheet_name='metadata')
total_steps = metadata['total_steps'].iloc[0] if 'total_steps' in metadata.columns else 'Unknown'
print(f"Total steps from metadata: {total_steps}")

# Check step column in a few parameter sheets to see if they're consistent
rf_power_df = pd.read_excel(os.path.join(source_data_dir, wafer_file), sheet_name='sheet-1-RF Source Power')
if 'STEP_NO' in rf_power_df.columns:
    unique_steps = rf_power_df['STEP_NO'].nunique()
    print(f"Number of unique steps in RF Source Power: {unique_steps}")
    print(f"Step sequence: {sorted(rf_power_df['STEP_NO'].unique())}")