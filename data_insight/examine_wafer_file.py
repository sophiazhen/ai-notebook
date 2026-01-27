import pandas as pd
import os

# Define the source data directory
source_data_dir = "E:/qingzhu/FAB/source_data"
wafer_file = "WFR_300MM_20250126_L001.xlsx"

print(f"=== Examining {wafer_file} ===")
try:
    # First, let's see all sheet names
    excel_file = pd.ExcelFile(os.path.join(source_data_dir, wafer_file))
    sheet_names = excel_file.sheet_names
    print(f"Sheet names: {sheet_names}")

    # Look for metadata sheet
    metadata_sheets = [s for s in sheet_names if 'metadata' in s.lower() or 'meta' in s.lower()]
    if metadata_sheets:
        print(f"\nFound metadata sheet(s): {metadata_sheets}")
        for meta_sheet in metadata_sheets[:1]:  # Just examine first one
            print(f"\n=== Examining {meta_sheet} ===")
            df_meta = pd.read_excel(os.path.join(source_data_dir, wafer_file), sheet_name=meta_sheet)
            print(f"Shape: {df_meta.shape}")
            print(f"Columns: {list(df_meta.columns)}")
            print("\nFirst 5 rows:")
            print(df_meta.head())

    # Look for parameter sheets
    param_sheets = [s for s in sheet_names if 'parameter' in s.lower() or 'param' in s.lower()]
    print(f"\nFound parameter sheets: {param_sheets[:5]}{'...' if len(param_sheets) > 5 else ''}")
    print(f"Total parameter sheets: {len(param_sheets)}")

    # Examine first few parameter sheets
    for i, param_sheet in enumerate(param_sheets[:3]):
        print(f"\n=== Examining {param_sheet} ===")
        try:
            df_param = pd.read_excel(os.path.join(source_data_dir, wafer_file), sheet_name=param_sheet)
            print(f"Shape: {df_param.shape}")
            print(f"Columns: {list(df_param.columns)}")

            # Check for time series data
            time_cols = [col for col in df_param.columns if 'time' in col.lower() or 'step' in col.lower() or 'step_no' in col.lower()]
            if time_cols:
                print(f"Time/Step related columns: {time_cols}")

            print("\nFirst 5 rows:")
            print(df_param.head())
            print("\nLast 5 rows:")
            print(df_param.tail())

        except Exception as e:
            print(f"Error reading sheet {param_sheet}: {e}")

    # Look for other potential data sheets
    data_sheets = [s for s in sheet_names if s not in metadata_sheets and s not in param_sheets]
    if data_sheets:
        print(f"\nOther sheets found: {data_sheets}")

except Exception as e:
    print(f"Error reading {wafer_file}: {e}")