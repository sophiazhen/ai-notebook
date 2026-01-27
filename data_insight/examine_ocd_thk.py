import pandas as pd
import os

# Define the source data directory
source_data_dir = "E:/qingzhu/FAB/source_data"

print("=== Examining OCD/THK indicators in etching-measurement.xlsx ===")

# Read etching measurement data
df_etching = pd.read_excel(os.path.join(source_data_dir, "etching-measurement.xlsx"))

print("Unique indicators found:")
indicators = df_etching['indicator'].unique()
for indicator in indicators:
    data = df_etching[df_etching['indicator'] == indicator]
    print(f"\n{indicator}:")
    print(f"  Number of wafers with this indicator: {len(data)}")
    print(f"  Units: {data['mean'].min():.2f} - {data['mean'].max():.2f}")
    print(f"  Wafers: {data['wafer_id'].tolist()}")

# Check all wafer files
all_wfr_files = [f for f in os.listdir(source_data_dir) if f.startswith('WFR_300MM_2025') and f.endswith('.xlsx')]
all_wfr_files.sort()

print(f"\n=== Found {len(all_wfr_files)} wafer data files ===")
print("Files:", all_wfr_files)

# Compare metadata across files
print("\n=== Comparing metadata across wafer files ===")
metadata_summary = []

for wfr_file in all_wfr_files:
    try:
        metadata = pd.read_excel(os.path.join(source_data_dir, wfr_file), sheet_name='metadata')
        metadata_summary.append({
            'wafer_id': metadata['wafer_id'].iloc[0],
            'product_type': metadata['product_type'].iloc[0],
            'recipe_id': metadata['recipe_id'].iloc[0],
            'chamber_id': metadata['chamber_id'].iloc[0],
            'total_steps': metadata['total_steps'].iloc[0],
            'process_duration_sec': metadata['process_duration_sec'].iloc[0]
        })
    except Exception as e:
        print(f"Error reading {wfr_file}: {e}")

metadata_df = pd.DataFrame(metadata_summary)
print(metadata_df)

# Check if all parameter sheets have the same structure across wafers
print(f"\n=== Checking parameter sheet consistency ===")
consistent_structure = True
base_structure = None

for i, wfr_file in enumerate(all_wfr_files[:2]):  # Check first 2 files
    excel_file = pd.ExcelFile(os.path.join(source_data_dir, wfr_file))

    # Check a parameter sheet structure
    param_sheet = 'sheet-1-RF Source Power'
    if param_sheet in excel_file.sheet_names:
        df = pd.read_excel(os.path.join(source_data_dir, wfr_file), sheet_name=param_sheet)
        structure = {
            'file': wfr_file,
            'columns': list(df.columns),
            'steps': df['Step'].nunique() if 'Step' in df.columns else 0,
            'rows': len(df)
        }

        if i == 0:
            base_structure = structure
            print(f"Base structure from {wfr_file}:")
            for k, v in structure.items():
                print(f"  {k}: {v}")
        else:
            print(f"\nComparing {wfr_file}:")
            for k, v in structure.items():
                if k == 'file':
                    continue
                print(f"  {k}: {v}")
                if structure[k] != base_structure[k]:
                    consistent_structure = False
                    print(f"    ⚠️ Different from base!")

print(f"\nParameter sheets have consistent structure: {'Yes' if consistent_structure else 'No'}")