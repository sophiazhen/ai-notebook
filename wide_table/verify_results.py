import pandas as pd
from pathlib import Path

# Check the results
data_path = Path("E:/qingzhu/FAB/ai-notebook/data_output")

# Read the wide table
df = pd.read_excel(data_path / "processed_wide_table.xlsx")

print("=== Verification Results ===")
print(f"Wide table shape: {df.shape}")
print(f"Number of wafers: {len(df)}")
print(f"Number of features: {len(df.columns)}")
print("\n" + "="*50 + "\n")

# Show columns by type
ocd_cols = [col for col in df.columns if 'OCD' in col]
thk_cols = [col for col in df.columns if 'THK' in col]
param_cols = [col for col in df.columns if any(agg in col for agg in ['_mean', '_min', '_max', '_median', '_stdev'])]

print(f"OCD columns ({len(ocd_cols)}):")
for col in ocd_cols:
    print(f"  - {col}")
print()

print(f"THK columns ({len(thk_cols)}):")
for col in thk_cols:
    print(f"  - {col}")
print()

print(f"Parameter aggregate columns ({len(param_cols)}):")
print(f"  Sample parameters (first 10):")
for col in param_cols[:10]:
    print(f"    - {col}")
if len(param_cols) > 10:
    print(f"    ... and {len(param_cols)-10} more")
print()

# Check metadata columns
print("Metadata columns:")
meta_cols = [col for col in df.columns if col not in ocd_cols + thk_cols + param_cols]
for col in meta_cols:
    print(f"  - {col}")

print("\n" + "="*50 + "\n")

# Show first few rows with key columns
key_cols = ['wafer_id'] + ocd_cols + thk_cols[:2] + param_cols[:3]
if len(key_cols) > 10:
    key_cols = key_cols[:10]

print("Sample data (first 3 rows):")
print(df[key_cols].head(3))

print("\n" + "="*50 + "\n")

# Check for step naming consistency
step_pattern_cols = [col for col in param_cols if any(f'_{i}_' in col for i in range(1, 40))]
print(f"Step pattern columns (step extraction snippet):")
step_counts = {}
for col in step_pattern_cols[:20]:
    # Extract step number from column name
    parts = col.split('_')
    for i, part in enumerate(parts):
        if part.isdigit() and 1 <= int(part) <= 40:
            step = int(part)
            step_counts[step] = step_counts.get(step, 0) + 1
            break

for step, count in sorted(step_counts.items()):
    print(f"  Step {step}: {count} parameters")

print(f"\nStep naming follows pattern: parameter_name_step_number_aggregate")
print("Example: RF_Source_Power_1_stdev")