# Data Structure Analysis Report

## Overview
This report documents the structure of the source data Excel files for the FAB AI notebook project.

## File Locations
All source files are located in: `E:/qingzhu/FAB/source_data/`

## 1. Etching Measurement Data (`etching-measurement.xlsx`)

### Structure:
- **Shape**: 6 rows × 9 columns
- **Contents**: Measurement data for OCD and THK indicators

### Columns:
- `wafer_id`: Wafer identification (e.g., WFR_300MM_20250126_L001)
- `product_id`: Product identifier (e.g., p001, p002)
- `source_lot`: Lot identifier
- `site_count`: Number of measurement sites
- `indicator`: Measurement type (OCD or THK)
- `mean`: Mean measurement value
- `max`: Maximum measurement value
- `min`: Minimum measurement value
- `stdev`: Standard deviation

### Measurements:
- **OCD measurements**: Range from 11.80 to 12.30
- **THK measurements**: Range from 154.50 to 155.20
- All 3 wafers have both OCD and THK measurements

## 2. Sampling Wafer List (`sampling-wafer-list.xlsx`)

### Structure:
- **Shape**: 3 rows × 5 columns
- **Contents**: Metadata for selected wafers

### Columns:
- `wafer_id`: Wafer identification
- `product_id`: Product identifier
- `source_lot`: Lot identifier
- `chamber_id`: Process chamber (CHA_A, CHA_B, CHA_C)
- `recipe`: Process recipe (REC_TRENCH_M2M1_A, REC_TRENCH_M2M1_B, REC_TRENCH_M2M1_C)

## 3. Wafer Parameter Data Files (WFR_300MM_20250126_LXXX.xlsx)

### File List:
- WFR_300MM_20250126_L001.xlsx
- WFR_300MM_20250126_L002.xlsx
- WFR_300MM_20250126_L003.xlsx

### Sheet Structure:
Each file contains 74 sheets:
1. **`metadata`** - Process metadata
2. **`FDC_Parameters`** - Parameter mapping
3. **73 parameter sheets** - Time series data for each parameter (sheet-1-X to sheet-73-X)

### Metadata Sheet:
**Columns**: context_id, wafer_id, lot_id, tool_id, recipe_id, process_start_time, process_end_time, process_duration_sec, operator_id, chamber_id, product_type, wafer_diameter_mm, target_etch_depth_nm, actual_etch_depth_nm, etch_rate_avg, uniformity_percent, endpoint_time_sec, total_steps, data_logging_interval_ms, fdc_version, data_source, shift, maintenance_flag

### FDC_Parameters Sheet:
Maps parameter IDs to names with descriptions (73 parameters total)

### Parameter Data Sheets:
**Sheet naming**: `sheet-{number}-{parameter_name}`
**Structure**: All parameter sheets have identical structure
- **Shape**: 603 rows × 5 columns
- **Columns**:
  - `Parameter`: Parameter name
  - `No.`: Sequential recording number (1-603)
  - `Step`: Process step number (0-34, total 35 steps)
  - `Value`: Measured value
  - `Timestamp`: Timestamp of measurement

### Parameter Categories:
1. **RF Parameters** (1-20): RF power, voltage, current, impedance, etc.
2. **DC Parameters** (21-30): DC bias voltage, current, power
3. **Pressure Parameters** (31-35): Various pressure measurements
4. **Gas Flow Parameters** (36-49): Gas flow rates (SF6, CF4, CHF3, etc.)
5. **Temperature Parameters** (50-57): ESC, chamber, electrode temperatures
6. **OES Parameters** (58-64): Optical emission spectroscopy data
7. **Process Metrics** (65-73): Etch rate, uniformity, selectivity, etc.

### Time Series Structure:
- **Total recordings**: 603 measurements per parameter
- **Total steps**: 35 process steps (Step 0-34)
- **Data logging interval**: Variable (~1-2 seconds between recordings)
- **Process duration**: ~614 seconds (10.2 minutes)

### Key Observations:
1. **Consistent Structure**: All wafer files have identical sheet structure and column names
2. **Same Process**: All wafers have 35 steps and 603 recordings per parameter
3. **Time Alignment**: All parameters from the same wafer have synchronized timestamps
4. **Product Type**: All wafers are M2_M1_TRENCH product type
5. **Recipe Variation**: Each chamber uses a different recipe variant (A, B, C)

## Data Processing Implications:

### For Time Series Analysis:
- Each parameter can be aggregated by step (mean, std, min, max per step)
- Step-based statistics can be computed across all parameters
- Cross-correlation between parameters can be analyzed

### For Wide Table Creation:
- 3 wafers × 35 steps × 73 parameters = 7,665 potential observations
- Final feature space will be 73 parameters × statistics (mean, std, etc.)
- Can aggregate measurements within each step for feature engineering

### For ML Model Structure:
- Multi-step time series data suggests using techniques specific to sequential data
- Wide table format with parameter statistics per step is suitable for XGBoost/LightGBM
- OCD and THK measurements serve as target variables (regression problem)

## Next Steps for Implementation:
1. Develop step-based aggregation function
2. Create wide table with statistics for each parameter and step
3. Merge with OCD/THK measurements as targets
4. Implement time series cross-validation strategy
5. Build XGBoost/LightGBM models for OCD and THK prediction