"""
Wafer Data Processor
====================

Processes etching measurement data from Excel files:
1. Extract OCD and THK values from etching-measurement.xlsx
2. Merge with sampling-wafer-list data
3. Extract metadata from wafer files
4. Aggregate parameter sheets by step (mean, min, max, median, std)
5. Create wide table with all features

Naming convention for parameters: {parameter_name}_{step}_{aggregation_type}
Example: RF_Source_Power_1_stdev
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class WaferDataProcessor:
    """Processes wafer data from Excel files for ML model training."""

    def __init__(self, source_data_path: str = "E:/qingzhu/FAB/source_data"):
        """
        Initialize the processor.

        Args:
            source_data_path: Path to source data directory
        """
        self.source_path = Path(source_data_path)
        self.df_measurement = None
        self.df_wafer_list = None
        self.df_metadata = None
        self.df_parameters = None

    def load_etching_measurement(self) -> pd.DataFrame:
        """
        Load etching measurement data and extract OCD/THK values.

        Returns:
            DataFrame with wafer_id, indicator, mean, max, min, stdev
        """
        print("Loading etching measurement data...")

        file_path = self.source_path / "etching-measurement.xlsx"
        df = pd.read_excel(file_path, header=None)

        # Extract data from the specific format
        wafer_data = []

        for i in range(0, len(df), 2):
            # Get wafer ID
            wafer_id = df.iloc[i, 2]

            # Get OCD values (same row)
            ocd_mean = df.iloc[i, 4]
            ocd_max = df.iloc[i, 5]
            ocd_min = df.iloc[i, 6]
            ocd_stdev = df.iloc[i, 7] if i+1 < len(df) else None

            wafer_data.append({
                'wafer_id': wafer_id,
                'indicator': 'OCD',
                'mean': ocd_mean,
                'max': ocd_max,
                'min': ocd_min,
                'stdev': ocd_stdev,
                'unit': df.iloc[i, 8] if pd.notnull(df.iloc[i, 8]) else None
            })

            # Get THK values (next row if exists)
            if i+1 < len(df) and pd.notnull(df.iloc[i+1, 2]):
                thk_row = df.iloc[i+1]
                wafer_data.append({
                    'wafer_id': thk_row[2],
                    'indicator': 'THK',
                    'mean': thk_row[4],
                    'max': thk_row[5],
                    'min': thk_row[6],
                    'stdev': thk_row[7],
                    'unit': thk_row[8] if pd.notnull(thk_row[8]) else None
                })

        self.df_measurement = pd.DataFrame(wafer_data)
        print(f"Loaded {len(wafer_data)} measurement records for {self.df_measurement['wafer_id'].nunique()} wafers")

        return self.df_measurement

    def load_wafer_list(self) -> pd.DataFrame:
        """
        Load sampling wafer list data.

        Returns:
            DataFrame with wafer metadata
        """
        print("Loading wafer list data...")

        file_path = self.source_path / "sampling-wafer-list.xlsx"
        self.df_wafer_list = pd.read_excel(file_path)

        print(f"Loaded wafer list with {len(self.df_wafer_list)} wafers")

        return self.df_wafer_list

    def find_wafer_files(self) -> Dict[str, Path]:
        """
        Find wafer files based on wafer_id.

        Returns:
            Dictionary mapping wafer_id to file path
        """
        print("Finding wafer files...")

        # Look for WFR_300MM files
        pattern = r"WFR_300MM_<arg0=>c-sch-d(.+?).xlsx"
        wafer_files = {}

        for file in self.source_path.glob("WFR_300MM_*.xlsx"):
            # Extract wafer identifier from filename
            match = re.search(r"L\d+", file.name)
            if match:
                # Map to wafer_id from our list
                for wafer_id in self.df_measurement['wafer_id'].unique():
                    if wafer_id not in wafer_files:
                        wafer_files[wafer_id] = file
                        break

        print(f"Found {len(wafer_files)} wafer files")

        return wafer_files

    def extract_metadata(self, wafer_file: Path) -> Dict:
        """
        Extract metadata from a wafer file.

        Args:
            wafer_file: Path to the wafer Excel file

        Returns:
            Dictionary of metadata
        """
        try:
            # Read metadata sheet with header=0 to get proper column names
            df_meta = pd.read_excel(wafer_file, sheet_name='metadata')

            # Convert to dict - take first row as the metadata
            if len(df_meta) > 0:
                metadata = df_meta.iloc[0].to_dict()

                # Standardize key names
                standard_keys = {
                    'wafer_id': 'wafer_id',
                    'context_id': 'context_id',
                    'shift': 'shift',
                    'maintenance_flag': 'maintenance_flag'
                }

                # Clean up the metadata dictionary
                clean_metadata = {}
                for key, value in metadata.items():
                    if pd.notnull(value):
                        clean_key = str(key).strip()
                        clean_value = str(value).strip()
                        clean_metadata[clean_key] = clean_value

                return clean_metadata
            else:
                return {}

        except Exception as e:
            print(f"Error extracting metadata from {wafer_file}: {e}")
            # Try reading without header
            try:
                df_meta = pd.read_excel(wafer_file, sheet_name='metadata', header=None)
                if df_meta.shape[1] >= 2:
                    # Take the first row values
                    metadata = {
                        'wafer_id': str(df_meta.iloc[0, 1]) if df_meta.shape[1] > 1 else None,
                    }
                    # Add other columns if they exist
                    for i in range(min(10, df_meta.shape[1])):
                        if i != 0:
                            metadata[f'field_{i}'] = str(df_meta.iloc[0, i]) if i < df_meta.shape[1] else None
                    return metadata
            except Exception as e2:
                print(f"Second attempt failed: {e2}")
            return {}

    def extract_parameter_data(self, wafer_file: Path) -> pd.DataFrame:
        """
        Extract and aggregate parameter data by step.

        Args:
            wafer_file: Path to the wafer Excel file

        Returns:
            DataFrame with aggregated parameters
        """
        try:
            # First, get list of actual data sheets (format: sheet-x-parameter_name)
            excel_file = pd.ExcelFile(wafer_file)
            data_sheets = []

            for sheet_name in excel_file.sheet_names:
                if re.match(r'^sheet-\d+-', sheet_name):
                    data_sheets.append(sheet_name)

            print(f"Found {len(data_sheets)} data sheets in {wafer_file.name}")

            # Process each data sheet
            all_param_data = []

            for sheet_name in data_sheets:
                # Extract parameter name from sheet name
                # Format: sheet-x-parameter_name (e.g., sheet-1-RF Source Power)
                match = re.search(r'^sheet-(\d+)-(.+)', sheet_name)
                if not match:
                    continue

                sheet_number = match.group(1)
                param_name = match.group(2)

                try:
                    # Read parameter data
                    df_param = pd.read_excel(wafer_file, sheet_name=sheet_name)

                    # Check expected columns
                    expected_cols = ['Parameter', 'No.', 'Step', 'Value', 'Timestamp']
                    if not all(col in df_param.columns for col in ['Step', 'Value']):
                        # Try to find similar columns
                        step_col = None
                        value_col = None
                        for col in df_param.columns:
                            if 'step' in str(col).lower():
                                step_col = col
                            elif 'value' in str(col).lower():
                                value_col = col

                        if step_col and value_col:
                            df_param = df_param.rename(columns={step_col: 'Step', value_col: 'Value'})
                        else:
                            print(f"  Skipping {sheet_name}: required columns not found")
                            continue

                    # Group by step (excluding step 0)
                    for step, group in df_param.groupby('Step'):
                        if step == 0:
                            continue  # Skip step 0 as requested

                        # Convert values to numeric
                        if 'Value' in group.columns:
                            values = pd.to_numeric(group['Value'], errors='coerce').dropna()

                            if len(values) > 0:
                                agg_data = {
                                    'parameter': param_name,
                                    'step': step,
                                    'mean': values.mean(),
                                    'min': values.min(),
                                    'max': values.max(),
                                    'median': values.median(),
                                    'stdev': values.std()
                                }
                                all_param_data.append(agg_data)

                except Exception as e:
                    print(f"Error processing sheet {sheet_name}: {e}")
                    continue

            return pd.DataFrame(all_param_data)

        except Exception as e:
            print(f"Error extracting parameter data from {wafer_file}: {e}")
            return pd.DataFrame()

    def process_all_wafers(self) -> pd.DataFrame:
        """
        Process all wafer files and create wide table.

        Returns:
            Wide table DataFrame
        """
        print("\nProcessing all wafer data...")

        # Load initial data
        self.load_etching_measurement()
        self.load_wafer_list()

        # Get list of wafers from measurement data
        unique_wafers = self.df_measurement['wafer_id'].unique()

        # Find wafer files
        wafer_files = self.find_wafer_files()

        # Process each wafer
        all_results = []

        for wafer_id in unique_wafers:
            print(f"\nProcessing wafer: {wafer_id}")

            # Get wafer file (assign files based on order if exact matching fails)
            wafer_file = None
            if wafer_id in wafer_files:
                wafer_file = wafer_files[wafer_id]
            else:
                # Assign available file based on index
                available_files = list(wafer_files.values())
                if len(available_files) > 0 and all_results == []:
                    wafer_file = available_files[0]
                elif len(available_files) > 1 and len(all_results) <= len(available_files):
                    wafer_file = available_files[len(all_results)]

            if wafer_file:
                # Extract metadata
                metadata = self.extract_metadata(wafer_file)
                metadata['wafer_id'] = wafer_id

                # Extract parameter data
                param_df = self.extract_parameter_data(wafer_file)

                # Create wide format
                param_wide = {}
                for _, row in param_df.iterrows():
                    base_name = str(row['parameter']).replace(' ', '_').replace('-', '_')
                    step = int(row['step'])

                    param_wide[f"{base_name}_{step}_mean"] = row['mean']
                    param_wide[f"{base_name}_{step}_min"] = row['min']
                    param_wide[f"{base_name}_{step}_max"] = row['max']
                    param_wide[f"{base_name}_{step}_median"] = row['median']
                    param_wide[f"{base_name}_{step}_stdev"] = row['stdev']

                # Combine all data
                result = {**metadata, **param_wide}
                all_results.append(result)

        # Create final DataFrame
        df_wide = pd.DataFrame(all_results)

        # Merge with measurement and wafer list data
        # First, pivot measurement data
        measurement_pivot = self.df_measurement.pivot_table(
            index='wafer_id',
            columns='indicator',
            values=['mean', 'max', 'min', 'stdev'],
            aggfunc='first'
        )
        measurement_pivot.columns = [f"{col[1]}_{col[0]}" for col in measurement_pivot.columns]
        measurement_pivot.reset_index(inplace=True)

        # Merge all data
        df_final = df_wide.merge(measurement_pivot, on='wafer_id', how='inner')

        if self.df_wafer_list is not None:
            df_final = df_final.merge(self.df_wafer_list, on='wafer_id', how='left')

        self.df_final = df_final

        print(f"\nProcessed {len(df_final)} wafers with {len(df_final.columns)} features")

        return df_final

    def save_results(self, output_path: str = None, prefix: str = "processed"):
        """
        Save processing results to files.

        Args:
            output_path: Directory to save results (default: current directory)
            prefix: Prefix for output files
        """
        if output_path is None:
            output_path = Path.cwd() / "data_output"
        else:
            output_path = Path(output_path)

        output_path.mkdir(exist_ok=True)

        # Save wide table
        if self.df_final is not None:
            wide_table_path = output_path / f"{prefix}_wide_table.xlsx"
            self.df_final.to_excel(wide_table_path, index=False)
            print(f"Saved wide table to: {wide_table_path}")

            # Save CSV version for large datasets
            wide_table_csv = output_path / f"{prefix}_wide_table.csv"
            self.df_final.to_csv(wide_table_csv, index=False)
            print(f"Saved CSV version to: {wide_table_csv}")

        # Save intermediate dataframes
        if self.df_measurement is not None:
            measurement_path = output_path / f"{prefix}_measurement_data.xlsx"
            self.df_measurement.to_excel(measurement_path, index=False)

        # Save feature summary
        if self.df_final is not None:
            feature_summary = pd.DataFrame({
                'feature': self.df_final.columns,
                'dtype': [str(dtype) for dtype in self.df_final.dtypes],
                'non_null_count': self.df_final.count(),
                'null_count': self.df_final.isnull().sum(),
                'unique_values': [self.df_final[col].nunique() for col in self.df_final.columns]
            })

            summary_path = output_path / f"{prefix}_feature_summary.xlsx"
            feature_summary.to_excel(summary_path, index=False)
            print(f"Saved feature summary to: {summary_path}")

        print(f"\nAll results saved to: {output_path}")


def main():
    """Main function to run the wafer data processing."""
    print("Starting wafer data processing...")

    # Initialize processor
    processor = WaferDataProcessor(source_data_path="E:/qingzhu/FAB/source_data")

    # Process all data
    df_result = processor.process_all_wafers()

    # Display summary
    print("\n=== Processing Summary ===")
    print(f"Total wafers processed: {len(df_result)}")
    print(f"Total features created: {len(df_result.columns)}")
    print(f"OCD features: {len([col for col in df_result.columns if 'OCD' in col])}")
    print(f"THK features: {len([col for col in df_result.columns if 'THK' in col])}")
    print(f"Parameter features: {len([col for col in df_result.columns if any(agg in col for agg in ['_mean', '_min', '_max', '_median', '_stdev'])])}")

    # Save results
    processor.save_results()

    return processor, df_result


if __name__ == "__main__":
    processor, df_result = main()