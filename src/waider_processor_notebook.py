"""
Wafer Processor Integration for Jupyter Notebooks
================================================

Interactive notebook functions for wafer data processing and analysis.
This module provides helper functions for data exploration and feature engineering.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple
from wide_table.wafer_data_processor import WaferDataProcessor


class WaiderProcessorNotebook:
    """Notebook interface for wafer data processing with visualization capabilities."""

    def __init__(self, source_data_path: str = "E:/qingzhu/FAB/source_data"):
        """Initialize with data paths."""
        self.processor = WaferDataProcessor(source_data_path)
        self.df_wide = None
        self.df_features = None
        self.target_col = None

    def load_and_process(self, save_results: bool = True) -> pd.DataFrame:
        """
        Load and process all wafer data.

        Args:
            save_results: Whether to save results to disk

        Returns:
            Wide table DataFrame
        """
        print("Processing wafer data...")
        self.df_wide = self.processor.process_all_wafers()

        if save_results:
            self.processor.save_results()

        # Prepare feature summary
        self._prepare_feature_summary()

        return self.df_wide

    def _prepare_feature_summary(self):
        """Create feature summary statistics."""
        if self.df_wide is not None:
            self.df_features = pd.DataFrame({
                'feature': self.df_wide.columns,
                'dtype': self.df_wide.dtypes,
                'non_null_count': self.df_wide.count(),
                'null_pct': self.df_wide.isnull().sum() / len(self.df_wide) * 100,
                'unique_values': [self.df_wide[col].nunique() for col in self.df_wide.columns],
                'min_value': [self.df_wide[col].min() if self.df_wide[col].dtype in ['int64', 'float64'] else None
                             for col in self.df_wide.columns],
                'max_value': [self.df_wide[col].max() if self.df_wide[col].dtype in ['int64', 'float64'] else None
                             for col in self.df_wide.columns],
                'std_value': [self.df_wide[col].std() if self.df_wide[col].dtype in ['int64', 'float64'] else None
                             for col in self.df_wide.columns]
            })

    def get_parameter_statistics(self) -> pd.DataFrame:
        """Get statistics for parameter types."""
        if self.df_features is None:
            self._prepare_feature_summary()

        param_stats = self.df_features[
            self.df_features['feature'].str.contains(r'_mean$|_min$|_max$|_median$|_stdev$')
        ].copy()

        # Extract parameter type from column names
        param_stats['agg_type'] = param_stats['feature'].str.extract(r'_([a-z]+)$')[0]
        param_stats['parameter'] = param_stats['feature'].str.replace(r'_[0-9]+_[a-z]+$', '', regex=True)
        param_stats['step'] = param_stats['feature'].str.extract(r'_([0-9]+)_')[0].astype(int)

        return param_stats

    def visualize_parameter_distribution(self, parameter_pattern: str, steps: List[int] = None):
        """
        Visualize parameter distributions across steps.

        Args:
            parameter_pattern: Pattern to match parameter names
            steps: Specific steps to visualize (None for all)
        """
        param_cols = [col for col in self.df_wide.columns
                     if parameter_pattern in col and col.endswith('_mean')]

        if steps:
            param_cols = [col for col in param_cols
                         if any(f'_{step}_' in col for step in steps)]

        n_cols = min(4, len(param_cols))
        n_rows = (len(param_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, col in enumerate(param_cols):
            row = idx // n_cols
            col_idx = idx % n_cols

            # Create histogram
            self.df_wide[col].hist(ax=axes[row, col_idx], bins=20)
            axes[row, col_idx].set_title(col, fontsize=10)
            axes[row, col_idx].set_ylabel('Frequency')

            # Add statistics text
            mean_val = self.df_wide[col].mean()
            std_val = self.df_wide[col].std()
            axes[row, col_idx].axvline(mean_val, color='red', linestyle='--', alpha=0.7)

        # Hide empty subplots
        for idx in range(len(param_cols), n_rows * n_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            axes[row, col_idx].set_visible(False)

        plt.tight_layout()
        plt.show()

    def find_correlated_parameters(self, target_col: str,
                                  threshold: float = 0.7,
                                  agg_types: List[str] = ['mean', 'stdev']) -> pd.DataFrame:
        """
        Find parameters highly correlated with target variable.

        Args:
            target_col: Target column name (e.g., 'OCD_mean', 'THK_mean')
            threshold: Correlation threshold
            agg_types: Aggregation types to include

        Returns:
            DataFrame with correlation results
        """
        self.target_col = target_col

        # Get all parameter columns
        param_cols = [col for col in self.df_wide.columns
                     if any(f'_{agg}' in col for agg in agg_types)
                     and col != target_col]

        # Calculate correlations
        correlations = []
        for col in param_cols:
            corr = self.df_wide[col].corr(self.df_wide[target_col])
            if abs(corr) >= threshold:
                correlations.append({
                    'feature': col,
                    'correlation': corr,
                    'abs_correlation': abs(corr),
                    'parameter': col.split('_')[0],
                    'step': col.split('_')[1] if col.split('_')[1].isdigit() else None,
                    'agg_type': col.split('_')[-1] if col.split('_')[-1] in ['mean', 'min', 'max', 'median', 'stdev'] else 'mean'
                })

        corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)

        return corr_df

    def reduce_feature_dimension(self, agg_prefix: str = 'RF_Source',
                                 steps: List[int] = None) -> pd.DataFrame:
        """
        Create a reduced feature set for initial modeling.

        Args:
            agg_prefix: Parameter prefix to keep
            steps: Specific steps to include (None for all)

        Returns:
            Reduced DataFrame
        """
        # Keep metadata and target columns
        keep_cols = ['wafer_id']
        keep_cols.extend([col for col in self.df_wide.columns if 'OCD_' in col or 'THK_' in col])
        keep_cols.extend([col for col in self.df_wide.columns if 'context_id' in col])
        keep_cols.extend([col for col in self.df_wide.columns if 'lot_id' in col])
        keep_cols.extend([col for col in self.df_wide.columns if 'recipe_id' in col])
        keep_cols.extend([col for col in self.df_wide.columns if 'chamber_id' in col])

        # Add parameters matching prefix
        param_cols = [col for col in self.df_wide.columns
                     if agg_prefix in col and col.endswith('_mean')]

        if steps:
            param_cols = [col for col in param_cols
                         if any(f'_{step}_' in col for step in steps)]

        keep_cols.extend(param_cols)

        # Remove duplicates
        keep_cols = list(set(keep_cols))

        return self.df_wide[keep_cols].copy()

    def export_for_modeling(self, output_path: str = None,
                           target_var: str = 'OCD_mean',
                           reduce_features: bool = True,
                           feature_prefix: str = 'RF_Source') -> Tuple[pd.DataFrame, List[str]]:
        """
        Export data ready for machine learning modeling.

        Args:
            output_path: Path to save the training data
            target_var: Target variable name
            reduce_features: Whether to reduce feature set
            feature_prefix: Prefix for feature selection

        Returns:
            (DataFrame, feature_list) - Ready for model training
        """
        if reduce_features:
            df_export = self.reduce_feature_dimension(feature_prefix)
        else:
            df_export = self.df_wide.copy()

        # Separate target and features
        if target_var in df_export.columns:
            y = df_export[target_var]
            feature_cols = [col for col in df_export.columns if col not in [target_var, 'wafer_id']]
            X = df_export[feature_cols]

            # Create modeling dataset
            df_modeling = pd.concat([df_export[['wafer_id']], X, y], axis=1)

            if output_path:
                df_modeling.to_csv(output_path, index=False)
                print(f"Modeling data saved to: {output_path}")

            return df_modeling, feature_cols
        else:
            raise ValueError(f"Target variable '{target_var}' not found in data")