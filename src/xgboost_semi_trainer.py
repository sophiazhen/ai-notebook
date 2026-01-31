import xgboost as xgb
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.xgboost_trainer import XGBoostTrainer


class XGBoostSemiconductorTrainer(XGBoostTrainer):
    """
    XGBoost trainer specifically optimized for semiconductor wafer data.

    Key optimizations include:
    - Quantile transformation for skewed sensor data
    - Feature selection based on semiconductor domain knowledge
    - Handling of high-dimensional data with colsample optimization
    - Physics-based feature engineering
    """

    def __init__(
        self,
        model_name: str = 'xgboost_semiconductor',
        task_type: str = 'regression',
        n_folds: int = 5,
        cv_strategy: str = 'group',  # Default to lot-based CV for semiconductor
        random_state: int = 42,
        scaler_type: str = 'robust',
        use_quantile_transform: bool = True,
        use_feature_selection: bool = True,
        max_features_ratio: float = 0.7,  # Use 70% of features for high-dimensional data
        apply_semiconductor_engineering: bool = True
    ):
        """
        Initialize XGBoost trainer optimized for semiconductor data.

        Args:
            model_name: Model identifier
            task_type: 'regression' or 'classification'
            n_folds: Cross-validation folds
            cv_strategy: Default 'group' for lot-based CV
            random_state: Random seed
            scaler_type: Feature scaler
            use_quantile_transform: Apply quantile transform to handle skewness
            use_feature_selection: Use domain knowledge for feature selection
            max_features_ratio: Maximum ratio of features to use due to high dimensionality
            apply_semiconductor_engineering: Apply physics-based feature engineering
        """
        super().__init__(model_name, task_type, n_folds, cv_strategy, random_state, scaler_type)
        self.early_stopping_rounds = 150  # Longer early stopping for deep data
        self.use_quantile_transform = use_quantile_transform
        self.use_feature_selection = use_feature_selection
        self.max_features_ratio = max_features_ratio
        self.apply_semiconductor_engineering = apply_semiconductor_engineering
        self.feature_groups = {
            'oes_intensity': [],
            'oes_stability': [],
            'rf_power': [],
            'chamber_pressure': [],
            'derived_features': []
        }
        self.selected_features_ = None
        self.quantile_transformer = None

    def preprocess_data(self, X: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Preprocess semiconductor wafer data with specialized transformations.

        Args:
            X: Feature matrix
            fit_scaler: Whether to fit the transformer

        Returns:
            Preprocessed features
        """
        X_processed = super().preprocess_data(X, fit_scaler)

        # Apply quantile transformation for skewed sensor data
        if self.use_quantile_transform:
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
            if fit_scaler or self.quantile_transformer is None:
                self.quantile_transformer = QuantileTransformer(
                    n_quantiles=1000,
                    output_distribution='normal',
                    random_state=self.random_state
                )
                X_processed[numeric_cols] = self.quantile_transformer.fit_transform(
                    X_processed[numeric_cols]
                )
            else:
                X_processed[numeric_cols] = self.quantile_transformer.transform(
                    X_processed[numeric_cols]
                )

        return X_processed

    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[int]]:
        """
        Categorize features based on semiconductor domain knowledge.

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary mapping categories to feature indices
        """
        feature_indices = {name: i for i, name in enumerate(feature_names)}

        categories = {
            'plasma_dynamics': [],
            'power_control': [],
            'pressure_dynamics': [],
            'selectivity_indicators': [],
            'stability_measures': []
        }

        # Optical Emission Spectroscopy (OES) features
        oes_patterns = [
            'OES_', 'OES_f', 'OES_ar', 'OES_cf', 'OES_si', 'OES_o',
            'F_Peak_Intensity', 'AR_Intensity', 'CF_Peak'
        ]
        for pattern in oes_patterns:
            categories['plasma_dynamics'].extend([
                feature_indices[name] for name in feature_names
                if pattern.lower() in name.lower()
            ])

        # RF Power features
        power_patterns = ['RF_Source_Power', 'Power_']
        for pattern in power_patterns:
            categories['power_control'].extend([
                feature_indices[name] for name in feature_names
                if pattern.lower() in name.lower()
            ])

        # Chamber Pressure features
        pressure_patterns = ['Chamber_Pressure', 'Pressure_']
        for pattern in pressure_patterns:
            categories['pressure_dynamics'].extend([
                feature_indices[name] for name in feature_names
                if pattern.lower() in name.lower()
            ])

        # Selectivity indicators (intensity ratios)
        oes_mean_cols = [i for i, name in enumerate(feature_names)
                        if name.startswith('OES_') and name.endswith('_mean')]

        # Stability measures (standard deviations)
        stability_cols = [i for i, name in enumerate(feature_names)
                         if name.endswith('_stdev') or name.endswith('_std')]
        categories['stability_measures'] = stability_cols

        return categories

    def _apply_semiconductor_domain_filters(self, X: np.ndarray, categories: Dict[str, List[int]]) -> np.ndarray:
        """
        Apply domain-specific filters for semiconductor manufacturing.

        Args:
            X: Feature matrix
            categories: Feature categories

        Returns:
            Filtered feature matrix
        """
        # Filter 1: Remove unstable sensors (high std/volume_ratio_mean)
        stability_cols = categories.get('stability_measures', [])
        if len(stability_cols) > 0:
            stability_mask = np.mean(X[:, stability_cols], axis=1) < 2.0  # Threshold based on domain
            valid_indices = np.where(stability_mask)[0]
            if len(valid_indices) > 0.8 * len(X):  # Keep at least 80% of samples
                return X[valid_indices, :]

        return X

    def _add_semiconductor_derived_features(self, X: np.ndarray, categories: Dict[str, List[int]]) -> np.ndarray:
        """
        Add physics-based derived features for semiconductor processes.

        Args:
            X: Feature matrix
            categories: Feature categories

        Returns:
            Feature matrix with new derived features
        """
        derived_features = []

        # Feature 1: Total RF power (sum of all power measurements)
        power_cols = categories.get('power_control', [])
        if len(power_cols) > 5:
            total_power = np.sum(X[:, power_cols[:10]], axis=1, keepdims=True)
            derived_features.append(total_power)

        # Feature 2: Elite OES intensity (top quartile mean)
        oes_cols = categories.get('plasma_dynamics', [])
        if len(oes_cols) > 10:
            oes_data = X[:, oes_cols[:20]]  # Use first 20 OES features
            top quartile_idx = int(0.75 * oes_data.shape[1])
            top_oes = np.sort(oes_data, axis=1)[:, -top_quartile_idx:]
            elite_oes = np.mean(top_oes, axis=1, keepdims=True)
            derived_features.append(elite_oes)

        # Feature 3: Chamber uniformity (std/mean ratio)
        pressure_cols = categories.get('pressure_dynamics', [])
        if len(pressure_cols) > 5:
            pressure_data = X[:, pressure_cols[:8]]
            pressure_mean = np.mean(pressure_data, axis=1, keepdims=True)
            pressure_std = np.std(pressure_data, axis=1, keepdims=True)
            pressure_uniform = pressure_std / (pressure_mean + 1e-6)
            derived_features.append(pressure_uniform)

        # Feature 4: Process stability index (inverse of coefficient of variation)
        stability_cols = categories.get('stability_measures', [])
        if len(stability_cols) > 0:
            mean_cols = [c for c in categories['plasma_dynamics'] if self.feature_names[c].endswith('_mean')][:5]
            stdev_cols = [c for c in stability_cols if c < X.shape[1]][:5]

            if len(stdev_cols) > 0 and len(mean_cols) > 0:
                stability_idx = 1.0 / np.mean(X[:, stdev_cols] / (X[:, mean_cols] + 1e-6), axis=1, keepdims=True)
                derived_features.append(stability_idx)

        if derived_features:
            derived_matrix = np.hstack(derived_features)
            return np.hstack([X, derived_matrix])

        return X

    def select_features(self, X: np.ndarray, y: np.ndarray, method: str = 'auto') -> List[str]:
        """
        Select features optimized for semiconductor data.

        Args:
            X: Feature matrix
            y: Target variable
            method: Feature selection method ('auto', 'gain', 'stability', 'domain')

        Returns:
            List of selected feature names
        """
        if not self.use_feature_selection:
            return list(range(X.shape[1]))

        n_features = X.shape[1]
        max_features = int(n_features * self.max_features_ratio)

        if method == 'auto':
            # Method 1: Remove unstable features first
            stability_cols = [i for i, name in enumerate(self.feature_names)
                             if name.endswith('stdev')]
            stability_scores = np.mean(X[:, stability_cols], axis=0) if stability_cols else []

            # Method 2: Random feature sampling (good for high-dimensional semiconductor data)
            feature_pool = list(range(n_features))

            # Remove extreme outlier features
            feature_means = np.mean(X, axis=0)
            feature_stds = np.std(X, axis=0)
            outlier_mask = (feature_stds > 0.001 * np.median(feature_stds)) & \
                          (np.abs(feature_means) < 10 * np.median(np.abs(feature_means)))
            filtered_features = [i for i in feature_pool if outlier_mask[i]]

            # Sample diverse features to prevent overfitting
            selected_features = np.random.choice(
                filtered_features,
                size=min(max_features, len(filtered_features)),
                replace=False
            )

        else:
            selected_features = list(range(min(max_features, n_features)))

        self.selected_features_ = [self.feature_names[i] for i in selected_features]
        logger.info(f"Selected {len(selected_features)} out of {n_features} features")

        return selected_features

    def preprocess_data(self, X: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Preprocess with semiconductor-specific optimizations.

        Args:
            X: Feature matrix
            fit_scaler: Whether to fit the transformer

        Returns:
            Preprocessed features
        """
        X_processed = super().preprocess_data(X, fit_scaler)

        # Categorize features if names are available
        if hasattr(self, 'feature_names') and self.apply_semiconductor_engineering:
            categories = self._categorize_features(X_processed.columns.tolist())

            # Apply domain filters
            X_processed = self._apply_semiconductor_domain_filters(X_processed, categories)

            # Add derived features
            if fit_scaler and len(X_processed) > 100:
                X_processed = self._add_semiconductor_derived_features(X_processed, categories)

        # Handle extreme values typical in semiconductor data
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1, q3 = X_processed[col].quantile([0.01, 0.99])
            if not np.isnan(q1) and not np.isnan(q3):
                X_processed[col] = X_processed[col].clip(q1, q3)

        return X_processed

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Enhanced cross-validation with semiconductor-specific CV strategy.

        Args:
            X: Features
            y: Target
            groups: Group labels (Wafer lots)
            params: Model parameters

        Returns:
            CV results
        """
        logger.info(f"Starting {self.n_folds}-fold CV with semiconductor optimizations...")

        # Determine grouped CV based on wafer lots if not provided
        if groups is None and self.cv_strategy == 'group':
            if hasattr(self, '_wafer_lot_groups'):
                groups = self._wafer_lot_groups
            else:
                logger.warning("No groups provided for group-based CV. Creating synthetic groups.")
                groups = pd.Series(range(len(X))) // (len(X) // self.n_folds)

        return super().cross_validate(X, y, groups, params)

    def train_semiconductor_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Train model with semiconductor-specific pipeline.

        Args:
            X: Features
            y: Target
            validation_split: Validation split size
            params: Model parameters

        Returns:
            Training results
        """
        logger.info("Starting semiconductor-specific training pipeline...")

        # Feature selection
        if self.use_feature_selection:
            feature_indices = self.select_features(X.values, y.values)
            if len(feature_indices) < X.shape[1]:
                logger.info(f"Selected {len(feature_indices)} features out of {X.shape[1]}")
                # Keep only selected features
                selected_mask = [True if i in feature_indices else False for i in range(X.shape[1])]
                selected_mask_series = pd.Series(selected_mask, index=X.columns)
                X_selected = X.loc[:, selected_mask_series]
                self.feature_names = X_selected.columns.tolist()
                X = X_selected

        # Create validation set with lot awareness
        if len(X) > 500:
            # Use proper validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=self.random_state
            )
        else:
            # For small datasets, use last samples
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train with early stopping
        self.build_model(params)

        if hasattr(self.model, 'fit') and len(X_val) > 0:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False
            )

            logger.info(f"Model trained with {getattr(self.model, 'best_iteration', len(X))} iterations")

        # Return training metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        if self.task_type == 'regression':
            train_metrics = self.calculate_regression_metrics(y_train.values, train_pred)
            val_metrics = self.calculate_regression_metrics(y_val.values, val_pred)
        else:
            train_metrics = self.calculate_classification_metrics(y_train.values, train_pred)
            val_metrics = self.calculate_classification_metrics(y_val.values, val_pred)

        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_iteration': getattr(self.model, 'best_iteration', None)
        }

    def interpret_results(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Provide semiconductor-specific interpretation of results.

        Args:
            X: Feature matrix

        Returns:
            Interpretation dictionary
        """
        interpretations = {}

        if hasattr(self.model, 'feature_importances_'):
            importance_df = self.get_feature_importance()['importance']

            # top_oes features
            oes_features = [idx for idx, name in enumerate(self.feature_names)
                           if name.startswith('OES_')]
            if len(oes_features) > 0:
                oes_importance = importance_df.iloc[oes_features].sort_values(ascending=False)
                interpretations['top_oes_characteristics'] = oes_importance.head(5)

            # Power system analysis
            power_features = [idx for idx, name in enumerate(self.feature_names)
                            if 'RF_Source_Power_' in name]
            if len(power_features) > 0:
                power_importance = importance_df.iloc[power_features].sort_values(ascending=False)
                interpretations['power_system_relevance'] = power_importance.head(3)

            # Pressure system analysis
            pressure_features = [idx for idx, name in enumerate(self.feature_names)
                               if 'Chamber_Pressure_' in name]
            if len(pressure_features) > 0:
                pressure_importance = importance_df.iloc[pressure_features].sort_values(ascending=False)
                interpretations['pressure_system_relevance'] = pressure_importance.head(3)

        return interpretations

    def save_model_info(self) -> Dict[str, Any]:
        """Extended model info with semiconductor-specific attributes."""
        info = super().save_model_info()
        info.update({
            'use_quantile_transform': self.use_quantile_transform,
            'use_feature_selection': self.use_feature_selection,
            'max_features_ratio': self.max_features_ratio,
            'semiconductor_optimizations': True
        })
        return info


# Factory function optimized for semiconductor data processing
def create_xgboost_semi_trainer(
    task_type: str = 'regression',  # 'regression' or 'classification'
    target_metric: str = 'rmse',    # 'rmse', 'mae', 'r2' for regression; 'accuracy', 'f1', 'auc' for classification
    cv_folds: int = 5,
    random_state: int = 42,
    handle_high_dimensionality: bool = True
) -> XGBoostSemiconductorTrainer:
    """
    Create an XGBoost trainer specifically optimized for semiconductor wafer data.

    This configuration includes:
    - Quantile transformation for sensor data
    - Feature selection to handle high-dimensional data
    - Domain-specific feature engineering
    - Physics-aware interpretation
    """
    trainer = XGBoostSemiconductorTrainer(
        task_type=task_type,
        n_folds=cv_folds,
        random_state=random_state,
        scaler_type='robust',  # Robust to sensor outliers
        use_quantile_transform=True,  # Good for log-normal sensor data
        use_feature_selection=handle_high_dimensionality,
        max_features_ratio=0.8,  # Use 80% of features if high-dimensional
        apply_semiconductor_engineering=True
    )

    # Configure based on target metric
    if task_type == 'regression' and target_metric == 'mae':
        trainer.eval_metrics['regression'] = 'mae'
        trainer.early_stopping_rounds = 100
    elif task_type == 'classification' and target_metric == 'auc':
        trainer.eval_metrics['classification'] = 'auc'
        trainer.early_stopping_rounds = 80

    return trainer


# Optimization helper for visualization
def plot_semiconductor_analysis(trainer: XGBoostSemiconductorTrainer, data: pd.DataFrame):
    """Generate semiconductor-specific analysis plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Feature importance by category
        if hasattr(trainer, 'feature_groups'):
            # Implement category-based feature importance
            pass

        # 2. OES vs Power feature correlation
        oes_cols = [c for c in data.columns if c.startswith('OES_')]
        power_cols = [c for c in data.columns if 'RF_Source_Power_' in c]
        if oes_cols and power_cols:
            from sklearn.decomposition import PCA

            # PCA of OES features
            pca_oes = PCA(n_components=2, random_state=42)
            oes_pca = pca_oes.fit_transform(data[oes_cols[:50]])

            # PCA of Power features
            pca_power = PCA(n_components=2, random_state=42)
            power_pca = pca_power.fit_transform(data[power_cols[:50]])

            axes[0, 0].scatter(oes_pca[:, 0], power_pca[:, 0], alpha=0.6)
            axes[0, 0].set_xlabel('OES PC1')
            axes[0, 0].set_ylabel('Power PC1')
            axes[0, 0].set_title('Plasma Dynamics vs Power System')

        # 3. Process stability analysis
        stability_features = [c for c in data.columns if 'stdev' in c]
        if stability_features:
            stability_by_step = {}
            for feature in stability_features[:10]:
                if any(step in feature for step in ['breakthrough', 'main_etch', 'overetch']):
                    stability_by_step[feature] = data[feature].mean()

            if stability_by_step:
                steps = list(stability_by_step.keys())
                values = list(stability_by_step.values())
                axes[0, 1].bar(range(len(steps)), values)
                axes[0, 1].set_xticks(range(len(steps)))
                axes[0, 1].set_xticklabels([s.split('_')[1] if '_' in s else s for s in steps], rotation=45)
                axes[0, 1].set_ylabel('Average Stability')
                axes[0, 1].set_title('Process Stability by Step')

        # 4. RF power distribution analysis
        power_features = [c for c in data.columns if 'RF_Source_Power_' in c]
        if power_features:
            power_data = data[power_features[:5]]
            power_data.boxplot(ax=axes[1, 0])
            axes[1, 0].set_xlabel('Power Source')
            axes[1, 0].set_ylabel('Power (W)')
            axes[1, 0].set_title('RF Power Distribution')

        plt.tight_layout()
        plt.savefig("results/semiconductor_analysis.png", dpi=300)
        plt.show()

    except ImportError:
        print("Matplotlib not available. Plots not generated.")

        print("Semiconductor analysis plots generated.")