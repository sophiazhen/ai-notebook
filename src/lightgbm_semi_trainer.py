import lightgbm as lgb
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import logging
import warnings
warnings.filterwarnings('ignore')

from src.lightgbm_trainer import LightGBMTrainer

logger = logging.getLogger(__name__)


class LightGBMSemiconductorTrainer(LightGBMTrainer):
    """
    LightGBM trainer specifically optimized for semiconductor wafer data.

    Key optimizations:
    - Native handling of categorical features (tool_id, recipe_id)
    - Efficient processing of high-dimensional sensor data
    - Physics-based feature engineering
    - Support for both frequency and time-domain features
    """

    def __init__(
        self,
        model_name: str = 'lightgbm_semiconductor',
        task_type: str = 'regression',
        n_folds: int = 5,
        cv_strategy: str = 'kfold',
        random_state: int = 42,
        scaler_type: str = None,  # LightGBM handles scaling internally
        use_categorical: bool = True,
        use_quantile_transform: bool = False,  # LightGBM handles skewed data natively
        use_group_folding: bool = True,  # Important for wafer lots
        sample_for_training: bool = False,  # Sampling for very large datasets
        sample_ratio: float = 1.0
    ):
        """
        Initialize LightGBM trainer optimized for semiconductor data.

        Args:
            model_name: Model identifier
            task_type: 'regression' or 'classification'
            n_folds: CV folds
            cv_strategy: 'kfold', 'stratified', 'group'
            random_state: Random seed
            scaler_type: None (LGBM handles internally)
            use_categorical: Enable automatic categorical handling (default True)
            use_quantile_transform: Apply quantile transform (usually not needed)
            use_group_folding: Use group-based folding for wafers
            sample_for_training: Sample large datasets
            sample_ratio: Sampling ratio if sampling enabled
        """
        super().__init__(model_name, task_type, n_folds, cv_strategy, random_state, scaler_type, use_categorical)
        self.early_stopping_rounds = 200  # LightGBM can handle more iterations
        self.eval_metrics = {
            'regression': 'rmse',
            'classification': 'multiclass'
        }
        self.use_quantile_transform = use_quantile_transform
        self.use_group_folding = use_group_folding
        self.sample_for_training = sample_for_training
        self.sample_ratio = sample_ratio
        self.quantile_transformer = None
        self.categorical_features_ = None
        self.feature_groups_ = {}

    def _build_regression_model(self, params: Optional[Dict] = None) -> lgb.LGBMRegressor:
        """Build LightGBM regression model optimized for semiconductor data."""
        default_params = {
            # High capacity for large-scale wafer data
            'n_estimators': 1500,  # More trees for deep data
            'num_leaves': 63,  # More leaves for sensor data
            'max_depth': -1,  # No limit - let data decide
            'learning_rate': 0.02,  # Conservative learning rate

            # Light subsampling for regularization
            'subsample': 0.7,
            'subsample_freq': 1,
            'colsample_bytree': 0.6,  # Aggressive column sampling for high-dim data

            # Strong regularization for wafer data stability
            'reg_alpha': 0.2,
            'reg_lambda': 0.5,
            'min_child_samples': 10,  # Higher for sensor reliability
            'min_split_gain': 0.01,

            # Meta-parameters
            'objective': 'regression',
            'metric': self.eval_metrics['regression'],
            'boosting_type': 'gbdt',
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': -1
        }

        if params:
            default_params.update(params)

        return lgb.LGBMRegressor(**default_params)

    def _build_classification_model(self, params: Optional[Dict] = None) -> lgb.LGBMClassifier:
        """Build LightGBM classification model."""
        default_params = {
            'n_estimators': 1500,
            'num_leaves': 31,
            'learning_rate': 0.02,
            'max_depth': -1,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.2,
            'reg_lambda': 0.8,
            'min_child_samples': 20,
            'min_split_gain': 0.01,
            'objective': 'multiclass',
            'metric': self.eval_metrics['classification'],
            'boosting_type': 'gbdt',
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': -1
        }

        if params:
            default_params.update(params)

        return lgb.LGBMClassifier(**default_params)

    def build_model(self, params: Optional[Dict] = None):
        """Build LightGBM model with semiconductor optimizations."""
        # Set categorical feature handling
        if self.use_categorical:
            # Default behavior - can be overridden via params
            if params and 'categorical_feature' in params:
                self.categorical_features_ = params['categorical_feature']
            else:
                # Auto-detect categorical features
                self.categorical_features_ = []

        super().build_model(params)

    def preprocess_data(self, X: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Preprocess semiconductor data for LightGBM.

        LightGBM handles most transformations natively, but we can add
        domain-specific preprocessing.
        """
        X_processed = super().preprocess_data(X, fit_scaler)

        # Optional quantile transformation
        if self.use_quantile_transform:
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
            if self.quantile_transformer is None or fit_scaler:
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

    def _categorize_sensor_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """
        Categorize sensor features for semiconductor manufacturing.

        Args:
            feature_names: List of feature names

        Returns:
            Dict mapping categories to feature lists
        """
        categories = {
            'oes_spectral': [f for f in feature_names if f.startswith('OES_')],
            'power_rf': [f for f in feature_names if 'RF_Source_Power_' in f],
            'pressure_chamber': [f for f in feature_names if 'Chamber_Pressure_' in f],
            'oes_stability': [f for f in feature_names if f.startswith('OES_') and f.endswith('_stdev')],
            'power_stability': [f for f in feature_names if 'RF_Source_Power_' in f and 'stdev' in f],
            'pressure_stability': [f for f in feature_names if 'Chamber_Pressure_' in f and 'stdev' in f]
        }

        # Add categorical features (wafer lots, recipes, tools)
        categories['manufacturing_meta'] = [f for f in feature_names
                                          if any(meta in f.lower() for meta in ['tool_id', 'recipe_id', 'lot_id'])]

        return categories

    def add_semiconductor_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add physics-based derived features for semiconductor processes.

        Args:
            data: Input dataframe

        Returns:
            DataFrame with new features
        """
        data_enhanced = data.copy()

        # Feature 1: Plasma ion-to-neutral ratio (crucial for etch anisotropy)
        if any('OES_F_' in c for c in data.columns):
            f_mean_cols = [c for c in data.columns if 'OES_F_' in c and c.endswith('_mean')]
            cf_mean_cols = [c for c in data.columns if 'OES_CF_' in c and c.endswith('_mean')]

            if f_mean_cols and cf_mean_cols:
                data_enhanced['F_CF_ratio'] = data[f_mean_cols[:3]].mean(axis=1) / \
                                            (data[cf_mean_cols[:3]].mean(axis=1) + 1e-6)

        # Feature 2: Plasma density indicator (sum of OES intensities)
        oes_mean_cols = [c for c in data.columns if c.startswith('OES_') and c.endswith('_mean')]
        if len(oes_mean_cols) > 5:
            data_enhanced['total_plasma_intensity'] = data[oes_mean_cols[:20]].sum(axis=1)
            data_enhanced['median_plasma_intensity'] = data[oes_mean_cols[:50]].median(axis=1)

        # Feature 3: RF power system efficiency
        power_cols = [c for c in data.columns if 'RF_Source_Power_' in c and c.endswith('_mean')]
        if len(power_cols) > 5:
            data_enhanced['total_rf_power'] = data[power_cols].sum(axis=1)
            data_enhanced['rf_power_variance'] = data[power_cols].var(axis=1)

        # Feature 4: Chamber condition indicators
        pressure_cols = [c for c in data.columns if 'Chamber_Pressure_' in c and c.endswith('_mean')]
        if len(pressure_cols) > 3:
            data_enhanced['mean_chamber_pressure'] = data[pressure_cols].mean(axis=1)
            data_enhanced['pressure_uniformity'] = data[pressure_cols[:10]].std(axis=1) / \
                                                  (data[pressure_cols[:10]].mean(axis=1) + 1e-6)

        # Feature 5: Equipment aging indicators (relative to reference)
        # Create relative features to normalize across different tools
        if len(power_cols) > 3 and len(data_enhanced) > 20:
            for col in power_cols[:3]:  # First 3 power sources
                data_enhanced[f'{col}_relative'] = data_enhanced[col] / data_enhanced[power_cols].mean(axis=1)

        logger.info(f"Added {len(data_enhanced.columns) - len(data)} semiconductor-specific features")

        return data_enhanced

    def train_with_feature_engineering(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        categorical_features: List[str] = None,
        validation_split: float = 0.2,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Train model with automatic feature engineering for semiconductor data.

        Args:
            X: Feature matrix
            y: Target vector
            categorical_features: Explicit categorical feature names
            validation_split: Validation split ratio
            params: Model parameters

        Returns:
            Training results
        """
        logger.info("Starting LightGBM training with semiconductor feature engineering...")

        # Feature engineering
        logger.info("Adding semiconductor-derived features...")
        X_enhanced = self.add_semiconductor_features(X)
        logger.info(f"Feature matrix before: {X.shape}, after: {X_enhanced.shape}")

        # Identify categorical features
        if categorical_features is None:
            self.categorical_features_ = self._identify_categorical_features(X_enhanced)
        else:
            self.categorical_features_ = categorical_features

        logger.info(f"Detected categorical features: {self.categorical_features_}")

        # Handle high dimensions
        if len(X_enhanced.columns) > 1000 and self.sample_for_training:
            n_samples = int(len(X_enhanced) * self.sample_ratio)
            sample_idx = np.random.choice(len(X_enhanced), n_samples, replace=False)
            X_sampled = X_enhanced.iloc[sample_idx]
            y_sampled = y.iloc[sample_idx]
            logger.info(f"Sampling data: {len(X_enhanced)} -> {len(X_sampled)} samples")
        else:
            X_sampled = X_enhanced
            y_sampled = y

        # Split data
        if len(X_sampled) > 200:
            X_train, X_val, y_train, y_val = train_test_split(
                X_sampled, y_sampled, test_size=validation_split, random_state=self.random_state
            )
        else:
            X_train, X_val = X_sampled, X_sampled.iloc[-20:]
            y_train, y_val = y_sampled, y_sampled.iloc[-20:]

        # Store feature names
        self.feature_names = list(X_train.columns)

        # Build and train model
        self.build_model(params)

        logger.info("Training LightGBM model...")
        if self.categorical_features_:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_names=['val'],
                categorical_feature=self.categorical_features_,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False
            )
        else:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_names=['val'],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False
            )

        logger.info(f"Model trained with {getattr(self.model, 'best_iteration_', -1)} iterations")

        # Compute metrics
        train_pred = self.model.predict(X_train, num_iteration=self.model.best_iteration_)
        val_pred = self.model.predict(X_val, num_iteration=self.model.best_iteration_)

        if self.task_type == 'regression':
            import numpy as np
            train_rmse = np.sqrt(((y_train.values - train_pred) ** 2).mean())
            val_rmse = np.sqrt(((y_val.values - val_pred) ** 2).mean())

            from sklearn.metrics import mean_absolute_error, r2_score
            metrics = {
                'train_rmse': train_rmse,
                'train_mae': mean_absolute_error(y_train, train_pred),
                'train_r2': r2_score(y_train, train_pred),
                'val_rmse': val_rmse,
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_r2': r2_score(y_val, val_pred)
            }
        return metrics

    def _identify_categorical_features(self, data: pd.DataFrame) -> List[str]:
        """Automatically identify categorical features in semiconductor data."""
        categorical_cols = []

        # Common categorical patterns in semiconductor data
        categorical_patterns = ['tool_id', 'recipe_id', 'lot_id', 'wafer_id', 'chamber_id',
                                'operator_id', 'product_type', 'process_step', 'recipe']

        for col in data.columns:
            # Check if column name matches categorical patterns
            if any(pattern.lower() in col.lower() for pattern in categorical_patterns):
                categorical_cols.append(col)
            # Check low cardinality (arbitrary threshold)
            elif data[col].nunique() <= 10 and data[col].dtype == 'object':
                categorical_cols.append(col)
            # Check if it's integer categorized
            elif data[col].dtype == 'int64' and data[col].nunique() <= 20:
                categorical_cols.append(col)

        return categorical_cols

    def interpret_wafer_results(self, X_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Provide interpretation specific to wafer manufacturing processes.
        """
        interpretations = {
            'feature_analysis': {},
            'process_insights': {},
            'tool_recommendations': {}
        }

        # Feature category analysis
        categorical_features = self._categorize_sensor_features(X_test.columns.tolist())
        total_importance = 0

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            for category, features in categorical_features.items():
                if features:
                    indices = [X_test.columns.get_loc(f) for f in features if f in X_test.columns]
                    category_importance = sum(importances[i] for i in indices if i < len(importances))
                    interpretations['feature_analysis'][category] = category_importance
                    total_importance += category_importance

            # Normalize
            for cat in interpretations['feature_analysis']:
                interpretations['feature_analysis'][cat] /= total_importance

        # Process insights based on feature groups
        max_category = max(interpretations['feature_analysis'].keys(),
                          key=lambda x: interpretations['feature_analysis'][x])

        interpretations['process_insights'] = {
            'dominant_factor_group': max_category,
            'interpretation': f"{max_category.replace('_', ' ').title()} is the most critical factor"
        }

        # Tool recommendations (if tool_id data available)
        if 'tool_id' in X_test.columns or any('tool' in c.lower() for c in X_test.columns):
            interpretations['tool_recommendations']['categorical_support'] = True

        return interpretations

    def export_model_for_production(self, filename: str = "wafer_vm_lgbm_model.pkl"):
        """Export model with production-ready configuration."""
        import joblib

        model_package = {
            'model': self.model,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features_,
            'task_type': self.task_type,
            'best_iteration': getattr(self.model, 'best_iteration_', None),
            'semiconductor_specifics': {
                'model_type': 'lightgbm_semiconductor',
                'optimized_for': 'wafer_virtual_metrology',
                'features_engineered': True,
                'categorical_handling': self.categorical_features_ is not None
            }
        }

        if self.quantile_transformer is not None:
            model_package['quantile_transformer'] = self.quantile_transformer

        joblib.dump(model_package, filename)
        logger.info(f"Semiconductor model exported to {filename}")

        return filename


# Simplified factory for semiconductor applications
def create_lightgbm_semi_trainer(
    task_type: str = 'regression',
    target_metric: str = 'rmse',
    cv_folds: int = 5,
    random_state: int = 42,
    handle_categorical: bool = True
) -> LightGBMSemiconductorTrainer:
    """
    Create a LightGBM trainer optimized for semiconductor manufacturing data.

    This trainer includes:
    - Automatic handling of categorical features (tool_id, recipe_id, etc.)
    - Wafer-specific feature engineering
    - Production-ready model export
    """
    trainer = LightGBMSemiconductorTrainer(
        task_type=task_type,
        n_folds=cv_folds,
        random_state=random_state,
        scaler_type=None,  # Let LGBM handle scaling
        use_categorical=handle_categorical,
        use_quantile_transform=False,  # LGBM handles skewness
        use_group_folding=True  # Important for production lots
    )

    # Configure based on metric
    if task_type == 'regression':
        if target_metric == 'mae':
            trainer.eval_metrics['regression'] = 'mae'
        elif target_metric == 'mape':
            trainer.eval_metrics['regression'] = 'mape'

    return trainer