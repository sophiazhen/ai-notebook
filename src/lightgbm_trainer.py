import lightgbm as lgb
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.base_ml_trainer import BaseMLTrainer


class LightGBMTrainer(BaseMLTrainer):
    """
    LightGBM model trainer for both regression and classification tasks.
    Optimized for high-dimensional semiconductor wafer data processing.
    """

    def __init__(
        self,
        model_name: str = 'lightgbm',
        task_type: str = 'regression',
        n_folds: int = 5,
        cv_strategy: str = 'kfold',
        random_state: int = 42,
        scaler_type: str = 'robust',
        use_categorical: bool = True
    ):
        """
        Initialize LightGBM trainer.

        Args:
            model_name: Name for the model (default: 'lightgbm')
            task_type: 'regression' or 'classification'
            n_folds: Number of cross-validation folds
            cv_strategy: 'kfold', 'stratified', 'group'
            random_state: Random seed for reproducibility
            scaler_type: Type of scaler to use (lightweight feature scaling for unnecessary features)
            use_categorical: Whether to handle categorical features automatically
        """
        super().__init__(model_name, task_type, n_folds, cv_strategy, random_state, scaler_type)
        self.use_categorical = use_categorical
        self.early_stopping_rounds = 50
        self.categorical_features = None
        self.eval_metrics = {
            'regression': 'rmse',
            'classification': 'multiclass' if task_type == 'classification' else 'binary'
        }

    def _build_regression_model(self, params: Optional[Dict] = None) -> lgb.LGBMRegressor:
        """Build LightGBM regression model."""
        default_params = {
            'n_estimators': 1000,
            'max_depth': -1,  # No limit, let leaves control complexity
            'num_leaves': 31,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'min_split_gain': 0,
            'subsample_freq': 1,
            'objective': 'regression',
            'metric': self.eval_metrics['regression'],
            'boosting_type': 'gbdt',
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': -1
        }

        if params:
            default_params.update(params)
        else:
            # Additional optimizations for semiconductor data
            default_params.update({
                'feature_fraction': 0.8,
                'bagging_fraction': 0.9,
                'bagging_freq': 1,
                'lambda_l1': 0.1,
                'lambda_l2': 0.2
            })

        return lgb.LGBMRegressor(**default_params)

    def _build_classification_model(self, params: Optional[Dict] = None) -> lgb.LGBMClassifier:
        """Build LightGBM classification model."""
        default_params = {
            'n_estimators': 1000,
            'max_depth': -1,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'min_split_gain': 0,
            'subsample_freq': 1,
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
        """Build LightGBM model based on task type."""
        if self.task_type == 'regression':
            self.model = self._build_regression_model(params)
        else:
            self.model = self._build_classification_model(params)
            self.n_classes = len(np.unique(params.get('classes', []))) if 'classes' in params else None

    def set_categorical_features(self, categorical_features: list):
        """Set categorical feature indices for LightGBM to handle automatically."""
        self.categorical_features = categorical_features

    def preprocess_data(self, X: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Preprocess data with LightGBM-specific optimizations.
        LightGBM handles categorical features natively, so minimal preprocessing needed.
        """
        X_processed = X.copy()

        # LightGBM handles missing values natively, no need to impute
        # LightGBM can use categorical features directly, just mark them
        if self.categorical_features and self.use_categorical:
            # Convert categorical columns to 'category' dtype
            for col in self.categorical_features:
                if col in X_processed.columns and X_processed[col].dtype == 'object':
                    X_processed[col] = X_processed[col].astype('category')

        # Light scaling for specific features if needed (not full scaling)
        if self.scaler is not None:
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
            if fit_scaler:
                self.scaler.fit(X_processed[numeric_cols])
            X_processed[numeric_cols] = self.scaler.transform(X_processed[numeric_cols])

        return X_processed

    def _train_model(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """Train the LightGBM model."""
        categorical_feature_indices = None
        if self.categorical_features is not None and self.use_categorical:
            categorical_feature_indices = [
                i for i, col in enumerate(X_train.columns)
                if col in self.categorical_features
            ]

        self.model.fit(
            X_train, y_train,
            categorical_feature=categorical_feature_indices,
            verbose=False
        )

    def fit_with_early_stopping(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Optional[Dict] = None,
        eval_names: list = None
    ):
        """
        Train model with early stopping using validation set.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            params: Model parameters
            eval_names: Names for evaluation sets
        """
        self.build_model(params)

        categorical_feature_indices = None
        if self.categorical_features is not None and self.use_categorical:
            categorical_feature_indices = [
                i for i, col in enumerate(X_train.columns)
                if col in self.categorical_features
            ]

        eval_set = [(X_val, y_val)]
        eval_names = eval_names or ['val']

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            categorical_feature=categorical_feature_indices,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=False
        )

        print(f"Early stopping at iteration {self.model.best_iteration_}")

    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance from trained LightGBM model.

        Args:
            importance_type: Type of importance ('gain', 'split', 'gain_per_folds', 'split_per_folds')

        Returns:
            DataFrame with features and their importance scores
        """
        if self.feature_names is None:
            raise ValueError("Model must be trained first")

        importance_scores = self.model.feature_importances_

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            f'importance_{importance_type}': importance_scores
        }).sort_values(f'importance_{importance_type}', ascending=False)

        return importance_df

    def plot_feature_importance_split_gain(self, top_n: int = 20):
        """Plot both split and gain importance."""
        booster = self.model.booster_

        # Get both importance types
        gain_importance = pd.DataFrame({
            'feature': self.feature_names,
            'gain': booster.feature_importance(importance_type='gain')
        }).sort_values('gain', ascending=False).head(top_n)

        split_importance = pd.DataFrame({
            'feature': self.feature_names,
            'split': booster.feature_importance(importance_type='split')
        }).sort_values('split', ascending=False).head(top_n)

        # Merge and plot
        importance_df = gain_importance.merge(
            split_importance, on='feature'
        ).sort_values('gain', ascending=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

        # Plot gain
        ax1.barh(importance_df['feature'], importance_df['gain'])
        ax1.set_xlabel('Gain')
        ax1.set_title(f'Top {top_n} Features by Gain')
        ax1.invert_yaxis()

        # Plot split
        ax2.barh(importance_df['feature'], importance_df['split'])
        ax2.set_xlabel('Split')
        ax2.set_title(f'Top {top_n} Features by Split Count')
        ax2.invert_yaxis()

        plt.tight_layout()
        plt.savefig(f"results/{self.model_name}_feature_importance_split_gain.png", dpi=300)
        plt.show()

        return importance_df

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict = None,
        n_trials: int = 50,
        scoring: str = None
    ) -> Dict:
        """
        Perform hyperparameter tuning using Optuna (LightGBM-optimized).

        Args:
            X: Features
            y: Target
            param_space: Search space for hyperparameters
            n_trials: Number of optimization trials
            scoring: Scoring metric

        Returns:
            Best parameters found
        """
        try:
            import optuna
            from sklearn.model_selection import cross_val_score

            if param_space is None:
                # LightGBM-optimized search space
                if self.task_type == 'regression':
                    param_space = {
                        'num_leaves': (20, 100),
                        'max_depth': (-1, 12),
                        'learning_rate': (0.01, 0.3),
                        'n_estimators': (100, 1500),
                        'min_child_samples': (10, 100),
                        'reg_alpha': (0, 1),
                        'reg_lambda': (0, 1),
                        'feature_fraction': (0.4, 1.0),
                        'bagging_fraction': (0.4, 1.0),
                        'bagging_freq': (1, 10),
                        'min_split_gain': (0, 0.5)
                    }
                else:  # classification
                    param_space = {
                        'num_leaves': (20, 100),
                        'max_depth': (-1, 12),
                        'learning_rate': (0.01, 0.3),
                        'n_estimators': (100, 1500),
                        'min_child_samples': (10, 100),
                        'reg_alpha': (0, 1),
                        'reg_lambda': (0, 1),
                        'feature_fraction': (0.4, 1.0),
                        'bagging_fraction': (0.4, 1.0),
                        'bagging_freq': (1, 10),
                        'min_split_gain': (0, 0.5)
                    }

            def objective(trial):
                params = {}
                for param, (low, high) in param_space.items():
                    if param == 'max_depth' and self.task_type == 'regression':
                        params[param] = trial.suggest_int(param, low, high)
                    elif param in ['n_estimators', 'num_leaves', 'min_child_samples', 'bagging_freq']:
                        params[param] = trial.suggest_int(param, low, high)
                    elif param in ['learning_rate']:
                        params[param] = trial.suggest_loguniform(param, low, high)
                    else:
                        params[param] = trial.suggest_uniform(param, low, high)

                # LightGBM-specific: Auto-handling of categorical features
                if self.categorical_features:
                    params['categorical_feature'] = self.categorical_features

                # Build model
                self.build_model(params)

                # Cross-validation
                X_processed = self.preprocess_data(X, fit_scaler=True)

                if scoring is None:
                    scoring = 'neg_root_mean_squared_error' if self.task_type == 'regression' else 'accuracy'

                scores = cross_val_score(
                    self.model, X_processed, y,
                    cv=5, scoring=scoring,
                    n_jobs=-1
                )

                return scores.mean()

            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.random_state)
            )
            study.optimize(objective, n_trials=n_trials, n_jobs=1)

            print(f"Best parameters: {study.best_params}")
            print(f"Best score: {study.best_value:.4f}")

            self.best_params = study.best_params
            return study.best_params

        except ImportError:
            print("Optuna not available. Please install with: pip install optuna")
            return None

    def create_learning_curve(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Optional[Dict] = None,
        train_sizes: list = None
    ):
        """
        Create learning curve for LightGBM model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            params: Model parameters
            train_sizes: Training set proportions
        """
        if train_sizes is None:
            train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

        train_errors = []
        val_errors = []

        for size in train_sizes:
            n_train = int(len(X_train) * size)
            indices = np.random.choice(len(X_train), n_train, replace=False)
            X_sub = X_train.iloc[indices]
            y_sub = y_train.iloc[indices]

            # Build and train
            self.build_model(params)
            self._train_model(X_sub, y_sub)

            # Evaluate
            train_pred = self._predict_model(X_sub)
            val_pred = self._predict_model(X_val)

            if self.task_type == 'regression':
                train_rmse = np.sqrt(((y_sub - train_pred) ** 2).mean())
                val_rmse = np.sqrt(((y_val - val_pred) ** 2).mean())
                train_errors.append(train_rmse)
                val_errors.append(val_rmse)
            else:
                train_acc = (y_sub == train_pred).mean()
                val_acc = (y_val == val_pred).mean()
                train_errors.append(1 - train_acc)
                val_errors.append(1 - val_acc)

        return train_sizes, train_errors, val_errors

    def compare_with_baseline(self, X_train, y_train, X_val, y_val) -> Dict[str, float]:
        """Compare LightGBM performance with simple baseline models."""
        from sklearn.dummy import DummyRegressor, DummyClassifier
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

        comparison = {}

        # LightGBM performance
        self.build_model()
        self.fit_with_early_stopping(X_train, y_train, X_val, y_val)
        pred_val = self._predict_model(X_val)

        if self.task_type == 'regression':
            comparison['lgb_rmse'] = np.sqrt(((y_val - pred_val) ** 2).mean())
            comparison['lgb_r2'] = self._r2_score(y_val, pred_val)
        else:
            comparison['lgb_accuracy'] = (y_val == pred_val).mean()

        # Baseline 1: Dummy predictor
        if self.task_type == 'regression':
            dummy = DummyRegressor(strategy='mean')
            dummy.fit(X_train, y_train)
            dummy_pred = dummy.predict(X_val)
            comparison['dummy_rmse'] = np.sqrt(((y_val - dummy_pred) ** 2).mean())
            comparison['dummy_r2'] = self._r2_score(y_val, dummy_pred)
        else:
            dummy = DummyClassifier(strategy='stratified')
            dummy.fit(X_train, y_train)
            dummy_pred = dummy.predict(X_val)
            comparison['dummy_accuracy'] = (y_val == dummy_pred).mean()

        print("Performance comparison:")
        for key, value in comparison.items():
            print(f"{key}: {value:.4f}")

        return comparison

    def save_model_info(self) -> Dict[str, Any]:
        """Get LightGBM model information for saving."""
        info = {
            'model_type': 'lightgbm',
            'task_type': self.task_type,
            'best_iteration': getattr(self.model, 'best_iteration_', None),
            'best_score': getattr(self.model, 'best_score_', None),
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'categorical_features': self.categorical_features,
            'objective': getattr(self.model, 'objective', None)
        }

        return info


# Factory function for creating LightGBM trainer instances
def create_lightgbm_trainer(
    task_type: str = 'regression',
    model_name: str = 'lightgbm',
    target_metric: str = 'rmse',
    cv_folds: int = 5,
    random_state: int = 42,
    use_categorical: bool = True
) -> LightGBMTrainer:
    """
    Create a LightGBM trainer with optimal parameters based on the task.

    Args:
        task_type: 'regression' or 'classification'
        model_name: Name for the trainer
        target_metric: Target metric for optimization
        cv_folds: Number of CV folds
        random_state: Random seed
        use_categorical: Whether to handle categorical features

    Returns:
        Configured LightGBM trainer
    """
    trainer = LightGBMTrainer(
        model_name=model_name,
        task_type=task_type,
        n_folds=cv_folds,
        random_state=random_state,
        scaler_type='robust' if task_type == 'regression' else None,
        use_categorical=use_categorical
    )

    if task_type == 'regression':
        if target_metric == 'mae':
            trainer.eval_metrics['regression'] = 'mae'
        elif target_metric == 'mape':
            trainer.eval_metrics['regression'] = 'mape'
    else:  # classification
        if target_metric == 'auc':
            trainer.eval_metrics['classification'] = 'auc'

    return trainer