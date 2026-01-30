import xgboost as xgb
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.base_ml_trainer import BaseMLTrainer


class XGBoostTrainer(BaseMLTrainer):
    """
    XGBoost model trainer for both regression and classification tasks.
    Optimized for semiconductor wafer data processing with built-in feature selection.
    """

    def __init__(
        self,
        model_name: str = 'xgboost',
        task_type: str = 'regression',
        n_folds: int = 5,
        cv_strategy: str = 'kfold',
        random_state: int = 42,
        scaler_type: str = 'robust'
    ):
        """
        Initialize XGBoost trainer.

        Args:
            model_name: Name for the model (default: 'xgboost')
            task_type: 'regression' or 'classification'
            n_folds: Number of cross-validation folds
            cv_strategy: 'kfold', 'stratified', 'group'
            random_state: Random seed for reproducibility
            scaler_type: Type of scaler to use
        """
        super().__init__(model_name, task_type, n_folds, cv_strategy, random_state, scaler_type)
        self.early_stopping_rounds = 50
        self.eval_metrics = {
            'regression': 'rmse',
            'classification': 'logloss'
        }

    def _build_regression_model(self, params: Optional[Dict] = None) -> xgb.XGBRegressor:
        """Build XGBoost regression model."""
        default_params = {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'colsample_bynode': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 1,
            'gamma': 0,
            'objective': 'reg:squarederror',
            'eval_metric': self.eval_metrics['regression'],
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0
        }

        if params:
            default_params.update(params)

        return xgb.XGBRegressor(**default_params)

    def _build_classification_model(self, params: Optional[Dict] = None) -> xgb.XGBClassifier:
        """Build XGBoost classification model."""
        default_params = {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'colsample_bynode': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 1,
            'gamma': 0,
            'objective': 'multi:softprob',
            'eval_metric': self.eval_metrics['classification'],
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0
        }

        if params:
            default_params.update(params)

        return xgb.XGBClassifier(**default_params)

    def build_model(self, params: Optional[Dict] = None):
        """Build XGBoost model based on task type."""
        if self.task_type == 'regression':
            self.model = self._build_regression_model(params)
        else:
            self.model = self._build_classification_model(params)
            self.n_classes = len(np.unique(params.get('classes', []))) if 'classes' in params else None

    def fit_with_early_stopping(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Optional[Dict] = None
    ):
        """
        Train model with early stopping using validation set.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            params: Model parameters
        """
        self.build_model(params)

        # Create evaluation set
        eval_set = [(X_val, y_val)]

        # Set early stopping parameters
        model_params = {
            'early_stopping_rounds': self.early_stopping_rounds,
            'eval_set': eval_set,
            'verbose': False
        }

        # Fit model with early stopping
        if self.task_type == 'regression':
            self.model.fit(
                X_train, y_train,
                **model_params
            )
        else:
            self.model.fit(
                X_train, y_train,
                **model_params
            )

        print(f"Early stopping at round {self.model.best_iteration}")

    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover', 'total_gain', 'total_cover')

        Returns:
            DataFrame with features and their importance scores
        """
        if self.feature_names is None:
            raise ValueError("Model must be trained first")

        importance_scores = self.model.feature_importances_

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)

        return importance_df

    def plot_tree_diagram(self, num_trees: int = 0, rankdir: str = 'UT'):
        """Plot XGBoost decision tree diagram."""
        return xgb.plot_tree(self.model, num_trees=num_trees, rankdir=rankdir)

    def plot_importance_bokeh(self):
        """Plot feature importance using Bokeh (if available)."""
        try:
            import xgboost as xgb
            return xgb.plot_importance(self.model, max_num_features=20, height=0.7)
        except:
            print("Bokeh not available, using matplotlib")
            return self.plot_tree_diagram

    def get_booster_attributions(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """
        Get SHAP-style feature attributions from XGBoost.

        Args:
            X: Features
            y: Target (optional)

        Returns:
            Feature attributions
        """
        booster = self.model.get_booster()

        # Method 1: Direct attribution via booster
        try:
            attributions = booster.predict(
                xgb.DMatrix(X),
                pred_contribs=True
            )
            return attributions[:, :-1]  # Exclude bias term
        except:
            print("Direct attribution not available")
            return None

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict = None,
        n_trials: int = 50,
        scoring: str = None
    ) -> Dict:
        """
        Perform hyperparameter tuning using Optuna (if available).

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

            # Default parameter search space
            if param_space is None:
                if self.task_type == 'regression':
                    param_space = {
                        'max_depth': (3, 8),
                        'learning_rate': (0.01, 0.3),
                        'n_estimators': (100, 1000),
                        'min_child_weight': (1, 10),
                        'subsample': (0.6, 1.0),
                        'colsample_bytree': (0.6, 1.0),
                        'reg_alpha': (0, 1),
                        'reg_lambda': (0, 1)
                    }
                else:  # classification
                    param_space = {
                        'max_depth': (3, 8),
                        'learning_rate': (0.01, 0.3),
                        'n_estimators': (100, 1000),
                        'min_child_weight': (1, 10),
                        'subsample': (0.6, 1.0),
                        'colsample_bytree': (0.6, 1.0),
                        'reg_alpha': (0, 1),
                        'reg_lambda': (0, 1),
                        'gamma': (0, 5)
                    }

            # Define objective function
            def objective(trial):
                params = {}
                for param, (low, high) in param_space.items():
                    if param in ['n_estimators']:
                        params[param] = trial.suggest_int(param, low, high)
                    elif param in ['max_depth', 'min_child_weight']:
                        params[param] = trial.suggest_int(param, low, high)
                    elif param in ['learning_rate']:
                        params[param] = trial.suggest_loguniform(param, low, high)
                    else:
                        params[param] = trial.suggest_uniform(param, low, high)

                # Build model with trial parameters
                self.build_model(params)

                # Use cross-validated score
                X_processed = self.preprocess_data(X, fit_scaler=True)

                if scoring is None:
                    if self.task_type == 'regression':
                        scoring = 'neg_root_mean_squared_error'
                    else:
                        scoring = 'accuracy'

                scores = cross_val_score(
                    self.model, X_processed, y,
                    cv=5, scoring=scoring,
                    n_jobs=-1
                )

                return scores.mean()

            # Run optimization
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.random_state)
            )
            study.optimize(objective, n_trials=n_trials, n_jobs=1)

            print(f"Best trial parameters: {study.best_params}")
            print(f"Best trial score: {study.best_value:.4f}")

            # Train final model with best parameters
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
        Create learning curve for XGBoost model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            params: Model parameters
            train_sizes: List of training set sizes to evaluate
        """
        if train_sizes is None:
            train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

        train_errors = []
        val_errors = []

        for size in train_sizes:
            # Sample training data
            n_train = int(len(X_train) * size)
            indices = np.random.choice(len(X_train), n_train, replace=False)
            X_sub = X_train.iloc[indices]
            y_sub = y_train.iloc[indices]

            # Train model
            self.build_model(params)
            self._train_model(X_sub, y_sub)

            # Evaluate on both training and validation sets
            train_pred = self._predict_model(X_sub)
            val_pred = self._predict_model(X_val)

            if self.task_type == 'regression':
                train_rmse = np.sqrt(((y_sub - train_pred) ** 2).mean())
                val_rmse = np.sqrt(((y_val - val_pred) ** 2).mean())
                train_errors.append(train_rmse)
                val_errors.append(val_rmse)
            else:  # classification
                train_acc = (y_sub == train_pred).mean()
                val_acc = (y_val == val_pred).mean()
                train_errors.append(1 - train_acc)  # Error rate
                val_errors.append(1 - val_acc)

        return train_sizes, train_errors, val_errors

    def save_model_info(self) -> Dict[str, Any]:
        """Get model information for saving."""
        info = {
            'model_type': 'xgboost',
            'task_type': self.task_type,
            'best_iteration': getattr(self.model, 'best_iteration', None),
            'best_score': getattr(self.model, 'best_score', None),
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names) if self.feature_names else 0
        }

        if hasattr(self.model, 'get_booster'):
            booster = self.model.get_booster()
            info.update({
                'trees': booster.num_features(),
                'n_estimators': getattr(self.model, 'n_estimators', None)
            })

        return info


# Factory function for creating XGBoost trainer instances
def create_xgboost_trainer(
    task_type: str = 'regression',
    model_name: str = 'xgboost',
    target_metric: str = 'rmse',
    cv_folds: int = 5,
    random_state: int = 42
) -> XGBoostTrainer:
    """
    Create an XGBoost trainer with optimal parameters based on the task.

    Args:
        task_type: 'regression' or 'classification'
        model_name: Name for the trainer
        target_metric: Target metric for optimization
        cv_folds: Number of CV folds
        random_state: Random seed

    Returns:
        Configured XGBoost trainer
    """
    trainer = XGBoostTrainer(
        model_name=model_name,
        task_type=task_type,
        n_folds=cv_folds,
        random_state=random_state,
        scaler_type='robust'  # Good for handling outliers
    )

    if task_type == 'regression':
        if target_metric == 'rmse':
            trainer.early_stopping_rounds = 100
        elif target_metric == 'mae':
            trainer.eval_metrics['regression'] = 'mae'
    else:  # classification
        if target_metric == 'auc':
            trainer.eval_metrics['classification'] = 'auc'

    return trainer