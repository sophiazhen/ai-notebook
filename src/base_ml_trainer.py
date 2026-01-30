import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class BaseMLTrainer(ABC):
    """
    Base class for machine learning model training, validation, and testing.
    Supports both regression and classification tasks with comprehensive evaluation metrics.
    """

    def __init__(
        self,
        model_name: str,
        task_type: str = 'regression',
        n_folds: int = 5,
        cv_strategy: str = 'kfold',
        random_state: int = 42,
        scaler_type: str = 'robust'
    ):
        """
        Initialize the trainer.

        Args:
            model_name: Name of the model
            task_type: 'regression' or 'classification'
            n_folds: Number of cross-validation folds
            cv_strategy: 'kfold', 'stratified', 'group', or 'time_series'
            random_state: Random seed for reproducibility
            scaler_type: Type of scaler to use ('standard', 'robust', 'minmax', None)
        """
        self.model_name = model_name
        self.task_type = task_type
        self.n_folds = n_folds
        self.cv_strategy = cv_strategy
        self.random_state = random_state
        self.scaler_type = scaler_type

        self.model = None
        self.scaler = None
        self.feature_names = None
        self.cv_results = {}
        self.best_params = None
        self.best_score = None

        self._setup_scaler()
        self._setup_cv_splitter()

    def _setup_scaler(self):
        """Initialize the data scaler."""
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

    def _setup_cv_splitter(self):
        """Setup cross-validation splitter."""
        if self.cv_strategy == 'stratified':
            self.cv_splitter = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state
            )
        elif self.cv_strategy == 'group':
            self.cv_splitter = GroupKFold(n_splits=self.n_folds)
        else:  # kfold
            self.cv_splitter = KFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state
            )

    @abstractmethod
    def build_model(self, params: Optional[Dict] = None):
        """Build the model with given parameters."""
        pass

    def preprocess_data(self, X: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """
        Preprocess the data including scaling and feature selection.

        Args:
            X: Input features
            fit_scaler: Whether to fit the scaler (use True for training data)

        Returns:
            Preprocessed features
        """
        X_processed = X.copy()

        # Handle scaling
        if self.scaler is not None:
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
            if fit_scaler:
                self.scaler.fit(X_processed[numeric_cols])
            X_processed[numeric_cols] = self.scaler.transform(X_processed[numeric_cols])

        return X_processed

    def calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression evaluation metrics."""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'huber_loss': self._huber_loss(y_true, y_pred, delta=1.0)
        }

        # Calculate percentage errors at different thresholds
        metrics['p95_error'] = self._percentage_error_within_threshold(y_true, y_pred, 0.95)
        metrics['p90_error'] = self._percentage_error_within_threshold(y_true, y_pred, 0.90)
        metrics['p80_error'] = self._percentage_error_within_threshold(y_true, y_pred, 0.80)

        return metrics

    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate classification evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'f1': f1_score(y_true, y_pred, average='macro'),
        }

        if y_prob is not None:
            try:
                if self.n_classes == 2:
                    metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
                    metrics['logloss'] = log_loss(y_true, y_prob[:, 1])
                else:
                    metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
                    metrics['logloss'] = log_loss(y_true, y_prob)
            except:
                metrics['auc'] = None
                metrics['logloss'] = None

        # Individual class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)

        for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
            metrics[f'precision_class_{i}'] = p
            metrics[f'recall_class_{i}'] = r
            metrics[f'f1_class_{i}'] = f

        return metrics

    @staticmethod
    def _huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
        """Calculate Huber loss."""
        error = y_true - y_pred
        is_small_error = np.abs(error) <= delta
        squared_loss = np.square(error) / 2
        linear_loss = delta * np.abs(error) - (delta ** 2) / 2
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))

    @staticmethod
    def _percentage_error_within_threshold(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float
    ) -> float:
        """Calculate percentage of predictions within threshold."""
        relative_error = np.abs(y_true - y_pred) / y_true
        return (relative_error <= threshold).mean() * 100

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation with comprehensive evaluation.

        Args:
            X: Features
            y: Target variable
            groups: Group labels for group-based cross-validation
            params: Model parameters

        Returns:
            Dictionary containing CV results
        """
        print(f"Starting {self.n_folds}-fold cross-validation for {self.model_name}...")

        self.feature_names = list(X.columns)
        X_processed = self.preprocess_data(X, fit_scaler=True)
        y_array = y.values

        cv_metrics = []
        all_predictions = []
        feature_importances = []
        model_save_path = f"models/{self.model_name}_cv_model.pkl"

        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)

        fold = 0
        for train_idx, val_idx in self.cv_splitter.split(X_processed, y_array, groups):
            print(f"\nFold {fold + 1}/{self.n_folds}")

            # Split data
            X_train, X_val = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
            y_train, y_val = y_array[train_idx], y_array[val_idx]

            # Build and train model
            self.build_model(params)
            self._train_model(X_train, y_train)

            # Make predictions
            train_pred = self._predict_model(X_train)
            val_pred = self._predict_model(X_val)

            # Calculate metrics
            if self.task_type == 'regression':
                train_metrics = self.calculate_regression_metrics(y_train, train_pred)
                val_metrics = self.calculate_regression_metrics(y_val, val_pred)
            else:  # classification
                train_prob = self._predict_proba_model(X_train) if callable(getattr(self.model, "predict_proba", None)) else None
                val_prob = self._predict_proba_model(X_val) if callable(getattr(self.model, "predict_proba", None)) else None

                train_metrics = self.calculate_classification_metrics(y_train, train_pred, train_prob)
                val_metrics = self.calculate_classification_metrics(y_val, val_pred, val_prob)

            # Store predictions
            fold_predictions = pd.DataFrame({
                'fold': fold,
                'actual': y_val,
                'predicted': val_pred,
                'sample_id': X_val.index
            })
            train_fold_predictions = pd.DataFrame({
                'fold': fold,
                'actual': y_train,
                'predicted': train_pred,
                'sample_id': X_train.index
            })
            all_predictions.append({
                'train': train_fold_predictions,
                'val': fold_predictions
            })

            # Store feature importance
            if hasattr(self.model, 'feature_importances_'):
                feature_importances.append(self.model.feature_importances_)
            elif hasattr(self.model, 'coef_'):
                feature_importances.append(np.abs(self.model.coef_))
            elif hasattr(self.model, 'feature_importances'):  # XGBoost
                feature_importances.append(self.model.feature_importances)

            # Display metrics
            print(f"Train - RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
            print(f"Val   - RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")

            cv_metrics.append({
                'fold': fold,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })

            # Save model for this fold
            fold_model_save_path = f"models/{self.model_name}_fold_{fold}.pkl"
            joblib.dump(self.model, fold_model_save_path)

            fold += 1

        # Aggregate results
        self.cv_results = self._aggregate_cv_results(cv_metrics, feature_importances)

        print(f"\nCross-validation completed!")
        print(f"Average Val RMSE: {self.cv_results['mean_val_rmse']:.4f} (±{self.cv_results['std_val_rmse']:.4f})")
        print(f"Average Val R²: {self.cv_results['mean_val_r2']:.4f} (±{self.cv_results['std_val_r2']:.4f})")
        if self.task_type == 'classification':
            print(f"Average Val Accuracy: {self.cv_results['mean_val_accuracy']:.4f} (±{self.cv_results['std_val_accuracy']:.4f})")

        # Save CV predictions
        self._save_cv_predictions(all_predictions, self.model_name)

        return self.cv_results

    def _train_model(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """Train the model (to be implemented by subclasses if needed)."""
        if hasattr(self.model, 'fit'):
            self.model.fit(X_train, y_train)
        else:
            raise NotImplementedError("Model training method not implemented")

    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the model."""
        if self.task_type == 'regression':
            if hasattr(self.model, 'predict'):
                return self.model.predict(X)
            else:
                raise NotImplementedError("Model prediction method not implemented")
        else:  # classification
            if hasattr(self.model, 'predict'):
                return self.model.predict(X)
            else:
                raise NotImplementedError("Model classification not implemented")

    def _predict_proba_model(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Get prediction probabilities (for classification)."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None

    def _aggregate_cv_results(
        self,
        cv_metrics: List[Dict],
        feature_importances: List[np.ndarray]
    ) -> Dict[str, Any]:
        """Aggregate cross-validation results."""
        results = {}

        # Aggregate metrics
        train_metrics_all = [m['train_metrics'] for m in cv_metrics]
        val_metrics_all = [m['val_metrics'] for m in cv_metrics]

        # Mean and std for each metric
        if self.task_type == 'regression':
            metrics_to_agg = ['rmse', 'mae', 'mse', 'r2', 'mape', 'huber_loss',
                            'p95_error', 'p90_error', 'p80_error']
        else:
            metrics_to_agg = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'logloss']
            for i in range(self.n_classes if hasattr(self, 'n_classes') else 2):
                metrics_to_agg.extend([f'precision_class_{i}', f'recall_class_{i}', f'f1_class_{i}'])

        for metric in metrics_to_agg:
            train_values = [m[metric] for m in train_metrics_all if m[metric] is not None]
            val_values = [m[metric] for m in val_metrics_all if m[metric] is not None]

            results[f'mean_train_{metric}'] = np.mean(train_values) if train_values else None
            results[f'std_train_{metric}'] = np.std(train_values) if train_values else None
            results[f'mean_val_{metric}'] = np.mean(val_values) if val_values else None
            results[f'std_val_{metric}'] = np.std(val_values) if val_values else None

        # Feature importance
        if feature_importances:
            feature_imp_array = np.array(feature_importances)
            results['feature_importance_mean'] = np.mean(feature_imp_array, axis=0)
            results['feature_importance_std'] = np.std(feature_imp_array, axis=0)
            results['feature_importance_ranking'] = np.argsort(results['feature_importance_mean'])[::-1]

        # Individual fold results
        results['fold_results'] = cv_metrics

        return results

    def _save_cv_predictions(self, all_predictions: List[Dict], model_name: str):
        """Save cross-validation predictions to CSV."""
        # Combine all fold predictions
        train_dfs = [pred['train'] for pred in all_predictions]
        val_dfs = [pred['val'] for pred in all_predictions]

        train_combined = pd.concat(train_dfs, ignore_index=True)
        val_combined = pd.concat(val_dfs, ignore_index=True)

        # Save to CSV
        train_combined.to_csv(f"results/{model_name}_cv_train_predictions.csv", index=False)
        val_combined.to_csv(f"results/{model_name}_cv_val_predictions.csv", index=False)

    def predict_test(self, X_test: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions on test set.

        Args:
            X_test: Test features

        Returns:
            Dictionary with predictions and optionally probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")

        X_test_processed = self.preprocess_data(X_test, fit_scaler=False)
        predictions = self._predict_model(X_test_processed)

        results = {'predictions': predictions}

        if self.task_type == 'classification':
            probabilities = self._predict_proba_model(X_test_processed)
            if probabilities is not None:
                results['probabilities'] = probabilities

        return results

    def plot_feature_importance(self, top_n: int = 20):
        """Plot feature importance."""
        if 'feature_importance_mean' in self.cv_results:
            importance_mean = self.cv_results['feature_importance_mean']
            importance_std = self.cv_results['feature_importance_std']
            ranking = self.cv_results['feature_importance_ranking']

            # Select top features
            top_indices = ranking[:top_n]
            top_features = [self.feature_names[i] for i in top_indices]
            top_importance = importance_mean[top_indices]
            top_std = importance_std[top_indices]

            # Create plot
            plt.figure(figsize=(10, 8))
            y_pos = np.arange(len(top_features))
            plt.errorbar(top_importance, y_pos, xerr=top_std, fmt='o', capsize=5)
            plt.yticks(y_pos, top_features)
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importances for {self.model_name}')
            plt.tight_layout()
            plt.savefig(f"results/{self.model_name}_feature_importance.png", dpi=300)
            plt.show()
        else:
            print("Feature importance not available for this model.")

    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, dataset_type: str = 'validation'):
        """Plot predictions vs actual values."""
        plt.figure(figsize=(12, 5))

        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{dataset_type.title()} Predictions vs Actual')

        # Residual plot
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title(f'{dataset_type.title()} Residuals')

        plt.tight_layout()
        plt.savefig(f"results/{self.model_name}_{dataset_type}_predictions.png", dpi=300)
        plt.show()

    def save_model(self, path: str):
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'cv_results': self.cv_results,
            'best_params': self.best_params,
            'feature_names': self.feature_names,
            'task_type': self.task_type,
            'model_name': self.model_name
        }
        joblib.dump(model_data, path)

    def load_model(self, path: str):
        """Load a saved model."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.cv_results = model_data['cv_results']
        self.best_params = model_data['best_params']
        self.feature_names = model_data['feature_names']
        self.task_type = model_data['task_type']
        self.model_name = model_data['model_name']