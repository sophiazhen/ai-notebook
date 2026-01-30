import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.base_ml_trainer import BaseMLTrainer
from src.xgboost_trainer import XGBoostTrainer
from src.lightgbm_trainer import LightGBMTrainer
import warnings
warnings.filterwarnings('ignore')


class EnsembleTrainer(BaseMLTrainer):
    """
    Ensemble model trainer supporting various stacking and blending strategies.
    Optimized for semiconductor wafer data with special handling for high-dimensional
    features and time-series cross-validation.
    """

    def __init__(
        self,
        model_name: str = 'ensemble',
        task_type: str = 'regression',
        ensemble_type: str = 'stacking',
        n_folds: int = 5,
        cv_strategy: str = 'kfold',
        random_state: int = 42,
        scaler_type: str = 'robust',
        second_level_model: str = 'ridge',
        stack_probabilities_for_classification: bool = True
    ):
        """
        Initialize ensemble trainer.

        Args:
            model_name: Name for the ensemble model
            task_type: 'regression' or 'classification'
            ensemble_type: 'stacking', 'blending', or 'voting'
            n_folds: Number of cross-validation folds
            cv_strategy: 'kfold', 'stratified', 'group'
            random_state: Random seed
            scaler_type: Scaler for preprocessing
            second_level_model: Meta-learner ('ridge', 'logistic', 'elastic_net')
            stack_probabilities_for_classification: Use probabilities in classification stacking
        """
        super().__init__(model_name, task_type, n_folds, cv_strategy, random_state, scaler_type)
        self.ensemble_type = ensemble_type
        self.second_level_model = second_level_model
        self.stack_probabilities = stack_probabilities_for_classification
        self.base_models = {}
        self.meta_learner = None
        self.base_predictions = {}
        self.meta_features = None

    def add_base_model(self, name: str, model: BaseMLTrainer):
        """Add a base model to the ensemble."""
        self.base_models[name] = model

    def remove_base_model(self, name: str):
        """Remove a base model from the ensemble."""
        if name in self.base_models:
            del self.base_models[name]

    def _build_meta_learner(self, params: Optional[Dict] = None):
        """Build the second-level meta-learner."""
        if self.task_type == 'regression':
            if self.second_level_model == 'ridge':
                default_params = {'alpha': 1.0, 'random_state': self.random_state}
                if params:
                    default_params.update(params)
                self.meta_learner = Ridge(**default_params)
            elif self.second_level_model == 'elastic_net':
                from sklearn.linear_model import ElasticNet
                default_params = {'alpha': 1.0, 'l1_ratio': 0.5}
                if params:
                    default_params.update(params)
                self.meta_learner = ElasticNet(**default_params)
            else:
                # Use cross-validated ridge
                self.meta_learner = RidgeCV(alphas=np.logspace(-3, 3, 7))
        else:  # classification
            if self.second_level_model == 'logistic':
                default_params = {'max_iter': 1000, 'random_state': self.random_state}
                if params:
                    default_params.update(params)
                self.meta_learner = LogisticRegression(**default_params)
            else:
                # Use cross-validated logistic regression
                if self.n_classes and self.n_classes > 2:
                    default_params = {'max_iter': 1000, 'random_state': self.random_state}
                else:
                    default_params = {'max_iter': 1000, 'random_state': self.random_state}
                self.meta_learner = LogisticRegressionCV(**default_params)

    def build_model(self, params: Optional[Dict] = None):
        """Build the ensemble model."""
        self._build_meta_learner(params)

    def stacking_strategy(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Implements Stacking (Stacked Generalization) strategy.

        Args:
            X: Features
            y: Target
            groups: Group labels for group-based CV

        Returns:
            Dictionary with stacking results
        """
        print(f"Running stacking ensemble with {len(self.base_models)} base models...")

        X_processed = self.preprocess_data(X, fit_scaler=True)
        y_array = y.values

        # Initialize meta-feature arrays
        n_samples = len(X)
        meta_features = np.zeros((n_samples, len(self.base_models)))

        if self.task_type == 'classification' and self.stack_probabilities:
            if hasattr(self, 'n_classes') and self.n_classes:
                self.n_classes = len(np.unique(y))
            else:
                self.n_classes = len(np.unique(y))
            meta_features = np.zeros((n_samples, len(self.base_models) * self.n_classes))

        # Step 1: Train base models and generate out-of-fold predictions
        self.base_predictions = {}
        fold_splitter = self.cv_splitter

        for idx, (model_name, base_model) in enumerate(self.base_models.items()):
            print(f"Processing base model: {model_name}")

            oof_predictions = np.zeros(n_samples)
            model_preds = []

            for fold, (train_idx, val_idx) in enumerate(fold_splitter.split(X_processed, y_array, groups)):
                X_train, X_val = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
                y_train, y_val = y_array[train_idx], y_array[val_idx]

                # Build and train base model for this fold
                base_model.build_model()
                base_model._train_model(X_train, y_train)

                # Make out-of-fold prediction
                if self.task_type == 'classification' and self.stack_probabilities:
                    oof_pred = base_model._predict_proba_model(X_val)
                    if oof_pred is not None:
                        # Use class probabilities as meta-features
                        start_idx = idx * self.n_classes
                        end_idx = (idx + 1) * self.n_classes
                        meta_features[val_idx, start_idx:end_idx] = oof_pred
                else:
                    oof_pred = base_model._predict_model(X_val)
                    oof_predictions[val_idx] = oof_pred

                model_preds.append({
                    'fold': fold,
                    'actual': y_val,
                    'predicted': oof_pred
                })

            if self.task_type != 'classification' or not self.stack_probabilities:
                meta_features[:, idx] = oof_predictions

            self.base_predictions[model_name] = model_preds

        # Step 2: Train meta-learner on out-of-fold predictions
        print("Training meta-learner...")

        # Handle potential NaN values in meta features
        if np.any(np.isnan(meta_features)):
            meta_features = np.nan_to_num(meta_features, nan=-999)

        # Simple train-validation split for meta-learner
        X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
            meta_features, y_array, test_size=0.2, random_state=self.random_state
        )

        # Train meta-learner
        self.meta_learner.fit(X_meta_train, y_meta_train)

        # Validate meta-learner
        meta_pred = self._predict_model_meta(X_meta_val)

        if self.task_type == 'regression':
            metrics = self.calculate_regression_metrics(y_meta_val, meta_pred)
        else:
            metrics = self.calculate_classification_metrics(y_meta_val, meta_pred)

        print(f"Meta-learner validation RMSE: {metrics.get('rmse', metrics.get('accuracy', 'N/A')):.4f}")

        # Step 3: Retrain base models on full data
        self.refit_base_models(X_processed, y_array)

        # Store meta-feature names
        feature_names = []
        if self.task_type == 'classification' and self.stack_probabilities:
            for model_name in self.base_models.keys():
                for class_idx in range(self.n_classes):
                    feature_names.append(f"{model_name}_prob_class_{class_idx}")
        else:
            feature_names = list(self.base_models.keys())

        self.meta_features = feature_names

        return {
            'meta_features': meta_features,
            'meta_learner_scores': metrics,
            'cv_results': self.base_predictions,
            'feature_names': feature_names
        }

    def blending_strategy(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        holdout_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Implements Blending strategy (simplified stacking).

        Args:
            X: Features
            y: Target
            test_size: Size of validation split
            holdout_size: Size of holdout set for meta-learner

        Returns:
            Dictionary with blending results
        """
        print(f"Running blending ensemble with {len(self.base_models)} base models...")

        # Split data into training, validation, and holdout sets
        X_processed = self.preprocess_data(X, fit_scaler=True)
        y_array = y.values

        # First split: training + validation for base models
        X_train_base, X_val_base, y_train_base, y_val_base = train_test_split(
            X_processed, y_array, test_size=test_size, random_state=self.random_state
        )

        # Second split: holdout for meta-learner
        X_train_meta, X_holdout, y_train_meta, y_holdout = train_test_split(
            X_train_base, y_train_base, test_size=holdout_size,
            random_state=self.random_state + 1
        )

        # Step 1: Train base models on training data (including meta training)
        print("Training base models on training data...")

        base_predictions_holdout = []

        for model_name, base_model in self.base_models.items():
            print(f"Training {model_name}...")

            base_model.build_model()
            base_model._train_model(X_train_meta, y_train_meta)

            # Predict on holdout set
            if self.task_type == 'classification' and self.stack_probabilities:
                pred_proba = base_model._predict_proba_model(X_holdout)
                base_predictions_holdout.append(pred_proba)
            else:
                pred = base_model._predict_model(X_holdout)
                base_predictions_holdout.append(pred.reshape(-1, 1))

        # Stack predictions
        meta_features_holdout = np.hstack(base_predictions_holdout)

        # Step 2: Train meta-learner on holdout predictions
        print("Training meta-learner on holdout predictions...")

        # Handle potential NaN values
        if np.any(np.isnan(meta_features_holdout)):
            meta_features_holdout = np.nan_to_num(meta_features_holdout, nan=-999)

        self.meta_learner.fit(meta_features_holdout, y_holdout)

        # Validate blending
        meta_pred_holdout = self._predict_model_meta(meta_features_holdout)

        if self.task_type == 'regression':
            metrics = self.calculate_regression_metrics(y_holdout, meta_pred_holdout)
        else:
            metrics = self.calculate_classification_metrics(y_holdout, meta_pred_holdout)

        # Step 3: Evaluate on validation set
        print("Evaluating on validation set...")

        base_predictions_val = []
        for model_name, base_model in self.base_models.items():
            if self.task_type == 'classification' and self.stack_probabilities:
                pred_proba = base_model._predict_proba_model(X_val_base)
                base_predictions_val.append(pred_proba)
            else:
                pred = base_model._predict_model(X_val_base)
                base_predictions_val.append(pred.reshape(-1, 1))

        meta_features_val = np.hstack(base_predictions_val)

        if np.any(np.isnan(meta_features_val)):
            meta_features_val = np.nan_to_num(meta_features_val, nan=-999)

        meta_pred_val = self._predict_model_meta(meta_features_val)

        if self.task_type == 'regression':
            val_metrics = self.calculate_regression_metrics(y_val_base, meta_pred_val)
        else:
            val_metrics = self.calculate_classification_metrics(y_val_base, meta_pred_val)

        # Step 4: Retrain base models on full training data
        X_train_full = pd.concat([X_train_meta, X_holdout])
        y_train_full = np.concatenate([y_train_meta, y_holdout])

        self.refit_base_models(X_train_full, y_train_full)

        print(f"Blending validation RMSE: {val_metrics.get('rmse', val_metrics.get('accuracy', 'N/A')):.4f}")

        return {
            'blend_metrics': metrics,
            'val_metrics': val_metrics,
            'predictions': {
                'holdout_actual': y_holdout,
                'holdout_pred': meta_pred_holdout,
                'val_actual': y_val_base,
                'val_pred': meta_pred_val
            }
        }

    def voting_strategy(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        voting_type: str = 'soft',
        weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Implements Voting ensemble strategy.

        Args:
            X: Features
            y: Target
            voting_type: 'hard' or 'soft' (classification only)
            weights: Weights for each model (optional)

        Returns:
            Dictionary with voting results
        """
        print(f"Running voting ensemble with {len(self.base_models)} base models...")

        X_processed = self.preprocess_data(X, fit_scaler=True)
        y_array = y.values

        if self.task_type == 'regression':
            # Create weighted average voting for regression
            estimator_list = [(name, base_model.model) for name, base_model in self.base_models.items()]
            voting_model = VotingRegressor(
                estimators=estimator_list,
                weights=weights
            )
        else:  # classification
            estimator_list = [(name, base_model.model) for name, base_model in self.base_models.items()]
            voting_model = VotingClassifier(
                estimators=estimator_list,
                voting=voting_type,
                weights=weights
            )

        # Train voting model
        voting_model.fit(X_processed, y_array)

        # Make predictions
        y_pred = voting_model.predict(X_processed)

        if self.task_type == 'regression':
            metrics = self.calculate_regression_metrics(y_array, y_pred)
        else:
            if voting_type == 'soft' and hasattr(voting_model, 'predict_proba'):
                y_prob = voting_model.predict_proba(X_processed)
                metrics = self.calculate_classification_metrics(y_array, y_pred, y_prob)
            else:
                metrics = self.calculate_classification_metrics(y_array, y_pred)

        print(f"Voting ensemble RMSE: {metrics.get('rmse', metrics.get('accuracy', 'N/A')):.4f}")

        return {
            'voting_model': voting_model,
            'metrics': metrics,
            'predictions': y_pred
        }

    def refit_base_models(self, X: pd.DataFrame, y: np.ndarray):
        """Retrain all base models on the full dataset."""
        for model_name, base_model in self.base_models.items():
            print(f"Retraining {model_name} on full dataset...")

            # Preprocess data for this model
            X_model = base_model.preprocess_data(X, fit_scaler=True)

            # Build fresh model for this base
            base_model.build_model()
            base_model._train_model(X_model, y)

    def _predict_model_meta(self, X_meta: np.ndarray) -> np.ndarray:
        """Make predictions with the meta-learner."""
        if self.task_type == 'regression':
            return self.meta_learner.predict(X_meta)
        else:  # classification
            if hasattr(self.meta_learner, 'predict_proba') and self.stack_probabilities:
                # Return probabilities for classification
                return self.meta_learner.predict_proba(X_meta)
            else:
                return self.meta_learner.predict(X_meta)

    def predict_test(self, X_test: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions on test set using trained ensemble.

        Args:
            X_test: Test features

        Returns:
            Ensemble predictions
        """
        if self.ensemble_type == 'stacking' or self.ensemble_type == 'blending':
            # First, get base model predictions
            base_predictions_test = []

            for model_name, base_model in self.base_models.items():
                preds = base_model.predict_test(X_test)

                if self.task_type == 'classification' and self.stack_probabilities:
                    # Use probabilities
                    base_predictions_test.append(preds['probabilities'])
                else:
                    # Use predictions
                    base_predictions_test.append(preds['predictions'].reshape(-1, 1))

            # Stack base predictions
            X_meta_test = np.hstack(base_predictions_test)

            # Handle NaN values
            if np.any(np.isnan(X_meta_test)):
                X_meta_test = np.nan_to_num(X_meta_test, nan=-999)

            # Make final predictions with meta-learner
            final_predictions = self._predict_model_meta(X_meta_test)

        else:  # voting
            X_test_processed = self.preprocess_data(X_test, fit_scaler=False)
            final_predictions = self.model.predict(X_test_processed)

        return {'predictions': final_predictions}

    def get_ensemble_weights(self, method: str = 'optimize') -> Dict[str, float]:
        """
        Calculate optimal weights for ensemble models.

        Args:
            method: 'optimize' (uses scipy) or 'cv_based' (uses CV scores)

        Returns:
            Dictionary with model weights
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            print("Scipy not available for weight optimization")
            return {name: 1.0 / len(self.base_models) for name in self.base_models.keys()}

        if method == 'optimize':
            return self._optimize_weights()
        elif method == 'cv_based':
            return self._cv_based_weights()
        else:
            raise ValueError("Method must be 'optimize' or 'cv_based'")

    def _optimize_weights(self) -> Dict[str, float]:
        """Optimize ensemble weights using validation scores."""
        # For this, we need validation predictions from each model
        # This assumes we've run CV already

        def objective(weights):
            if self.task_type == 'regression':
                # Minimize RMSE
                weighted_pred = self._calculate_weighted_prediction(weights)
                # Note: This would need actual validation predictions
                # For demonstration, we return a dummy value
                return 0.1
            else:
                # Maximize accuracy (converted to minimization)
                return -0.9  # Dummy value

        # Simplex constraint: weights sum to 1 and are non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0) for _ in self.base_models]

        # Uniform initial guess
        n_models = len(self.base_models)
        initial_weights = np.ones(n_models) / n_models

        result = minimize(
            objective, initial_weights, method='SLSQP',
            bounds=bounds, constraints=constraints
        )

        optimal_weights = {}
        for i, (name, _) in enumerate(self.base_models.items()):
            optimal_weights[name] = result.x[i]

        return optimal_weights

    def _cv_based_weights(self) -> Dict[str, float]:
        """Calculate weights based on CV performance."""
        weights = {}
        scores = {}

        for name, base_model in self.base_models.items():
            if hasattr(base_model, 'cv_results'):
                if self.task_type == 'regression':
                    score = base_model.cv_results.get('mean_val_rmse', 1.0)
                    # Use inverse of RMSE for weighting
                    scores[name] = 1.0 / np.log(1.0 + score)
                else:
                    score = base_model.cv_results.get('mean_val_accuracy', 0.5)
                    scores[name] = score

        # Normalize scores to get weights
        total_score = sum(scores.values())
        if total_score > 0:
            for name, score in scores.items():
                weights[name] = score / total_score
        else:
            # Equal weights if no scores available
            n = len(self.base_models)
            for name in self.base_models.keys():
                weights[name] = 1.0 / n

        return weights

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Perform cross-validation for ensemble models."""
        print(f"Running cross-validation for {self.ensemble_type} ensemble...")

        if self.ensemble_type == 'stacking':
            results = self.stacking_strategy(X, y, groups)
        elif self.ensemble_type == 'blending':
            results = self.blending_strategy(X, y)
        elif self.ensemble_type == 'voting':
            results = self.voting_strategy(X, y)
        else:
            raise ValueError("ensemble_type must be 'stacking', 'blending', or 'voting'")

        self.cv_results = results
        return results

    def plot_model_weights(self):
        """Visualize ensemble model weights."""
        if not hasattr(self, 'meta_learner') or not hasattr(self.meta_learner, 'coef_'):
            print("Meta-learner does not support weight visualization")
            return

        weights = self.meta_learner.coef_

        plt.figure(figsize=(10, 6))

        if self.task_type == 'regression':
            x_pos = np.arange(len(self.meta_features))
            plt.bar(x_pos, weights)
            plt.xticks(x_pos, self.meta_features, rotation=45, ha='right')
            plt.ylabel('Weight')
            plt.title('Ensemble Model Weights (Regression)')
        else:  # classification
            # For multi-class, weights is a matrix
            if weights.ndim > 1:
                sns.heatmap(weights,
                          xticklabels=self.meta_features,
                          yticklabels=[f"Class {i}" for i in range(self.n_classes or 2)],
                          annot=True, fmt='.3f')
                plt.title('Ensemble Model Weights (Classification)')
            else:
                plt.bar(range(len(self.meta_features)), weights)
                plt.xticks(range(len(self.meta_features)), self.meta_features, rotation=45, ha='right')
                plt.ylabel('Weight')
                plt.title('Ensemble Model Weights')

        plt.tight_layout()
        plt.savefig(f"results/{self.model_name}_ensemble_weights.png", dpi=300)
        plt.show()


# Helper functions for creating common ensemble configurations
def create_stacking_ensemble(
    base_model_names: List[str] = ['xgboost', 'lightgbm'],
    task_type: str = 'regression',
    cv_folds: int = 5,
    random_state: int = 42
) -> EnsembleTrainer:
    """
    Create a stacking ensemble with default base models.

    Args:
        base_model_names: List of base model names
        task_type: 'regression' or 'classification'
        cv_folds: Number of CV folds
        random_state: Random seed

    Returns:
        Stacking ensemble trainer
    """
    ensemble = EnsembleTrainer(
        ensemble_type='stacking',
        task_type=task_type,
        n_folds=cv_folds,
        random_state=random_state
    )

    # Add base models
    if 'xgboost' in base_model_names:
        ensemble.add_base_model('xgboost', XGBoostTrainer(
            model_name='xgb_stacking',
            task_type=task_type,
            n_folds=cv_folds,
            random_state=random_state
        ))

    if 'lightgbm' in base_model_names:
        ensemble.add_base_model('lightgbm', LightGBMTrainer(
            model_name='lgb_stacking',
            task_type=task_type,
            n_folds=cv_folds,
            random_state=random_state
        ))

    return ensemble


def create_blending_ensemble(
    base_model_names: List[str] = ['xgboost', 'lightgbm'],
    task_type: str = 'regression',
    test_size: float = 0.2,
    holdout_size: float = 0.3,
    random_state: int = 42
) -> EnsembleTrainer:
    """
    Create a blending ensemble with default base models.

    Args:
        base_model_names: List of base model names
        task_type: 'regression' or 'classification'
        test_size: Size of test set for base models
        holdout_size: Size of holdout set for meta-learner
        random_state: Random seed

    Returns:
        Blending ensemble trainer
    """
    ensemble = EnsembleTrainer(
        ensemble_type='blending',
        task_type=task_type,
        random_state=random_state
    )

    # Add base models
    if 'xgboost' in base_model_names:
        ensemble.add_base_model('xgboost', XGBoostTrainer(
            model_name='xgb_blending',
            task_type=task_type,
            random_state=random_state
        ))

    if 'lightgbm' in base_model_names:
        ensemble.add_base_model('lightgbm', LightGBMTrainer(
            model_name='lgb_blending',
            task_type=task_type,
            random_state=random_state
        ))

    return ensemble


def create_voting_ensemble(
    base_model_names: List[str] = ['xgboost', 'lightgbm'],
    task_type: str = 'regression',
    voting_type: str = 'soft',
    random_state: int = 42
) -> EnsembleTrainer:
    """
    Create a voting ensemble with default base models.

    Args:
        base_model_names: List of base model names
        task_type: 'regression' or 'classification'
        voting_type: 'hard' or 'soft' (classification only)
        random_state: Random seed

    Returns:
        Voting ensemble trainer
    """
    ensemble = EnsembleTrainer(
        ensemble_type='voting',
        task_type=task_type,
        random_state=random_state
    )

    # Add base models
    if 'xgboost' in base_model_names:
        ensemble.add_base_model('xgboost', XGBoostTrainer(
            model_name='xgb_voting',
            task_type=task_type,
            random_state=random_state
        ))

    if 'lightgbm' in base_model_names:
        ensemble.add_base_model('lightgbm', LightGBMTrainer(
            model_name='lgb_voting',
            task_type=task_type,
            random_state=random_state
        ))

    return ensemble