import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from db.database import DatabaseManager
from src.base_ml_trainer import BaseMLTrainer
from src.xgboost_trainer import XGBoostTrainer, create_xgboost_trainer
from src.lightgbm_trainer import LightGBMTrainer, create_lightgbm_trainer
from src.ensemble_trainer import (
    EnsembleTrainer, create_stacking_ensemble,
    create_blending_ensemble, create_voting_ensemble
)


class ModelComparisonSuite:
    """
    Comprehensive model comparison suite for evaluating multiple ML models
    and ensemble strategies on semiconductor wafer data.
    """

    def __init__(
        self,
        task_type: str = 'regression',
        random_state: int = 42,
        db_path: str = "ml_experiments.db"
    ):
        """
        Initialize comparison suite.

        Args:
            task_type: 'regression' or 'classification'
            random_state: Random seed for reproducibility
            db_path: Path to database for storing results
        """
        self.task_type = task_type
        self.random_state = random_state
        self.experiments = {}
        self.results = {}

        # Initialize database
        self.db = DatabaseManager(db_path)

        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def add_experiment(
        self,
        experiment_name: str,
        model: BaseMLTrainer,
        params: Optional[Dict] = None
    ):
        """Add an experiment to the comparison suite."""
        self.experiments[experiment_name] = {
            'model': model,
            'params': params or {},
            'experiment_name': experiment_name
        }

    def run_experiments(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_models: bool = True,
        verbose: bool = True
    ):
        """
        Run all experiments and compare results.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            X_test: Test features
            y_test: Test target
            save_models: Whether to save trained models
            verbose: Whether to print progress
        """
        print(f"\n{'='*60}")
        print(f"{' '*20} MODEL COMPARISON SUITE")
        print(f"{'='*60}\n")
        print(f"Task Type: {self.task_type.upper()}")
        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"\nExperiments to run: {len(self.experiments)}")

        # Create results directory
        import os
        os.makedirs("results", exist_ok=True)

        # Run each experiment
        for exp_name, exp_config in self.experiments.items():
            print(f"\n{'-'*40}")
            print(f"Running experiment: {exp_name}")
            print(f"{'-'*40}")

            start_time = pd.Timestamp.now()
            experiment_id = None

            try:
                model = exp_config['model']
                params = exp_config['params']

                # Create experiment in database
                experiment_id = self.db.create_experiment(
                    experiment_name=exp_name,
                    model_type=model.model_name,
                    features=list(X_train.columns),
                    train_samples=len(X_train),
                    val_samples=len(X_val),
                    test_samples=len(X_test),
                    parameters=params,
                    ensemble_type=model.ensemble_type if hasattr(model, 'ensemble_type') else None
                )

                # Train model
                if verbose:
                    print(f"Training {model.model_name}...")

                # Run cross-validation
                cv_results = model.cross_validate(X_train, y_train)

                # Save CV metrics to database
                for fold, metrics in enumerate(cv_results.get('fold_results', [])):
                    self.db.save_metrics(
                        experiment_id,
                        metrics['train_metrics'],
                        'train',
                        fold_id=fold
                    )
                    self.db.save_metrics(
                        experiment_id,
                        metrics['val_metrics'],
                        'val',
                        fold_id=fold
                    )

                # Train on full training set
                if verbose:
                    print("Training on full training set...")

                model.build_model(params)
                model._train_model(model.preprocess_data(X_train, fit_scaler=True), y_train.values)

                # Evaluate on validation set
                val_predictions = model.predict_test(X_val)
                val_pred = val_predictions['predictions']

                if self.task_type == 'regression':
                    val_metrics = model.calculate_regression_metrics(y_val.values, val_pred)
                else:
                    val_prob = val_predictions.get('probabilities')
                    val_metrics = model.calculate_classification_metrics(y_val.values, val_pred, val_prob)

                # Save validation metrics
                self.db.save_metrics(experiment_id, val_metrics, 'val')

                # Evaluate on test set
                test_predictions = model.predict_test(X_test)
                test_pred = test_predictions['predictions']

                if self.task_type == 'regression':
                    test_metrics = model.calculate_regression_metrics(y_test.values, test_pred)
                else:
                    test_prob = test_predictions.get('probabilities')
                    test_metrics = model.calculate_classification_metrics(y_test.values, test_pred, test_prob)

                # Save test metrics
                self.db.save_metrics(experiment_id, test_metrics, 'test')

                # Save feature importance
                if 'feature_importance_mean' in cv_results:
                    self.db.save_feature_importance(
                        experiment_id,
                        model.feature_names,
                        cv_results['feature_importance_mean']
                    )

                # Save model artifacts
                self.db.save_model_artifact(experiment_id, 'model', model)
                if model.scaler is not None:
                    self.db.save_model_artifact(experiment_id, 'scaler', model.scaler)

                # Save predictions
                val_pred_df = pd.DataFrame({
                    'sample_id': X_val.index,
                    'actual': y_val.values,
                    'predicted': val_pred,
                    'fold': None
                })
                self.db.save_predictions(experiment_id, val_pred_df, 'val')

                test_pred_df = pd.DataFrame({
                    'sample_id': X_test.index,
                    'actual': y_test.values,
                    'predicted': test_pred,
                    'fold': None
                })
                self.db.save_predictions(experiment_id, test_pred_df, 'test')

                # Store results
                self.results[exp_name] = {
                    'experiment_id': experiment_id,
                    'cv_metrics': cv_results,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics,
                    'feature_importance': cv_results.get('feature_importance_mean', None),
                    'runtime': (pd.Timestamp.now() - start_time).seconds
                }

                # Save model
                if save_models:
                    model_save_path = f"results/{exp_name}_model.pkl"
                    model.save_model(model_save_path)

                if verbose:
                    print(f"Completed experiment {exp_name}")
                    print(f"Test RMSE: {test_metrics.get('rmse', test_metrics.get('accuracy', 'N/A')):.4f}")
                    print(f"Runtime: {self.results[exp_name]['runtime']}s")

            except Exception as e:
                print(f"Error in experiment {exp_name}: {str(e)}")
                import traceback
                traceback.print_exc()

    def get_comparison_results(self, dataset_type: str = 'test') -> pd.DataFrame:
        """
        Get comparison results for all experiments.

        Args:
            dataset_type: Dataset to compare ('val' or 'test')

        Returns:
            Comparison DataFrame
        """
        results = []

        for exp_name, exp_results in self.results.items():
            metrics = exp_results[f'{dataset_type}_metrics']
            cv_results = exp_results['cv_metrics']

            record = {
                'Experiment': exp_name,
                'Model': self.experiments[exp_name]['model'].model_name,
                'Ensemble': getattr(self.experiments[exp_name]['model'], 'ensemble_type', 'N/A'),
                'CV_RMSE_mean': cv_results.get('mean_val_rmse', np.nan),
                'CV_RMSE_std': cv_results.get('std_val_rmse', np.nan),
                f'{dataset_type.title()}_RMSE': metrics.get('rmse', np.nan),
                f'{dataset_type.title()}_MAE': metrics.get('mae', np.nan),
                f'{dataset_type.title()}_R2': metrics.get('r2', np.nan),
                f'{dataset_type.title()}_MAPE': metrics.get('mape', np.nan),
                'Runtime_sec': exp_results.get('runtime', np.nan)
            }

            # Add accuracy and other classification metrics if classification task
            if self.task_type == 'classification':
                record.update({
                    f'{dataset_type.title()}_Accuracy': metrics.get('accuracy', np.nan),
                    f'{dataset_type.title()}_Precision': metrics.get('precision', np.nan),
                    f'{dataset_type.title()}_Recall': metrics.get('recall', np.nan),
                    f'{dataset_type.title()}_F1': metrics.get('f1', np.nan),
                    f'{dataset_type.title()}_AUC': metrics.get('auc', np.nan)
                })

            results.append(record)

        return pd.DataFrame(results).round(4)

    def plot_model_comparison(self, dataset_type: str = 'test', metric: str = 'rmse', metric_label: str = None):
        """
        Plot model comparison for specific metric.

        Args:
            dataset_type: Dataset to compare ('val' or 'test')
            metric: Metric to compare ('rmse', 'mae', 'r2', 'accuracy', etc.)
            metric_label: Custom label for the metric
        """
        results_df = self.get_comparison_results(dataset_type)

        # Sort by CV performance
        results_df = results_df.sort_values('CV_RMSE_mean', ascending=self.task_type == 'classification')

        plt.figure(figsize=(15, 8))

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: CV Performance with error bars
        ax1.barh(results_df['Experiment'], results_df['CV_RMSE_mean'],
                xerr=results_df['CV_RMSE_std'], capsize=5)
        ax1.set_xlabel('CV RMSE')
        ax1.set_title('Cross-Validation Performance (Mean ± Std)')
        ax1.invert_yaxis()

        # Plot 2: Test Metric
        metric_col = f'{dataset_type.title()}_{metric.upper()}'
        if metric_col in results_df.columns:
            ax2.barh(results_df['Experiment'], results_df[metric_col])
            ax2.set_xlabel(metric_label or metric.upper())
            ax2.set_title(f'{dataset_type.title()} Set Performance ({metric.upper()})')
            ax2.invert_yaxis()

        # Plot 3: R2 Score (for regression) or Accuracy (for classification)
        if self.task_type == 'regression':
            metric_plot = f'{dataset_type.title()}_R2'
            ax3.barh(results_df['Experiment'], results_df[metric_plot])
            ax3.set_xlabel('R² Score')
            ax3.set_title(f'{dataset_type.title()} Set R² Score')
        else:
            metric_plot = f'{dataset_type.title()}_Accuracy'
            ax3.barh(results_df['Experiment'], results_df[metric_plot])
            ax3.set_xlabel('Accuracy')
            ax3.set_title(f'{dataset_type.title()} Set Accuracy')
        ax3.invert_yaxis()

        # Plot 4: Runtime
        ax4.barh(results_df['Experiment'], results_df['Runtime_sec'])
        ax4.set_xlabel('Runtime (seconds)')
        ax4.set_title('Training Runtime')
        ax4.invert_yaxis()

        plt.tight_layout()
        plt.savefig(f"results/model_comparison_{dataset_type}.png", dpi=300)
        plt.show()

    def plot_error_analysis(self, experiment_name: str, dataset_type: str = 'test'):
        """
        Plot error analysis for a specific experiment.

        Args:
            experiment_name: Name of the experiment
            dataset_type: Dataset to analyze ('val' or 'test')
        """
        if experiment_name not in self.results:
            print(f"Experiment {experiment_name} not found in results")
            return

        # Get predictions from database
        exp_results = self.results[experiment_name]
        experiment_id = exp_results['experiment_id']

        self.db.cursor.execute('''
            SELECT sample_id, actual_value, predicted_value
            FROM predictions
            WHERE experiment_id = ? AND dataset_type = ?
        ''', (experiment_id, dataset_type))

        pred_data = self.db.cursor.fetchall()
        pred_df = pd.DataFrame(pred_data, columns=['sample_id', 'actual', 'predicted'])

        # Calculate errors
        if self.task_type == 'regression':
            pred_df['error'] = pred_df['actual'] - pred_df['predicted']
            pred_df['abs_error'] = pred_df['error'].abs()
            pred_df['rel_error'] = pred_df['error'] / pred_df['actual'] * 100

            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # 1. Scatter plot
            axes[0,0].scatter(pred_df['actual'], pred_df['predicted'], alpha=0.6)
            axes[0,0].plot([pred_df['actual'].min(), pred_df['actual'].max()],
                          [pred_df['actual'].min(), pred_df['actual'].max()], 'r--')
            axes[0,0].set_xlabel('Actual')
            axes[0,0].set_ylabel('Predicted')
            axes[0,0].set_title(f'{experiment_name} - Actual vs Predicted')

            # 2. Error distribution
            axes[0,1].hist(pred_df['error'], bins=30, alpha=0.7)
            axes[0,1].axvline(x=0, color='r', linestyle='--')
            axes[0,1].set_xlabel('Error (Actual - Predicted)')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].set_title(f'{experiment_name} - Error Distribution')

            # 3. Residuals plot
            axes[1,0].scatter(pred_df['predicted'], pred_df['error'], alpha=0.6)
            axes[1,0].axhline(y=0, color='r', linestyle='--')
            axes[1,0].set_xlabel('Predicted')
            axes[1,0].set_ylabel('Residuals')
            axes[1,0].set_title(f'{experiment_name} - Residuals')

            # 4. Absolute error by percentile
            pred_df['percentile'] = pd.qcut(pred_df['actual'], 10, labels=range(10))
            error_by_percentile = pred_df.groupby('percentile')['abs_error'].mean()
            axes[1,1].plot(error_by_percentile, marker='o')
            axes[1,1].set_xlabel('Actual Value Decile')
            axes[1,1].set_ylabel('Mean Absolute Error')
            axes[1,1].set_title(f'{experiment_name} - Error by Value Range')

        else:  # classification
            # Create confusion matrix and classification report
            cm = self._calculate_confusion_matrix(pred_df['actual'], pred_df['predicted'])

            # Plot confusion matrix
            plt.figure(figsize=(8, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{experiment_name} - Confusion Matrix ({dataset_type})')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig(f"results/{experiment_name}_confusion_matrix.png", dpi=300)
            plt.show()

            return  # For classification, we've already plotted the confusion matrix

        plt.tight_layout()
        plt.savefig(f"results/{experiment_name}_error_analysis.png", dpi=300)
        plt.show()

    def _calculate_confusion_matrix(self, y_true, y_pred):
        """Calculate confusion matrix."""
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(y_true, y_pred)

    def generate_report(self, dataset_type: str = 'test') -> str:
        """
        Generate a comprehensive comparison report.

        Args:
            dataset_type: Dataset to report on

        Returns:
            Report string
        """
        results_df = self.get_comparison_results(dataset_type)

        report = f"""
        Model Comparison Report
        =======================

        Task Type: {self.task_type.upper()}
        Dataset: {dataset_type.upper()} Set
        Total Experiments: {len(results_df)}
        Comparison Date: {pd.Timestamp.now()}

        Ranking by {dataset_type.title()} RMSE/Accuracy:
        ------------------------
        """

        # Sort results
        if self.task_type == 'regression':
            metric_col = f'{dataset_type.title()}_RMSE'
            results_sorted = results_df.sort_values(metric_col, ascending=True)
        else:
            metric_col = f'{dataset_type.title()}_Accuracy'
            results_sorted = results_df.sort_values(metric_col, ascending=False)

        # Add ranking
        for i, (_, row) in enumerate(results_sorted.iterrows(), 1):
            if self.task_type == 'regression':
                report += f"\n{i}. {row['Experiment']} (RMSE: {row[metric_col]:.4f})"
                report += f"\n   CV: {row['CV_RMSE_mean']:.4f} ± {row['CV_RMSE_std']:.4f}"
                report += f"\n   R²: {row[f'{dataset_type.title()}_R2']:.4f}"
                report += f"\n   MAE: {row[f'{dataset_type.title()}_MAE']:.4f}"
            else:
                report += f"\n{i}. {row['Experiment']} (Accuracy: {row[metric_col]:.4f})"
                report += f"\n   CV: {row['CV_RMSE_mean']:.4f} ± {row['CV_RMSE_std']:.4f}"
                report += f"\n   Precision: {row[f'{dataset_type.title()}_Precision']:.4f}"
                report += f"\n   F1: {row[f'{dataset_type.title()}_F1']:.4f}"

        # Best model identifier
        report += f"\n\nBest Model: {results_sorted.iloc[0]['Experiment']}"
        report += f"\nModel Type: {results_sorted.iloc[0]['Model']}"
        report += f"\nEnsemble: {results_sorted.iloc[0]['Ensemble']}"
        report += f"\nRuntime: {results_sorted.iloc[0]['Runtime_sec']}s"

        # Statistical significant tests
        if len(results_df) >= 3:
            report += "\n\nModel Diversity Analysis:\n"
            report += self._analyze_model_diversity(results_df)

        return report

    def _analyze_model_diversity(self, results_df: pd.DataFrame) -> str:
        """Analyze diversity among models."""
        analysis = f"  - Range of {self.task_type} performance: "
        if self.task_type == 'regression':
            min_rmse = results_df['Test_RMSE'].min()
            max_rmse = results_df['Test_RMSE'].max()
            analysis += f"{max_rmse - min_rmse:.4f} RMSE units\n"
        else:
            min_acc = results_df['Test_Accuracy'].min()
            max_acc = results_df['Test_Accuracy'].max()
            analysis += f"{max_acc - min_acc:.4f} accuracy points\n"

        analysis += f"  - Average performance improvement from worst to best: "
        analysis += f"{100 * (1 - min_rmse/max_rmse if self.task_type == 'regression' else (max_acc - min_acc)/min_acc):.1f}%\n"

        fastest = results_df.loc[results_df['Runtime_sec'].idxmin()]
        slowest = results_df.loc[results_df['Runtime_sec'].idxmax()]
        analysis += f"  - Training time range: {fastest['Runtime_sec']}s to {slowest['Runtime_sec']}s\n"
        analysis += f"  - Fastest model: {fastest['Experiment']}\n"

        return analysis

    def get_best_model(self, metric: str = 'rmse', dataset_type: str = 'test'):
        """
        Get the best model based on specified metric.

        Args:
            metric: Metric to optimize
            dataset_type: Dataset to use for selection

        Returns:
            Best experiment configuration and model
        """
        best_exp_id, best_score = self.db.get_best_model(metric, dataset_type)

        if best_exp_id is None:
            print(f"No experiment found for {metric} on {dataset_type} dataset")
            return None, None

        # Get experiment details
        exp_details = self.experiments[
            next(k for k, v in self.results.items() if v['experiment_id'] == best_exp_id)
        ]

        # Load model
        best_model = self.db.load_model_artifact(best_exp_id, 'model')

        return exp_details, best_model

    def compare_models_from_database(self, experiment_ids: List[int]) -> pd.DataFrame:
        """Compare models based on experiment IDs from database."""
        return self.db.compare_experiments(experiment_ids)

    def export_results(self, output_dir: str = "results"):
        """Export all results to CSV files."""
        os.makedirs(output_dir, exist_ok=True)

        # Export comparison results
        for dataset_type in ['val', 'test']:
            results_df = self.get_comparison_results(dataset_type)
            results_df.to_csv(f"{output_dir}/comparison_{dataset_type}.csv", index=False)

        # Export individual experiment results
        for exp_name, results in self.results.items():
            experiment_id = results['experiment_id']
            self.db.export_experiment_results(experiment_id, output_dir)

        print(f"Results exported to {output_dir}")

    def close(self):
        """Close database connection."""
        self.db.close()


# Helper function to create standard comparison suite with all models
def create_standard_comparison_suite(
    task_type: str = 'regression',
    random_state: int = 42
) -> ModelComparisonSuite:
    """
    Create a standard comparison suite with popular models and ensembles.

    Args:
        task_type: 'regression' or 'classification'
        random_state: Random seed

    Returns:
        Configured comparison suite
    """
    suite = ModelComparisonSuite(task_type=task_type, random_state=random_state)

    # Add XGBoost
    xgb_trainer = create_xgboost_trainer(task_type=task_type)
    suite.add_experiment('XGBoost_CV', xgb_trainer, {'n_estimators': 1000})

    # Add LightGBM
    lgb_trainer = create_lightgbm_trainer(task_type=task_type)
    suite.add_experiment('LightGBM_CV', lgb_trainer, {'n_estimators': 1000})

    # Add Stacking Ensemble
    stacking_ensemble = create_stacking_ensemble(
        base_model_names=['xgboost', 'lightgbm'],
        task_type=task_type
    )
    suite.add_experiment('Stacking_Ensemble', stacking_ensemble, {})

    # Add Blending Ensemble
    blending_ensemble = create_blending_ensemble(
        base_model_names=['xgboost', 'lightgbm'],
        task_type=task_type
    )
    suite.add_experiment('Blending_Ensemble', blending_ensemble, {})

    # Add Voting Ensemble
    voting_ensemble = create_voting_ensemble(
        base_model_names=['xgboost', 'lightgbm'],
        task_type=task_type
    )
    suite.add_experiment('Voting_Ensemble', voting_ensemble, {})

    return suite