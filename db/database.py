import sqlite3
import json
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import pickle
import os


class DatabaseManager:
    """
    Database management class for storing machine learning experiment data.
    Handles SQLite operations for model training results, predictions, and metrics.
    """

    def __init__(self, db_path: str = "ml_experiments.db"):
        """
        Initialize database connection and create tables if they don't exist.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        """Create necessary tables for ML experiments."""
        # Experiments table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                ensemble_type TEXT,
                feature_count INTEGER,
                train_samples INTEGER,
                val_samples INTEGER,
                test_samples INTEGER,
                parameters TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Model performance metrics table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                fold_id INTEGER,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                dataset_type TEXT, -- 'train', 'val', 'test'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        ''')

        # Model predictions table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                sample_id TEXT,
                actual_value REAL,
                predicted_value REAL,
                prediction_probabilities TEXT, -- JSON for classification
                dataset_type TEXT, -- 'train', 'val', 'test'
                fold_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        ''')

        # Feature importance table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_importance (
                importance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                feature_name TEXT,
                importance_score REAL,
                rank INTEGER,
                fold_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        ''')

        # Model artifacts table (for storing serialized models)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_artifacts (
                artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                fold_id INTEGER,
                artifact_name TEXT, -- 'model', 'preprocessor', 'encoder'
                artifact_data BLOB, -- Pickled object
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        ''')

        self.conn.commit()

    def create_experiment(
        self,
        experiment_name: str,
        model_type: str,
        features: List[str],
        train_size: int,
        val_size: int,
        test_size: int,
        parameters: Dict[str, Any],
        ensemble_type: Optional[str] = None
    ) -> int:
        """
        Create a new experiment record.

        Returns:
            experiment_id
        """
        self.cursor.execute('''
            INSERT INTO experiments
            (experiment_name, model_type, ensemble_type, feature_count,
             train_samples, val_samples, test_samples, parameters)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (experiment_name, model_type, ensemble_type, len(features),
              train_size, val_size, test_size, json.dumps(parameters)))

        self.conn.commit()
        return self.cursor.lastrowid

    def save_metrics(
        self,
        experiment_id: int,
        metrics: Dict[str, float],
        dataset_type: str,
        fold_id: Optional[int] = None
    ):
        """Save model metrics to database."""
        for metric_name, metric_value in metrics.items():
            self.cursor.execute('''
                INSERT INTO model_metrics
                (experiment_id, fold_id, metric_name, metric_value, dataset_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (experiment_id, fold_id, metric_name, metric_value, dataset_type))

        self.conn.commit()

    def save_predictions(
        self,
        experiment_id: int,
        predictions: pd.DataFrame,
        dataset_type: str,
        fold_id: Optional[int] = None
    ):
        """Save model predictions to database."""
        required_cols = ['sample_id', 'actual', 'predicted']
        if not all(col in predictions.columns for col in required_cols):
            raise ValueError(f"Predictions dataframe must contain columns: {required_cols}")

        for _, row in predictions.iterrows():
            probs = json.dumps(row.get('probabilities', {}))
            self.cursor.execute('''
                INSERT INTO predictions
                (experiment_id, sample_id, actual_value, predicted_value,
                 prediction_probabilities, dataset_type, fold_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (experiment_id, str(row['sample_id']), row['actual'],
                  row['predicted'], probs, dataset_type, fold_id))

        self.conn.commit()

    def save_feature_importance(
        self,
        experiment_id: int,
        feature_names: List[str],
        importance_scores: List[float],
        fold_id: Optional[int] = None
    ):
        """Save feature importance scores."""
        # Combine and sort features by importance
        feature_importance = list(zip(feature_names, importance_scores))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        for rank, (feature, score) in enumerate(feature_importance, 1):
            self.cursor.execute('''
                INSERT INTO feature_importance
                (experiment_id, feature_name, importance_score, rank, fold_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (experiment_id, feature, float(score), rank, fold_id))

        self.conn.commit()

    def save_model_artifact(
        self,
        experiment_id: int,
        artifact_name: str,
        artifact_object: Any,
        fold_id: Optional[int] = None
    ):
        """Save serialized model or preprocessing object."""
        artifact_data = pickle.dumps(artifact_object)

        self.cursor.execute('''
            INSERT INTO model_artifacts
            (experiment_id, fold_id, artifact_name, artifact_data)
            VALUES (?, ?, ?, ?)
        ''', (experiment_id, fold_id, artifact_name, artifact_data))

        self.conn.commit()

    def get_experiment_results(
        self,
        experiment_id: int
    ) -> Dict[str, Any]:
        """Retrieve complete experiment results."""
        # Get experiment info
        self.cursor.execute('SELECT * FROM experiments WHERE experiment_id = ?', (experiment_id,))
        experiment = self.cursor.fetchone()

        if not experiment:
            return {}

        # Get metrics
        self.cursor.execute('''
            SELECT metric_name, metric_value, dataset_type, fold_id
            FROM model_metrics
            WHERE experiment_id = ?
        ''', (experiment_id,))
        metrics = self.cursor.fetchall()

        # Get feature importance
        self.cursor.execute('''
            SELECT feature_name, importance_score, rank, fold_id
            FROM feature_importance
            WHERE experiment_id = ?
            ORDER BY rank
        ''', (experiment_id,))
        feature_importance = self.cursor.fetchall()

        # Get predictions
        self.cursor.execute('''
            SELECT sample_id, actual_value, predicted_value, dataset_type, fold_id
            FROM predictions
            WHERE experiment_id = ?
        ''', (experiment_id,))
        predictions = self.cursor.fetchall()

        return {
            'experiment': experiment,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'predictions': predictions
        }

    def compare_experiments(
        self,
        experiment_ids: List[int]
    ) -> pd.DataFrame:
        """Compare metrics across multiple experiments."""
        placeholders = ','.join('?' * len(experiment_ids))
        query = f'''
            SELECT e.experiment_name, e.model_type, e.ensemble_type,
                   m.metric_name, m.metric_value, m.dataset_type
            FROM experiments e
            JOIN model_metrics m ON e.experiment_id = m.experiment_id
            WHERE e.experiment_id IN ({placeholders})
            ORDER BY e.experiment_id, m.dataset_type, m.metric_name
        '''

        df = pd.read_sql_query(query, self.conn, params=experiment_ids)
        return df

    def get_best_model(
        self,
        metric_name: str = 'rmse',
        dataset_type: str = 'val'
    ) -> Tuple[int, float]:
        """Get the best model based on a specific metric."""
        # For RMSE/MAE, lower is better; for R2, higher is better
        order = 'ASC' if metric_name.lower() in ['rmse', 'mae', 'mse'] else 'DESC'

        self.cursor.execute(f'''
            SELECT e.experiment_id, m.metric_value
            FROM experiments e
            JOIN model_metrics m ON e.experiment_id = m.experiment_id
            WHERE m.metric_name = ? AND m.dataset_type = ?
            ORDER BY m.metric_value {order}
            LIMIT 1
        ''', (metric_name, dataset_type))

        result = self.cursor.fetchone()
        return result if result else (None, None)

    def load_model_artifact(
        self,
        experiment_id: int,
        artifact_name: str,
        fold_id: Optional[int] = None
    ) -> Optional[Any]:
        """Load a saved model artifact."""
        self.cursor.execute('''
            SELECT artifact_data
            FROM model_artifacts
            WHERE experiment_id = ? AND artifact_name = ? AND fold_id IS ?
        ''', (experiment_id, artifact_name, fold_id))

        result = self.cursor.fetchone()
        if result:
            return pickle.loads(result[0])
        return None

    def export_experiment_results(
        self,
        experiment_id: int,
        output_path: str
    ):
        """Export experiment results to CSV files."""
        os.makedirs(output_path, exist_ok=True)

        # Get results
        results = self.get_experiment_results(experiment_id)

        # Export metrics
        if results['metrics']:
            metrics_df = pd.DataFrame(results['metrics'],
                                    columns=['metric_name', 'metric_value', 'dataset_type', 'fold_id'])
            metrics_df.to_csv(f"{output_path}/metrics_exp_{experiment_id}.csv", index=False)

        # Export feature importance
        if results['feature_importance']:
            importance_df = pd.DataFrame(results['feature_importance'],
                                       columns=['feature', 'importance', 'rank', 'fold_id'])
            importance_df.to_csv(f"{output_path}/feature_importance_exp_{experiment_id}.csv", index=False)

        # Export predictions
        if results['predictions']:
            pred_df = pd.DataFrame(results['predictions'],
                                 columns=['sample_id', 'actual', 'predicted', 'dataset_type', 'fold_id'])
            pred_df.to_csv(f"{output_path}/predictions_exp_{experiment_id}.csv", index=False)

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()