"""
Example script demonstrating the usage of the machine learning framework
for semiconductor wafer virtual metrology.

This example shows how to:
1. Load and preprocess wafer data
2. Train multiple models (XGBoost, LightGBM, Ensemble)
3. Compare results
4. Save to database
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import our ML framework
from src.model_comparison import create_standard_comparison_suite
from wide_table.wafer_data_processor import WaferDataProcessor


def load_sample_data():
    """
    Create sample semiconductor wafer data for demonstration.
    In real use case, this would be loaded from actual Excel files.
    """
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    # Create synthetic wafer data
    # Features representing OES/RGA sensor data across multiple etching steps
    time_points = ['breakthrough', 'main_etch', 'overetch']
    sensors = ['oes_cf', 'oes_ar', 'oes_chf', 'rga_mass1', 'rga_mass2']
    aggregations = ['mean', 'std', 'min', 'max']

    feature_names = []
    for step in time_points:
        for sensor in sensors:
            for agg in aggregations:
                feature_names.append(f"{sensor}_{step}_{agg}")

    # Generate synthetic features (simulating real wafer data)
    X = pd.DataFrame(
        data=np.random.normal(0, 1, size=(n_samples, len(feature_names))),
        columns=feature_names
    )

    # Add some realistic structure to the data
    # OES intensity decreases during etching
    for sensor in sensors:
        for i, step in enumerate(time_points):
            for agg in aggregations:
                col = f"{sensor}_{step}_{agg}"
                X[col] += np.random.normal(loc=(len(time_points) - i) * 0.5, scale=0.2, size=n_samples)

    # Add categorical features (tool ID, process ID)
    tool_ids = ['CHAMBER_A', 'CHAMBER_B', 'CHAMBER_C']
    X['tool_id'] = np.random.choice(tool_ids, n_samples)
    X['recipe'] = np.random.choice(['RECIPE_1', 'RECIPE_2', 'RECIPE_3'], n_samples)

    # Create target variables
    # CD (Critical Dimension) - regression target
    cd_error = np.random.normal(0, 2, n_samples)
    # Add some signal based on features
    cd_error += 0.3 * X['oes_cf_breakthrough_mean'] - 0.5 * X['oes_ar_main_etch_std']
    X['cd_target'] = cd_error

    # Create classification target (good/bad based on CD error threshold)
    cd_threshold = np.percentile(cd_error, 80)  # Top 20% are "bad"
    X['quality_class'] = (cd_error > cd_threshold).astype(int)

    return X, feature_names


def prepare_data_for_ml(data_df, target_col='cd_target'):
    """
    Prepare wafer data for machine learning.

    Args:
        data_df: DataFrame with wafer data
        target_col: Target column name

    Returns:
        X: Features
        y: Target
    """
    # Separate features and target
    X = data_df.drop(columns=[target_col, 'quality_class'])  # Assuming classification target exists
    y = data_df[target_col]

    # Handle categorical variables
    categorical_cols = ['tool_id', 'recipe']
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    return X, y


def main():
    """Main function to run the example."""
    print("=" * 60)
    print("Semiconductor Wafer Virtual Metrology Example")
    print("=" * 60)

    # Load sample data
    print("\n1. Loading sample wafer data...")
    wafer_data, feature_names = load_sample_data()
    print(f"   - Loaded {len(wafer_data)} samples with {len(feature_names)} features")

    # Prepare data for machine learning
    print("\n2. Preparing data for ML...")
    X, y = prepare_data_for_ml(wafer_data)
    print(f"   - Features shape: {X.shape}")
    print(f"   - Target: {y.name}")

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )

    print(f"\n   - Training set: {len(X_train)} samples")
    print(f"   - Validation set: {len(X_val)} samples")
    print(f"   - Test set: {len(X_test)} samples")

    # Create model comparison suite
    print("\n3. Creating model comparison suite...")
    comparison_suite = create_standard_comparison_suite(
        task_type='regression',  # or 'classification' based on your problem
        random_state=42
    )

    # Run experiments
    print("\n4. Running experiments...")
    comparison_suite.run_experiments(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        verbose=True
    )

    # Generate comparison report
    print("\n5. Generating comparison report...")
    report = comparison_suite.generate_report(dataset_type='test')
    print(report)

    # Save report to file
    with open('results/wafer_vm_comparison_report.txt', 'w') as f:
        f.write(report)

    # Plot comparisons
    print("\n6. Plotting model comparisons...")
    comparison_suite.plot_model_comparison(dataset_type='test', metric='rmse')
    comparison_suite.plot_model_comparison(dataset_type='test', metric='r2', metric_label='R² Score')

    # Detailed error analysis for best model
    print("\n7. Error analysis for best model...")
    results_df = comparison_suite.get_comparison_results('test')
    best_model = results_df.iloc[0]['Experiment']
    comparison_suite.plot_error_analysis(best_model, dataset_type='test')

    # Export all results
    print("\n8. Exporting results...")
    comparison_suite.export_results(output_dir='results/wafer_vm_results')

    # Get best model details
    print("\n9. Best model details...")
    best_exp_config, best_model_obj = comparison_suite.get_best_model(metric='rmse', dataset_type='test')
    if best_exp_config:
        print(f"   Best Model: {best_exp_config['model'].model_name}")
        print(f"   Best Experiment: {best_exp_config['experiment_name']}")

    # Access feature importance from database
    print("\n10. Feature importance analysis...")
    experiment_id = comparison_suite.results[best_model]['experiment_id']
    feature_importance_data = comparison_suite.db.get_experiment_results(experiment_id)['feature_importance']

    if feature_importance_data:
        importance_df = pd.DataFrame(
            feature_importance_data,
            columns=['feature_name', 'importance_score', 'rank', 'fold_id']
        ).head(10)
        print("\nTop 10 Most Important Features:")
        print(importance_df[['feature_name', 'importance_score', 'rank']].to_string(index=False))

    # Close database connection
    comparison_suite.close()

    # Print summary
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("Results saved to:")
    print("  - results/wafer_vm_comparison_report.txt")
    print("  - results/model_comparison_test.png")
    print("  - results/wafer_vm_results/")


if __name__ == "__main__":
    main()