"""
Realistic example demonstrating M2 trench etching virtual metrology
Based on actual wafer data structure from processed_wide_table.xlsx

This example creates synthetic data that matches the real OCD (Optical Critical Dimension)
prediction use case for semiconductor M2 metal etching.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import our ML framework
from src.model_comparison import create_standard_comparison_suite, ModelComparisonSuite
from src.xgboost_trainer import create_xgboost_trainer
from src.ensemble_trainer import create_stacking_ensemble, create_blending_ensemble
from src.xgboost_semi_trainer import create_xgboost_semi_trainer
from src.lightgbm_semi_trainer import create_lightgbm_semi_trainer


def generate_realistic_wafer_data(n_samples: int = 500, random_state: int = 42):
    """
    Generate synthetic M2 etching wafer data matching real structure.
    Based on analysis of processed_wide_table.xlsx containing:
    - 1530 OES columns (spectroscopy intensity)
    - 1020 Power columns (RF source power)
    - 1020 Pressure columns (chamber pressure)
    - Target: OCD_mean (critical dimension)
    """
    np.random.seed(random_state)

    print("Generating realistic M2 etching wafer data...")
    print(f"Creating {n_samples} samples...")

    # Create metadata columns
    metadata = {
        'context_id': [f"CTX_{i:06d}" for i in range(n_samples)],
        'wafer_id': [f"W{i:05d}" for i in range(n_samples)],
        'lot_id': np.random.choice([f"LOT_{i:04d}" for i in range(50)], n_samples),
        'tool_id': np.random.choice(['CHAMBER_A', 'CHAMBER_B', 'CHAMBER_C'], n_samples),
        'recipe_id': np.random.choice(['M2_MAIN_HIGH', 'M2_MAIN_LOW', 'M2_RAMP'], n_samples),
        'product_type': np.random.choice(['M2_M1_TRENCH', 'M2_M1_VIA', 'M2_M1_BRIDGE'], n_samples),
        'wafer_diameter_mm': 300 + np.random.normal(0, 2, n_samples)
    }

    # Build features following actual wafer data pattern
    features = {}

    # 1. OES (Optical Emission Spectroscopy) features - 1530 columns total
    oes_elements = ['F', 'Ar', 'CF', 'CF2', 'C2', 'Si', 'O']  # Plasma species
    oes_peaks = list(range(1, 11))  # 10 different spectral peaks per element
    agg_methods = ['mean', 'min', 'max', 'median', 'stdev']  # 5 aggregations
    oes_steps = ['breakthrough', 'main_etch', 'overetch']  # 3 etching steps (inferred)

    n_oes_features = len(oes_elements) * len(oes_peaks) * len(agg_methods) * len(oes_steps)  # 1050

    for element in oes_elements:
        for peak in oes_peaks:
            base_intensity = np.random.lognormal(3, 0.5, n_samples)  # Typical plasma intensity
            for step in oes_steps:
                step_factor = 1.2 if step == 'breakthrough' else (1.0 if step == 'main_etch' else 0.8)
                for agg in agg_methods:
                    # Simulate sensor behavior during etching
                    if agg == 'mean':
                        value = base_intensity * step_factor * (0.8 + np.random.normal(0, 0.1))
                    elif agg == 'max':
                        value = base_intensity * step_factor * 1.5
                    elif agg == 'min':
                        value = base_intensity * step_factor * 0.5
                    elif agg == 'median':
                        value = base_intensity * step_factor * (0.9 + np.random.normal(0, 0.05))
                    else:  # stdev
                        value = base_intensity * step_factor * 0.1 * np.random.exponential(1)

                    features[f'OES_{element}_Peak_Intensity_{peak}_{agg}'] = value

    # Add more OES features (total 1530 as in real data)
    remaining_oes = 1530 - len([k for k in features.keys() if 'OES_' in k])

    # Additional OES measurements (different emission lines, different regions)
    ose_wavelengths = ['450nm', '500nm', '600nm', '650nm', '700nm']
    ose_regions = ['center', 'edge', 'uniformity']

    for i in range(remaining_oes // 5):  # Add in groups of 5 (agg methods)
        wavelength = np.random.choice(ose_wavelengths)
        region = np.random.choice(ose_regions)
        element = np.random.choice(oes_elements)

        base_value = np.random.normal(2, 0.5, n_samples)
        for agg in agg_methods:
            if agg == 'mean':
                value = base_value + np.random.normal(0, 0.1)
            elif agg == 'max':
                value = base_value * 1.3
            elif agg == 'min':
                value = base_value * 0.7
            elif agg == 'median':
                value = base_value + np.random.normal(0, 0.05)
            else:
                value = np.abs(np.random.normal(0, base_value/10))

            features[f'OES_{wavelength}_{region}_{element}_Intensity_{agg}'] = value

    # 2. RF Power features (1020 columns) - RF source power measurements
    # Based on real pattern: RF_Source_Power_{1,2,3,...}_{mean,min,max,median,stdev}
    n_rf_sources = 30  # Total RF sources per chamber
    rf_steps = ['ignition', 'stabilization', 'etching', 'cleaning']

    for source_num in range(1, n_rf_sources + 1):
        # Base power (watts) - realistic semiconductor ranges
        base_power = 500 + source_num * 10 + np.random.exponential(0.5, n_samples)

        for step in rf_steps:
            if step == 'ignition':
                power = base_power * 1.5  # Higher power for ignition
            elif step == 'stabilization':
                power = base_power * 1.1
            elif step == 'cleaning':
                power = base_power * 0.9
            else:  # etching
                power = base_power + np.sin(source_num/10) * 50

            # Add noise based on equipment specifications (±5% variations)
            noise_std = power * 0.05

            for agg in agg_methods:
                if agg == 'mean':
                    value = power + np.random.normal(0, noise_std)
                elif agg == 'min':
                    value = power * 0.95 - np.random.exponential(noise_std/2)
                elif agg == 'max':
                    value = power * 1.05 + np.random.exponential(noise_std/2)
                elif agg == 'median':
                    value = power + np.random.laplace(0, noise_std/2)
                else:  # stdev
                    value = np.random.normal(noise_std, noise_std/4)

                features[f'RF_Source_Power_{source_num}_{agg}'] = np.clip(value, 100, 2000)

    # 3. Chamber Pressure features (1020 columns) - Based on actual data
    pressure_cells = 30  # Number of pressure measurement points

    for cell in range(1, pressure_cells + 1):
        # Base pressure (mTorr) - typical semiconductor etching pressures
        base_pressure = 5.0 + cell * 0.3 + np.random.gamma(2, 1, n_samples)

        for agg in agg_methods:
            if agg == 'mean':
                value = base_pressure + np.random.normal(0, base_pressure/20)
            elif agg == 'min':
                value = base_pressure * 0.8
            elif agg == 'max':
                value = base_pressure * 1.2 + np.random.exponential(0.5)
            elif agg == 'median':
                value = base_pressure + np.random.laplace(0, base_pressure/30)
            else:  # stdev - pressure instability measure
                value = np.abs(np.random.normal(0.1, base_pressure/50))

            features[f'Chamber_Pressure_{cell}_{agg}'] = np.clip(value, 1, 100)

    # Create the target variable (OCD - Optical Critical Dimension)
    # This is what we want to predict - the actual physical feature size

    # Combine metadata and features
    wafer_data = pd.DataFrame(metadata)

    # Add features
    for col, vals in features.items():
        wafer_data[col] = vals

    # Create target variable based on realistic semiconductor physics
    print("Creating OCD target based on physics-based model...")

    # Base OCD target (nm) - realistic M2 metal line width
    base_cd = np.random.normal(32, 2, n_samples)  # Typical 32nm CD variation

    # OES determines etch rate (ion density)
    oes_intensity = wafer_data[[c for c in wafer_data.columns if 'OES_' in c and c.endswith('_mean')]].mean(axis=1)
    etch_rate_factor = -0.3 * (oes_intensity - oes_intensity.median()) / 100  # Negative correlation

    # Power affects selective etching
    rf_power = wafer_data[[c for c in wafer_data.columns if 'RF_Source_Power_' in c and c.endswith('_mean')]].mean(axis=1)
    selectivity_factor = -0.15 * (rf_power - rf_power.median()) / 500

    # Pressure affects etch uniformity
    pressure = wafer_data[[c for c in wafer_data.columns if 'Chamber_Pressure_' in c and c.endswith('_mean')]].mean(axis=1)
    uniformity_factor = -0.1 * (pressure - pressure.median()) / 10

    # Add tool effects (systematic variation)
    tool_effects = {'CHAMBER_A': -0.5, 'CHAMBER_B': 0, 'CHAMBER_C': 0.8}
    tool_factor = pd.Series(wafer_data['tool_id'].map(tool_effects)).fillna(0)

    # Final CD calculation
    wafer_data['OCD_mean'] = base_cd + etch_rate_factor + selectivity_factor + uniformity_factor + tool_factor
    wafer_data['OCD_mean'] += np.random.normal(0, 0.5, n_samples)  # Random error

    # Create quality class for classification task
    # Define specs: ±10% is acceptable, ±10-20% marginal, >20% fails
    nominal_cd = 32.0
    cd_error_pct = (wafer_data['OCD_mean'] - nominal_cd) / nominal_cd * 100
    wafer_data['quality_class'] = pd.cut(cd_error_pct,
                                        bins=[-np.inf, -20, -10, 10, 20, np.inf],
                                        labels=[3, 2, 1, 2, 3])  # 1=good, 2=margin, 3=bad

    # Also create numerical quality score
    wafer_data['cd_error_nm'] = np.abs(wafer_data['OCD_mean'] - nominal_cd)

    print(f"\\nGenerated wafer data structure:")
    print(f"- Total samples: {len(wafer_data)}")
    print(f"- Total features: {len(wafer_data.columns) - len(metadata)}")
    print(f"- OES features: {len([c for c in wafer_data.columns if c.startswith('OES_')])}")
    print(f"- RF Power features: {len([c for c in wafer_data.columns if 'RF_Source_Power_' in c])}")
    print(f"- Pressure features: {len([c for c in wafer_data.columns if 'Chamber_Pressure_' in c])}")

    print(f"\\nTarget variable summary:")
    print(f"OCD_mean range: {wafer_data['OCD_mean'].min():.2f} - {wafer_data['OCD_mean'].max():.2f} nm")
    print(f"OCD_mean mean (±std): {wafer_data['OCD_mean'].mean():.2f} ± {wafer_data['OCD_mean'].std():.2f} nm")

    print(f"\\nQuality distribution:")
    print(wafer_data['quality_class'].value_counts().sort_index())

    # Add some more metadata columns matching real structure
    wafer_data['process_start_time'] = pd.Timestamp('2023-01-01') + pd.to_timedelta(range(n_samples), unit='h')
    wafer_data['process_end_time'] = wafer_data['process_start_time'] + pd.to_timedelta(np.random.exponential(2, n_samples), unit='h')
    wafer_data['process_duration_sec'] = np.random.normal(3600, 300, n_samples)  # Typical 1-hour process

    return wafer_data


def prepare_semiconductor_data(data_df, target_col='OCD_mean', classification_target=None):
    """
    Prepare semiconductor wafer data for ML, handling tool and categorical variables.

    Args:
        data_df: DataFrame with wafer data
        target_col: Target column name (OCD_mean for regression)
        classification_target: Additional classification target if needed

    Returns:
        X: Features
        y: Target variable
        processor_info: Dict with preprocessing info
    """
    print("\\nPreparing semiconductor data for ML...")

    # Separate features and target
    metadata_cols = ['context_id', 'wafer_id', 'lot_id', 'tool_id', 'recipe_id',
                    'product_type', 'process_start_time', 'process_end_time',
                    'process_duration_sec', 'wafer_diameter_mm']

    # Also remove operator_id and chamber_id columns that might be present
    removable_cols = ['operator_id', 'chamber_id', 'chamber_id_x', 'operator_id_x', 'fdc_version',
                      'data_logging_interval_ms'] + metadata_cols

    # Get feature columns
    feature_cols = [c for c in data_df.columns if c not in removable_cols]

    print(f"\\nData preprocessing summary:")
    print(f"- Total features before preprocessing: {len(feature_cols)}")
    print(f"- Target column: {target_col}")
    print(f"- Removing metadata columns: {len([c for c in metadata_cols if c in data_df.columns])}")

    # Separate features and target
    X_df = data_df[feature_cols].copy()
    y = data_df[target_col].copy()

    # Log transform some skewed features (common in sensor data)
    skewed_features = [c for c in X_df.columns if 'Intensity' in c and 'stdev' not in c]
    print(f"- Log-transforming {len(skewed_features)} intensity features...")
    for col in skewed_features:
        X_df[col] = np.log1p(X_df[col].abs())

    # Handle categorical variables (match real preprocessing)
    categorical_cols = ['tool_id', 'recipe_id', 'product_type']
    categorical_encoders = {}

    print(f"- Encoding categorical variables: {categorical_cols}")
    for col in categorical_cols:
        if col in X_df.columns:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))
            categorical_encoders[col] = le

    # Clip extreme outliers (semiconductor data has physical limits)
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns
    print(f"- Clipping outliers in {len(numeric_cols)} numeric features...")
    for col in numeric_cols:
        q1, q3 = X_df[col].quantile([0.01, 0.99])
        if X_df[col].std() > 0:  # Only clip if there's variation
            X_df[col] = np.clip(X_df[col], q1, q3)

    # Add derived features for semiconductor physics
    # These are known to be important for etch process control
    print("- Adding physics-based derived features...")

    # 1. OES ratios (ion-to-neutral ratios)
    oes_mean_cols = [c for c in X_df.columns if c.startswith('OES_') and c.endswith('_mean')]
    if len(oes_mean_cols) > 10:
        # Calculate F/CF ratio (fluorine concentration indicator)
        f_cols = [c for c in oes_mean_cols if '_F_' in c][:3]
        cf_cols = [c for c in oes_mean_cols if '_CF_' in c][:3]
        if f_cols and cf_cols:
            X_df['F_CF_ratio'] = X_df[f_cols].mean(axis=1) / (X_df[cf_cols].mean(axis=1) + 1e-6)

        # Calculate OES selectivity (anisotropy indicator)
        X_df['OES_selectivity'] = X_df[oes_mean_cols[:10]].std(axis=1) / (X_df[oes_mean_cols[:10]].mean(axis=1) + 1e-6)

    # 2. Power efficiency
    power_mean_cols = [c for c in X_df.columns if 'RF_Source_Power' in c and c.endswith('_mean')]
    pressure_mean_cols = [c for c in X_df.columns if 'Chamber_Pressure' in c and c.endswith('_mean')]

    if power_mean_cols and pressure_mean_cols:
        X_df['Power_efficiency'] = X_df[power_mean_cols].mean(axis=1) / (X_df[pressure_mean_cols].mean(axis=1) + 1)
        X_df['Power_stability'] = X_df[power_mean_cols].std(axis=1) / (X_df[power_mean_cols].mean(axis=1) + 1)

    # 3. Process stability indicators
    for prefix in ['OES', 'RF_Source_Power', 'Chamber_Pressure']:
        cols = [c for c in X_df.columns if c.startswith(prefix) and c.endswith('_stdev')]
        if len(cols) > 5:
            X_df[f'{prefix}_stability'] = X_df[cols[:5]].mean(axis=1)

    X = X_df.values

    processor_info = {
        'n_features': len(X_df.columns),
        'n_samples': len(X_df),
        'categorical_encoders': categorical_encoders,
        'target_description': f'{target_col} (nm)'
    }

    print(f"\\nFinal feature matrix: {X.shape}")

    return X, y, processor_info, X_df  # Also return the dataframe for debugging


def optimize_algorithms_for_semiconductor():
    """
    Return optimized parameters for semiconductor wafer data based on industry best practices.
    """

    # XGBoost parameters optimized for wafer data (based on reference materials)
    xgb_params = {
        'regression': {
            'n_estimators': 800,
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'colsample_level': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 3,
            'gamma': 0.1,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'early_stopping_rounds': 50,
            'random_state': 42
        },
        'classification': {
            'n_estimators': 800,
            'max_depth': 5,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.8,
            'min_child_weight': 5,
            'gamma': 0.2,
            'objective': 'multi:softprob',
            'eval_metric': 'logloss',
            'early_stopping_rounds': 30,
            'random_state': 42
        }
    }

    # LightGBM parameters for semiconductor data
    lgb_params = {
        'regression': {
            'n_estimators': 1000,
            'num_leaves': 31,
            'learning_rate': 0.02,
            'max_depth': -1,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.2,
            'reg_lambda': 0.8,
            'min_child_samples': 10,
            'min_child_weight': 0.001,
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'random_state': 42,
            'verbosity': -1
        }
    }

    # Ensemble parameters - optimized based on semiconductor insights
    ensemble_params = {
        'stacking': {
            'second_level_model': 'ridge',
            'stack_probabilities': False  # Use predictions, not probabilities for wafer data
        },
        'blending': {
            'test_size': 0.25,
            'holdout_size': 0.3
        }
    }

    return xgb_params, lgb_params, ensemble_params


def create_semiconductor_optimized_suite(random_state=42):
    """
    Create a model comparison suite optimized specifically for semiconductor wafer data.
    """
    suite = ModelComparisonSuite(task_type='regression', random_state=random_state)

    # Optimized parameters for semiconductor data
    xgb_params, lgb_params, ensemble_params = optimize_algorithms_for_semiconductor()

    # 1. Standard XGBoost optimized
    xgb_trainer = create_xgboost_trainer(
        task_type='regression',
        target_metric='rmse',
        cv_folds=5,
        random_state=random_state
    )
    xgb_trainer.early_stopping_rounds = xgb_params['regression']['early_stopping_rounds']
    suite.add_experiment('XGBoost_Optimized', xgb_trainer, xgb_params['regression'])

    # 2. LightGBM optimized for categorical (wafer machine)
    lgb_trainer = create_lightgbm_trainer(
        task_type='regression',
        target_metric='rmse',
        cv_folds=5,
        random_state=random_state,
        use_categorical=True  # Important for tool_id, recipe_id
    )
    suite.add_experiment('LightGBM_CategoricalOptimized', lgb_trainer, lgb_params['regression'])

    # 3. XGBoost with feature selection (Tianchi competition approach)
    xgb_fs_trainer = create_xgboost_trainer(
        task_type='regression',
        target_metric='rmse',
        cv_folds=5,
        random_state=random_state
    )
    xgb_fs_params = xgb_params['regression'].copy()
    xgb_fs_params.update({
        'max_depth': 5,
        'colsample_bytree': 0.6,  # More aggressive selection like genetic algorithm
        'colsample_bylevel': 0.8,
        'colsample_bynode': 0.8,
        'min_child_weight': 5,  # Select stable features
        'reg_lambda': 1.2,  # Stronger regularization
        'reg_alpha': 0.3,
    })
    suite.add_experiment('XGBoost_FeatureSelection', xgb_fs_trainer, xgb_fs_params)

    # 4. Stacking ensemble with hyperparameter tuning
    stacking_ensemble = create_stacking_ensemble(
        base_model_names=['xgboost', 'lightgbm'],
        cv_folds=5,
        random_state=random_state
    )
    stacking_ensemble.second_level_model = ensemble_params['stacking']['second_level_model']
    stacking_ensemble.stack_probabilities = ensemble_params['stacking']['stack_probabilities']
    suite.add_experiment('Stacking_Ensemble', stacking_ensemble, {})

    # 5. Blending ensemble for faster production deployment
    blending_ensemble = create_blending_ensemble(
        base_model_names=['xgboost', 'lightgbm'],
        random_state=random_state
    )
    suite.add_experiment('Blending_Ensemble_Fast', blending_ensemble, ensemble_params['blending'])

    # 6. Custom XGBoost optimized for wafer data
    xgb_semi_trainer = create_xgboost_semi_trainer(
        task_type='regression',
        target_metric='rmse',
        cv_folds=5,
        random_state=random_state,
        handle_high_dimensionality=True
    )
    suite.add_experiment('XGBoost_Semiconductor', xgb_semi_trainer, {})

    # 7. LightGBM with direct categorical handling
    lgb_semi_trainer = create_lightgbm_semi_trainer(
        task_type='regression',
        target_metric='rmse',
        cv_folds=5,
        random_state=random_state,
        handle_categorical=True
    )
    suite.add_experiment('LightGBM_Semiconductor', lgb_semi_trainer, {})

    return suite


def main():
    """Main function with realistic wafer data experiment."""

    print("=" * 70)
    print(" M2 TRENCH ETCHING VIRTUAL METROLOGY - REALISTIC EXAMPLE")
    print("=" * 70)
    print("Based on actual wafer data structure analysis:")
    print("- 1530 OES spectroscopy features")
    print("- 1020 RF power control features")
    print("- 1020 Chamber pressure features")
    print("- Target: OCD_mean (Optical Critical Dimension)")
    print("=" * 70)

    # Generate realistic data (smaller for speed - increase as needed)
    n_samples = 800  # Can increase to 1000+ for real scenarios
    wafer_data = generate_realistic_wafer_data(n_samples=n_samples)

    # Prepare data for ML
    X, y, processor_info, X_df = prepare_semiconductor_data(wafer_data)

    print(f"\\nData preparation complete.")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target: OCD_mean (mean={y.mean():.2f}, std={y.std():.2f}) nm")

    # Split data strategically for semiconductor data (avoid lot leakage)
    unique_lots = wafer_data['lot_id'].unique()
    train_lots, temp_lots = train_test_split(unique_lots, test_size=0.4, random_state=42)
    val_lots, test_lots = train_test_split(temp_lots, test_size=0.5, random_state=42)

    train_mask = wafer_data['lot_id'].isin(train_lots)
    val_mask = wafer_data['lot_id'].isin(val_lots)
    test_mask = wafer_data['lot_id'].isin(test_lots)

    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

    print(f"\\nLot-based splitting (prevents data leakage):")
    print(f"Training lots: {len(train_lots)} ({X_train.shape[0]} samples)")
    print(f"Validation lots: {len(val_lots)} ({X_val.shape[0]} samples)")
    print(f"Test lots: {len(test_lots)} ({X_test.shape[0]} samples)")

    # Create optimized comparison suite for semiconductor data
    print("\\n" + "="*50)
    print("Creating optimized model comparison suite...")
    print("="*50)

    comparison_suite = create_semiconductor_optimized_suite(random_state=42)

    # Run experiments
    print("\\nRunning experiments with semiconductor-optimized pipeline...")
    comparison_suite.run_experiments(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        verbose=True
    )

    # Analyze results
    print("\\n" + "="*50)
    print("EXPERIMENT RESULTS COMPARISON")
    print("="*50)

    # Get comparison results
    results_df = comparison_suite.get_comparison_results('test')
    print("\nPerformance Summary (Test Set):")
    print(results_df[['Experiment', 'Model', 'Test_RMSE', 'Test_R2', 'CV_RMSE_mean']].round(4))

    # Detailed analysis
    best_model = results_df.iloc[0]['Experiment']
    print(f"\\nBest model: {best_model}")
    print(f"Test RMSE: {results_df.iloc[0]['Test_RMSE']:.4f}")
    print(f"Test R²: {results_df.iloc[0]['Test_R2']:.4f}")

    # Generate comprehensive report
    report = comparison_suite.generate_report(dataset_type='test')
    print("\\n" + "="*50)
    print("COMPREHENSIVE ANALYSIS REPORT")
    print("="*50)
    print(report)

    # Save results
    comparison_suite.export_results(output_dir='results/semiconductor_vm_results')

    with open('results/semiconductor_vm_analysis.txt', 'w') as f:
        f.write(report)

    # Feature importance analysis
    print("\\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)

    if hasattr(comparison_suite.db, 'get_experiment_results'):
        best_exp_id = comparison_suite.results[best_model]['experiment_id']
        exp_results = comparison_suite.db.get_experiment_results(best_exp_id)

        if exp_results['feature_importance']:
            importance_df = pd.DataFrame(
                exp_results['feature_importance'][:10],
                columns=['feature_name', 'importance_score', 'rank', 'fold_id']
            )
            print("\\nTop 10 Most Important Features:")
            print(importance_df[['feature_name', 'importance_score']].to_string(index=False))

            # Categorize top features
            oes_important = [f for f in importance_df['feature_name'] if 'OES_' in f]
            power_important = [f for f in importance_df['feature_name'] if 'RF_Source_Power_' in f]
            pressure_important = [f for f in importance_df['feature_name'] if 'Chamber_Pressure_' in f]

            print(f"\\nFeature category breakdown:")
            print(f"- OES spectroscopy: {len(oes_important)}")
            print(f"- RF power: {len(power_important)}")
            print(f"- Chamber pressure: {len(pressure_important)}")

    # Close database connection
    comparison_suite.close()

    print("\\n" + "="*70)
    print(" SEMICONDUCTOR VM ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to:")
    print(f"- Database: ml_experiments.db")
    print(f"- Plots: results/semiconductor_vm_results/")
    print(f"- Report: results/semiconductor_vm_analysis.txt")
    print("="*70)


if __name__ == "__main__":
    main()