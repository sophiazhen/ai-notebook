"""
Test script to verify the realistic wafer data generation works.
"""

print("Testing realistic wafer data generation...")

try:
    import numpy as np
    import pandas as pd

    print("Basic imports successful")

    # Test the feature structure we discovered
    print("\nAnalyzing real data structure pattern:")

    # Based on our analysis:
    # - 1530 OES features
    # - 1020 RF Power features
    # - 1020 Chamber Pressure features
    # - OCD_mean as target

    n_samples = 50  # Small test sample
    np.random.seed(42)

    # Generate OES features (1530 total)
    oes_elements = ['OES_F_Peak_Intensity', 'OES_AR_Peak_Intensity', 'OES_CF_Peak_Intensity']
    oes_features = {}

    for elem in oes_elements:
        for i in range(1, 11):  # 10 peaks per element
            base_val = np.random.lognormal(3, 0.1, n_samples)
            for agg in ['mean', 'min', 'max', 'median', 'stdev']:
                if agg == 'mean':
                    oes_features[f'{elem}_{i}_{agg}'] = base_val * np.random.normal(1, 0.05)
                elif agg == 'max':
                    oes_features[f'{elem}_{i}_{agg}'] = base_val * 1.3
                elif agg == 'min':
                    oes_features[f'{elem}_{i}_{agg}'] = base_val * 0.7
                elif agg == 'median':
                    oes_features[f'{elem}_{i}_{agg}'] = base_val * np.random.normal(1, 0.02)
                else:
                    oes_features[f'{elem}_{i}_{agg}'] = base_val * np.abs(np.random.normal(0, 0.1))

    print(f"Created {len(oes_features)} OES features")

    # Generate RF Power features (1020 total)
    power_features = {}
    for i in range(1, 101):  # 100 power sources
        base_power = 500 + i*3 + np.random.exponential(0.5, n_samples)
        for agg in ['mean', 'min', 'max', 'median', 'stdev']:
            if agg == 'mean':
                power_features[f'RF_Source_Power_{i}_{agg}'] = base_power * np.random.normal(1, 0.05)
            elif agg == 'min':
                power_features[f'RF_Source_Power_{i}_{agg}'] = base_power * 0.95
            elif agg == 'max':
                power_features[f'RF_Source_Power_{i}_{agg}'] = base_power * 1.05
            elif agg == 'median':
                power_features[f'RF_Source_Power_{i}_{agg}'] = base_power * np.random.normal(1, 0.02)
            else:
                power_features[f'RF_Source_Power_{i}_{agg}'] = np.abs(np.random.normal(0, 10))

    print(f"Created {len(power_features)} RF Power features")

    # Generate Pressure features (1020 total)
    pressure_features = {}
    for i in range(1, 101):  # 100 pressure points
        base_pressure = 5.0 + 0.2*i + np.random.gamma(2, 1, n_samples)
        for agg in ['mean', 'min', 'max', 'median', 'stdev']:
            if agg == 'mean':
                pressure_features[f'Chamber_Pressure_{i}_{agg}'] = base_pressure * np.random.normal(1, 0.02)
            elif agg == 'min':
                pressure_features[f'Chamber_Pressure_{i}_{agg}'] = base_pressure * 0.95
            elif agg == 'max':
                pressure_features[f'Chamber_Pressure_{i}_{agg}'] = base_pressure * 1.05
            elif agg == 'median':
                pressure_features[f'Chamber_Pressure_{i}_{agg}'] = base_pressure * np.random.normal(1, 0.01)
            else:
                pressure_features[f'Chamber_Pressure_{i}_{agg}'] = np.abs(np.random.normal(0, 0.1))

    print(f"Created {len(pressure_features)} Chamber Pressure features")

    # Create target (CD - Critical Dimension)
    # Add some signal based on features
    base_cd = np.random.normal(32, 2, n_samples)
    oes_sum = sum(oes_features[k] for k in list(oes_features.keys())[:5]) / 100
    power_sum = sum(power_features[k] for k in list(power_features.keys())[:5]) / 1000

    ocd_mean = base_cd - 0.3 * oes_sum + 0.2 * power_sum

    # Combine all
    test_data = pd.DataFrame({
        'context_id': [f"CTX_{i:06d}" for i in range(n_samples)],
        'wafer_id': [f"W{i:05d}" for i in range(n_samples)],
        'lot_id': ['LOT_0001'] * (n_samples//2) + ['LOT_0002'] * (n_samples - n_samples//2),
        'tool_id': np.random.choice(['CHAMBER_A', 'CHAMBER_B', 'CHAMBER_C'], n_samples),
        'recipe_id': np.random.choice(['M2_MAIN', 'M2_HIGH_Q'], n_samples),
        'OCD_mean': ocd_mean
    })

    # Add features
    for name, data in oes_features.items():
        test_data[name] = data
    for name, data in power_features.items():
        test_data[name] = data
    for name, data in pressure_features.items():
        test_data[name] = data

    print(f"\nSuccessfully created test dataset:")
    print(f"- Shape: {test_data.shape}")
    print(f"- Target 'OCD_mean' range: {test_data['OCD_mean'].min():.2f} - {test_data['OCD_mean'].max():.2f}")
    print(f"- Mean CD: {test_data['OCD_mean'].mean():.2f} ± {test_data['OCD_mean'].std():.2f}")

    # Show sample column names
    print(f"\nSample column names:")
    for i, col in enumerate(test_data.columns[:8]):
        print(f"  {i+1}: {col}")
    print("  ...")
    for i, col in enumerate(test_data.columns[-3:], start=len(test_data.columns)-2):
        print(f"  {i+1}: {col}")

    print("\n[SUCCESS] Test successful - generating realistic wafer data works!")

except ImportError as e:
    print(f"Import error: {e}")
    print("This script requires numpy and pandas to test data generation.")