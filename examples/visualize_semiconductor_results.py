"""
Visualization script for semiconductor virtual metrology results.
Demonstrates the enhanced analysis capabilities of the optimized framework.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wafer_vm_realistic_example import (
    generate_realistic_wafer_data, prepare_semiconductor_data,
    optimize_algorithms_for_semiconductor
)

def plot_wafer_feature_distribution(data, save_path="results/"):
    """Visualize key semiconductor features and their distributions."""

    # Key feature groups for semiconductor manufacturing
    feature_groups = {
        'OES Plasma Dynamics': {
            'pattern': 'OES.*Mean_1_mean',  # First mean of each OES peak
            'color': 'blue',
            'label': 'Plasma Emission'
        },
        'RF Power Control': {
            'pattern': 'RF_Source_Power_1_mean',
            'color': 'red',
            'label': 'RF Power'
        },
        'Chamber Pressure': {
            'pattern': 'Chamber_Pressure_1_mean',
            'color': 'green',
            'label': 'Chamber Pressure'
        }
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Feature group distributions
    ax1 = axes[0, 0]
    x_features = []
    for group_name, group_info in feature_groups.items():
        pattern = group_info['pattern']
        features = [c for c in data.columns if "AR_M" in c][:5] if "OES" in pattern else [c for c in data.columns if pattern.replace(".*", "") in c][:5]
        if features:
            # Sample from first wafer
            sample_data = data[features].iloc[0]
            ax1.hist(sample_data, alpha=0.7, label=group_name, bins=20)

    ax1.set_title('Key Semiconductor Feature Distributions')
    ax1.set_xlabel('Feature Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    # 2. OES by Element
    ax2 = axes[0, 1]
    oes_elements = ['OES_F_', 'OES_AR_', 'OES_CF_']
    element_means = {}
    for elem in oes_elements:
        element_cols = [c for c in data.columns if elem in c and 'mean' in c][:10]
        if element_cols:
            element_means[elem.replace('OES_', '').replace('_', '')] = data[element_cols].iloc[0].mean()

    if element_means:
        ax2.bar(element_means.keys(), element_means.values(), color=['cyan', 'orange', 'purple'])
        ax2.set_title('OES Intensity by Element (Sample)')
        ax2.set_ylabel('Mean Intensity')

    # 3. Equipment Variation
    ax3 = axes[1, 0]
    if 'tool_id' in data.columns:
        tool_specific = data.groupby('tool_id')['OCD_mean'].agg(['mean', 'std']).reset_index()
        x_pos = range(len(tool_specific))
        axes[1, 0].errorbar(x_pos, tool_specific['mean'],
                           yerr=tool_specific['std'],
                           marker='o', capsize=5, linewidth=2)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(tool_specific['tool_id'])
        ax3.set_title('CD Mean by Tool (±1σ)')
        ax3.set_ylabel('CD (nm)')

    # 4. Recipe Effects
    ax4 = axes[1, 1]
    if 'recipe_id' in data.columns:
        recipe_summary = data.groupby('recipe_id')['OCD_mean'].agg(['mean', 'count']).reset_index()
        recipe_summary.columns = ['Recipe', 'Mean_CD', 'Count']

        bars = ax4.bar(recipe_summary['Recipe'], recipe_summary['Mean_CD'],
                      color=plt.cm.viridis(np.linspace(0, 1, len(recipe_summary))))
        ax4.set_title('CD Mean by Recipe')
        ax4.set_ylabel('Mean CD (nm)')

        # Add count labels
        for bar, count in zip(bars, recipe_summary['Count']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'n={count}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"{save_path}semiconductor_features_overview.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_algorithms_comparison(results_df, save_path="results/"):
    """Plot comparison of different algorithms optimized for semiconductor data."""

    if results_df.empty:
        print("No results to visualize. Run experiments first.")
        return

    # Prepare data for visualization
    viz_data = results_df.copy()
    viz_data['Model_Type'] = viz_data['Experiment'].str.split('_').str[0]
    viz_data['Optimization'] = viz_data['Experiment'].str.split('_').str[1:].str.join('_')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Test RMSE Comparison
    ax1 = axes[0, 0]
    sorted_data = viz_data.sort_values('Test_RMSE', ascending=True)
    bars = ax1.barh(range(len(sorted_data)), sorted_data['Test_RMSE'],
                   color=plt.cm.Viridis(np.linspace(0, 1, len(sorted_data))))

    for i, (idx, row) in enumerate(sorted_data.iterrows()):
        ax1.text(row['Test_RMSE'] + row['Test_RMSE']*0.01, i,
                f"{row['Test_RMSE']:.3f}", va='center', fontsize=9)

    ax1.set_yticks(range(len(sorted_data)))
    ax1.set_yticklabels(sorted_data['Experiment'])
    ax1.set_xlabel('Test RMSE (nm)')
    ax1.set_title('Algorithm Performance (RMSE)')
    ax1.invert_yaxis()

    # 2. R² Score Comparison
    ax2 = axes[0, 1]
    r2_data = viz_data.sort_values('Test_R2', ascending=False)
    bars = ax2.barh(range(len(r2_data)), r2_data['Test_R2'],
                   color=plt.cm.PuRd(np.linspace(0, 1, len(r2_data))))

    for i, (idx, row) in enumerate(r2_data.iterrows()):
        ax2.text(row['Test_R2'] + 0.01, i, f"{row['Test_R2']:.3f}",
                va='center', fontsize=9)

    ax2.set_yticks(range(len(r2_data)))
    ax2.set_yticklabels(r2_data['Experiment'])
    ax2.set_xlabel('Test R² Score')
    ax2.set_title('Model Goodness of Fit (R²)')
    ax2.invert_yaxis()

    # 3. CV vs Test Performance
    ax3 = axes[1, 0]
    ax3.scatter(viz_data['CV_RMSE_mean'], viz_data['Test_RMSE'],
               c=viz_data['Test_R2'], cmap='viridis', s=100, alpha=0.8)

    # Add diagonal line for perfect CV-Test agreement
    max_val = max(viz_data['CV_RMSE_mean'].max(), viz_data['Test_RMSE'].max())
    ax3.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect Agreement')

    ax3.set_xlabel('CV RMSE Mean')
    ax3.set_ylabel('Test RMSE')
    ax3.set_title('CV vs Test Performance\n(Color = R² Score)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Model Type Summary
    ax4 = axes[1, 1]
    model_summary = viz_data.groupby('Model_Type').agg({
        'Test_RMSE': 'mean',
        'Test_R2': 'mean',
        'CV_RMSE_std': 'mean'
    }).round(4)

    # Create summary plot
    x = range(len(model_summary))
    width = 0.25

    ax4.bar([i - width for i in x], model_summary['Test_RMSE'],
           width, label='Test RMSE', color='red', alpha=0.7)
    ax4.bar(x, model_summary['Test_R2'], width,
           label='Test R²', color='blue', alpha=0.7)
    ax4.bar([i + width for i in x], model_summary['CV_RMSE_std'],
           width, label='CV Std', color='orange', alpha=0.7)

    ax4.set_xlabel('Model Type')
    ax4.set_ylabel('Performance Metric')
    ax4.set_title('Average Performance by Model Type')
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_summary.index)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}semiconductor_models_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_engineering_impact(data, engineered_data=None):
    """Show the impact of semiconductor-specific feature engineering."""

    if engineered_data is None:
        from wafer_vm_realistic_example import generate_realistic_wafer_data
        engineered_data = generate_realistic_wafer_data(n_samples=100)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original vs Enhanced features
    if 'F_CF_ratio' in engineered_data.columns:
        ax1 = axes[0, 0]
        ax1.hist(engineered_data['F_CF_ratio'], bins=30, alpha=0.7, color='red', label='F/CF Ratio')
        ax1.set_title('Plasma Ion-to-Neutral Ratio')
        ax1.set_xlabel('F/CF Ratio')
        ax1.set_ylabel('Frequency')

        # Add percentile lines
        p90 = np.percentile(engineered_data['F_CF_ratio'], 90)
        p10 = np.percentile(engineered_data['F_CF_ratio'], 10)
        ax1.axvline(p10, color='green', linestyle='--', alpha=0.7, label='P10')
        ax1.axvline(p90, color='green', linestyle='--', alpha=0.7, label='P90')
        ax1.legend()

    # Power system analysis
    if 'total_rf_power' in engineered_data.columns:
        ax2 = axes[0, 1]
        data_to_plot = engineered_data[
            engineered_data.columns[pd.Series(engineered_data.columns).str.contains('RF_Source_Power_1')]
        ].iloc[0, :5]  # Only first 5 power sources for clarity

        ax2.plot(range(1, len(data_to_plot)+1), data_to_plot.values, 'o-', color='darkblue')
        ax2.set_title('RF Power Profile (Wafer 0)')
        ax2.set_xlabel('Power Source Index')
        ax2.set_ylabel('Power (W)')
        ax2.grid(True, alpha=0.3)

    # Chamber pressure analysis
    if 'pressure_uniformity' in engineered_data.columns:
        ax3 = axes[0, 2]
        ax3.scatter(engineered_data['mean_chamber_pressure'], engineered_data['pressure_uniformity'],
                   alpha=0.6, c=data['OCD_mean'], cmap='viridis')
        ax3.set_xlabel('Mean Chamber Pressure (mTorr)')
        ax3.set_ylabel('Pressure Uniformity')
        ax3.set_title('Pressure Performance vs CD')
        cbar = plt.colorbar(ax3.collections[0], ax=ax3)
        cbar.set_label('CD (nm)')

    # Tool comparison
    ax4 = axes[1, 0]
    if 'tool_id' in data.columns:
        tool_cd = data.groupby('tool_id')['OCD_mean']
        tools = tool_cd.index
        means = tool_cd.values
        ax4.bar(tools, means, color=['red', 'green', 'blue', 'orange'][:len(tools)])
        ax4.set_title('CD by Tool')
        ax4.set_ylabel('Mean CD (nm)')

        # Add count annotations
        tool_counts = data['tool_id'].value_counts()
        for i, (tool, cd_mean) in enumerate(zip(tools, means)):
            ax4.text(i, cd_mean + 0.1, f'n={tool_counts[tool]}', ha='center')

    # Recipe effects
    ax5 = axes[1, 1]
    if 'recipe_id' in data.columns:
        recipe_stats = data.groupby('recipe_id')['OCD_mean'].agg(['mean', 'std', 'count'])
        ax5.bar(recipe_stats.index, recipe_stats['mean'],
               yerr=recipe_stats['std'], capsize=5, alpha=0.7)
        ax5.set_title('Recipe Performance (Mean ± Std)')
        ax5.set_ylabel('CD (nm)')

        # Add count labels
        for i, (recipe, stats) in enumerate(recipe_stats.iterrows()):
            ax5.text(i, stats['mean'] + stats['std'] + 0.2,
                    f'n={stats["count"]}', ha='center', fontsize=10)

    # Quality Classification
    ax6 = axes[1, 2]
    if 'quality_class' in data.columns:
        quality_dist = data['quality_class'].value_counts().sort_index()
        colors = ['red', 'orange', 'green']
        wedges, texts, autotexts = ax6.pie(quality_dist.values,
                                          labels=quality_dist.index,
                                          autopct='%1.1f%%',
                                          colors=colors[:len(quality_dist)])
        ax6.set_title('Quality Classification\n(1=Good, 2=Marginal, 3=Bad)')

    plt.tight_layout()
    plt.savefig(f"{save_path}semiconductor_feature_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Generate example plots demonstrating the framework capabilities."""

    print("Generating semiconductor virtual metrology visualization examples...")

    os.makedirs("results", exist_ok=True)

    # Generate toy data similar to realistic example
    print("Creating sample wafer data...")

    # For demo, we'll simulate what the realistic example would produce
    sample_data = generate_demo_wafer_data(n_samples=200)

    print("Visualizing semiconductor features...")
    plot_wafer_feature_distribution(sample_data)

    # Create demo results DataFrame
    demo_results = pd.DataFrame({
        'Experiment': ['XGBoost_Optimized', 'LightGBM_CategoricalOptimized',
                      'XGBoost_FeatureSelection', 'Stacking_Ensemble',
                      'Blending_Ensemble_Fast', 'XGBoost_Semiconductor',
                      'LightGBM_Semiconductor'],
        'Model': ['XGBoost', 'LightGBM', 'XGBoost', 'Stacking',
                 'Blending', 'XGBoost', 'LightGBM'],
        'Test_RMSE': [0.421, 0.398, 0.445, 0.387, 0.412, 0.395, 0.389],
        'Test_R2': [0.874, 0.888, 0.860, 0.903, 0.879, 0.894, 0.889],
        'CV_RMSE_mean': [0.415, 0.402, 0.438, 0.390, 0.408, 0.401, 0.387],
        'CV_RMSE_std': [0.031, 0.019, 0.028, 0.016, 0.021, 0.018, 0.015]
    })

    print("Visualizing model comparison results...")
    plot_algorithms_comparison(demo_results)

    # Enhance the demo data with engineered features
    print("Visualizing feature engineering impact...")
    plot_feature_engineering_impact(sample_data)

    print("\nVisualization complete!")
    print("Generated plots:")
    print("- results/semiconductor_features_overview.png")
    print("- results/semiconductor_models_comparison.png")
    print("- results/semiconductor_feature_analysis.png")
    print("\nThese demonstrate the capabilities of the optimized framework.")


def generate_demo_wafer_data(n_samples=200):
    """
    Quick demo data generator with semiconductor-specific structure.
    """
    np.random.seed(42)

    # Simple demo structure matching our analysis
    data = pd.DataFrame({
        'context_id': [f"CTX_{i:06d}" for i in range(n_samples)],
        'wafer_id': [f"W{i:05d}" for i in range(n_samples)],
        'lot_id': np.random.choice([f"LOT_{i:04d}" for i in range(20)], n_samples),
        'tool_id': np.random.choice(['CHAMBER_A', 'CHAMBER_B', 'CHAMBER_C'], n_samples),
        'recipe_id': np.random.choice(['M2_MAIN_HIGH', 'M2_MAIN_LOW', 'M2_RAMP'], n_samples),
    })

    # Add some synthetic OES features
    for i in range(1, 6):  # 5 OES measurements
        base_val = np.random.lognormal(3, 0.1, n_samples)
        data[f'OES_F_Peak_Intensity_{i}_mean'] = base_val * np.random.normal(1, 0.05)
        data[f'OES_AR_Peak_Intensity_{i}_mean'] = base_val * np.random.normal(1.2, 0.08)
        data[f'OES_CF_Peak_Intensity_{i}_mean'] = base_val * np.random.normal(0.8, 0.06)

    # Add some power features
    for i in range(1, 6):
        power_base = 500 + i*5 + np.random.normal(0, 20, n_samples)
        data[f'RF_Source_Power_{i}_mean'] = power_base

    # Add pressure features
    for i in range(1, 6):
        pressure_base = 5.0 + i*0.1 + np.random.normal(0, 1, n_samples)
        data[f'Chamber_Pressure_{i}_mean'] = pressure_base

    # Create physics-based target
    oes_avg = data[[c for c in data.columns if 'OES_' in c]].mean(axis=1)
    power_avg = data[[c for c in data.columns if 'RF_Source_Power' in c]].mean(axis=1)
    pressure_avg = data[[c for c in data.columns if 'Chamber_Pressure' in c]].mean(axis=1)

    # Simulate CD with tool effects
    base_cd = np.random.normal(32, 1.5, n_samples)
    tool_effects = {'CHAMBER_A': -0.3, 'CHAMBER_B': 0, 'CHAMBER_C': 0.4}

    data['OCD_mean'] = (base_cd
                       - 0.2 * (oes_avg - oes_avg.mean()) / oes_avg.std()
                       + 0.1 * (power_avg - power_avg.mean()) / power_avg.std()
                       - 0.05 * (pressure_avg - pressure_avg.mean()) / pressure_avg.std()
                       + data['tool_id'].map(tool_effects).fillna(0))

    # Add quality classification
    cd_error = (data['OCD_mean'] - 32.0) / 32.0 * 100
    data['quality_class'] = pd.cut(cd_error,
                                   bins=[-np.inf, -20, -10, 10, 20, np.inf],
                                   labels=[3, 2, 1, 2, 3])

    # Add derived features
    data['F_CF_ratio'] = data['OES_F_Peak_Intensity_1_mean'] / \
                        (data['OES_CF_Peak_Intensity_1_mean'] + 1e-6)
    data['total_rf_power'] = data[[c for c in data.columns if 'RF_Source_Power' in c]].sum(axis=1)
    data['mean_chamber_pressure'] = data[[c for c in data.columns if 'Chamber_Pressure' in c]].mean(axis=1)

    return data


if __name__ == "__main__":
    main()