# AI/ML Notebook - Wafer Virtual Metrology Framework

A comprehensive machine learning framework for semiconductor wafer virtual metrology (VM) that supports automated batch processing, multi-step time series prediction, and ensemble modeling strategies.

## Features

### Core Capabilities
- **Batch Excel Processing**: Automated processing of wafer data from multiple Excel files
- **Multi-step Time Series Analysis**: Support for temporal sequences in wafer processing
- **Wide Table Data Processing**: Convert distributed wafer metrics to centralized wide format
- **Advanced ML Models**: XGBoost, LightGBM, and ensemble strategies
- **Comprehensive Evaluation**: Regression and classification metrics with statistical validation
- **Database Integration**: SQLite-based storage for experiments, metrics, and predictions
- **Model Comparison**: Automated model selection and performance analysis

### Model Support
- **XGBoost**: Optimized for semiconductor data with early stopping and hyperparameter tuning
- **LightGBM**: Native categorical feature handling, efficient for high-dimensional data
- **Ensemble Methods**:
  - Stacking with meta-learner optimization
  - Blending for robust generalization
  - Voting with weight optimization

### Evaluation Metrics (Industry Standard)
For Regression:
- RMSE, MAE, MSE
- R² Score
- MAPE (Mean Absolute Percentage Error)
- Percentage within threshold (P95, P90, P80)
- Huber Loss (robust to outliers)

For Classification:
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC
- Log Loss
- Per-class metrics

## Project Structure

```
ai-notebook/
├── src/                      # Core ML framework
│   ├── base_ml_trainer.py   # Base class for all trainers
│   ├── xgboost_trainer.py   # XGBoost implementation
│   ├── lightgbm_trainer.py  # LightGBM implementation
│   ├── ensemble_trainer.py  # Ensemble strategies (stacking, blending, voting)
│   └── model_comparison.py  # Comprehensive model comparison suite
├── db/                       # Database operations
│   └── database.py          # SQLite database manager
├── wide_table/              # Data preprocessing
│   └── wafer_data_processor.py  # Wide table generation
├── data_insight/            # Data analysis utilities
├── examples/                # Usage examples
│   └── wafer_vm_example.py  # Complete workflow example
├── test/                    # Test suite
└── snippets/               # Reference materials and Q&A
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Basic Usage Example

```python
from src.model_comparison import create_standard_comparison_suite
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your wafer data (CSV/Excel format)
data = pd.read_csv('wafer_data.csv')
X = data.drop(['target'], axis=1)
y = data['target']

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Create comparison suite with standard models
suite = create_standard_comparison_suite(task_type='regression')

# Run experiments
suite.run_experiments(X_train, y_train, X_val, y_val, X_test, y_test)

# Get results
results_df = suite.get_comparison_results('test')
print(results_df)

# Generate report
report = suite.generate_report()
print(report)
```

### Advanced Example with Custom Models

```python
from src.xgboost_trainer import create_xgboost_trainer
from src.lightgbm_trainer import create_lightgbm_trainer
from src.ensemble_trainer import create_stacking_ensemble
from src.model_comparison import ModelComparisonSuite

# Create comparison suite
suite = ModelComparisonSuite(task_type='regression')

# Add custom XGBoost model with optimal parameters
xgb_trainer = create_xgboost_trainer(
    task_type='regression',
    target_metric='rmse',
    cv_folds=5
)
suite.add_experiment('XGBoost_Optimized', xgb_trainer, {
    'n_estimators': 1500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1
})

# Add stacking ensemble
stacking_ensemble = create_stacking_ensemble(
    base_model_names=['xgboost', 'lightgbm'],
    cv_folds=5
)
suite.add_experiment('Advanced_Stacking', stacking_ensemble, {})

# Run comparison
suite.run_experiments(X_train, y_train, X_val, y_val, X_test, y_test)
```

## Data Format

### Input Data Structure
The framework expects wafer data in a tabular format with columns representing:

- **Features**: OES/RGA sensor readings across multiple etching steps
- **Categorical**: Tool ID, Recipe ID
- **Target**: CD target, quality class, etc.

Example feature naming convention:
```
{sensor_type}_{etching_step}_{aggregation_type}
# e.g., oes_cf_main_etch_mean, rga_mass1_overetch_std
```

### Wide Table Format
The `wafer_data_processor.py` converts distributed data into wide format:
```
{parameter_name}_{step}_{aggregation_type}
```

## Database Schema

The SQLite database stores:
- **Experiments**: Model configurations and metadata
- **Metrics**: Performance metrics for train/val/test splits
- **Predictions**: Actual vs predicted values
- **Feature Importance**: Model-specific feature rankings
- **Model Artifacts**: Serialized trained models

## Ensemble Strategies

### Stacking
- Base models generate out-of-fold predictions
- Meta-learner trained on these predictions
- Second-level predictions as final output

### Blending
- Faster alternative to stacking
- Uses holdout set for meta-learner training
- Single training round for base models

### Voting
- Simple weighted average (regression) or majority vote (classification)
- Weights can be optimized based on CV performance
- Minimal computational overhead

## Performance Baselines

Based on competition research (see `snippets/reference.md`):

1. **Tianchi Industrial AI Competition**:
   - GBDT + XGBoost ensemble achieved winning results
   - Feature engineering with tool ID and KNN imputation

2. **ISSM Smart Metrology Challenge**:
   - XGBoost for classification, Neural Networks for continuous prediction
   - OES/SVID data fusion improved accuracy

3. **Recommended Approach**:
   - LightGBM for high-dimensional sparse data
   - XGBoost for precision-critical applications
   - Ensemble methods for robust performance

## Performance Optimization Tips

1. **Data Preprocessing**:
   - Use KNN imputation for missing values
   - Apply robust scaling for sensor data
   - Consider feature selection with genetic algorithms

2. **Model Selection**:
   - LightGBM: Best for high-dimensional data
   - XGBoost: Best for medium-sized datasets
   - Ensemble: Best for robust generalization

3. **Hyperparameter Tuning**:
   - Start with default parameters
   - Use Optuna for systematic optimization
   - Focus on learning rate and tree complexity

4. **Evaluation Strategy**:
   - Use GroupKFold with lot-based grouping
   - Monitor for data leakage (time awareness)
   - Consider online learning for concept drift

## Data Science Best Practices Implemented

1. **Anti-leakage Architecture**:
   - Proper train/validation/test splits
   - Cross-validation with appropriate grouping
   - No test data exposure during training

2. **Reproducibility**:
   - Fixed random seeds throughout
   - Deterministic cross-validation
   - Complete experiment tracking

3. **Scalability**:
   - Parallel processing support
   - Efficient memory usage
   - Database persistence

4. **Interpretability**:
   - Feature importance analysis
   - SHAP-style analysis capabilities
   - Model weight visualization

## Unit Testing

Run the example script to verify installation:
```bash
python examples/wafer_vm_example.py
```

This will:
1. Generate sample wafer data
2. Run all models in the comparison suite
3. Save results to database
4. Generate plots and reports

## Requirements

See `requirements.txt` for complete dependencies. Core requirements:
- Python >= 3.7
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- xgboost, lightgbm

## Usage Notes

1. **Database Connection**: SQLite database is created automatically
2. **Result Storage**: All experiments saved to `ml_experiments.db`
3. **Figures**: Plots saved to `results/` directory
4. **Models**: Serialized models saved alongside for reuse
## Reference Implementation

The framework is based on research from:
- 天池工业AI大赛 (Tianchi Industrial AI Competition)
- ISSM Smart Metrology Challenge
- Bosch Production Line Performance Competition

For detailed technical background, see:
- `snippets/reference.md` - Competition-winning strategies
- `snippets/qa.md` - Best practices for batch Excel processing

## License

This project is part of the AI/ML notebook for semiconductor manufacturing research.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request with detailed description

For questions or suggestions, please refer to the documentation in `snippets/` directory or create an issue.

