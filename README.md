# BME6938-Spring-2026-Project-1-Group-1
# Cardiac Arrhythmia Classification with scikit-learn

## Project Overview

This project implements a machine learning pipeline for multi-class classification of cardiac arrhythmias from electrocardiogram (ECG) data. The goal is to develop and compare multiple scikit-learn algorithms to predict arrhythmia types with high accuracy, supporting early detection and patient safety in clinical settings.

## Problem Statement

Cardiac arrhythmias are irregular heartbeats that can range from benign to life-threatening. Early and accurate detection is crucial for patient outcomes. This project addresses a **16-class classification problem**:
- **Class 1**: Normal heart rhythm
- **Classes 2-15**: Different types of arrhythmias (e.g., atrial fibrillation, ventricular tachycardia, etc.)
- **Class 16**: Unclassified

The objective is to minimize the difference between a cardiologist's classification and the machine learning model's predictions using ECG measurements and patient demographics.

## Data Source

**Dataset**: OpenML Cardiac Arrhythmia (Dataset #5)
Link: https://www.openml.org/d/5

**Specifications**:
- **Samples**: 452 patient ECG recordings
- **Features**: 279 total features
	- 206 numeric features (ECG measurements in milliseconds and millivolts)
	- 73 nominal binary flags (presence/absence of specific ECG patterns)
- **Types of measurements**:
	- Heart rate, PR interval, QRS duration, QT interval
	- P wave, T wave, ST segment characteristics
	- Diagnostic codes from standard ECG interpretation
- **Target**: 16 arrhythmia classes

## Environment Setup

### Requirements

- **Python**: version 3.8 or higher
- **Key dependencies**:
	- scikit-learn (>=1.0.0) — ML algorithms and model evaluation
	- pandas (>=1.0.0) — data manipulation and analysis
	- numpy (>=1.19.0) — numerical computations
	- matplotlib (>=3.2.0) — static visualizations
	- seaborn (>=0.11.0) — statistical data visualization
	- scipy (>=1.5.0) — ARFF file parsing and scientific computing
	- jupyter (>=1.0.0) — interactive notebooks
	- arff (>=0.9) — ARFF file format support

### Installation

1. Clone the repository:
	 ```bash
	 git clone <repository-url>
	 cd BME6938-Spring-2026-Project-1-Group-1
	 ```

2. Install dependencies:
	 ```bash
	 pip install -r requirements.txt
	 ```

3. Verify installation:
	 ```bash
	 python -c "import sklearn; print(f'scikit-learn {sklearn.__version__}')"
	 ```

## Quick Start

### 3-Step Example to Run Full Pipeline

```bash
# Step 1: Explore data and preprocessing decisions
jupyter notebook notebooks/EDA.ipynb

# Step 2: Train all 5 algorithms with cross-validation
python scripts/train_pipeline.py

# Step 3: View results and model comparison
jupyter notebook notebooks/results.ipynb
```

**Expected runtime**: 
- EDA notebook: ~5-10 minutes
- Training pipeline: ~30-60 minutes (depends on hardware)
- Results notebook: ~2-5 minutes

## Usage Instructions

### 1. Exploratory Data Analysis (EDA)

Launch the EDA notebook to understand the dataset and inform modeling decisions:

```bash
jupyter notebook notebooks/EDA.ipynb
```

**Contents**:
- Dataset overview (size, feature types, class distribution)
- Missingness analysis (% missing values per feature)
- Class imbalance patterns
- Feature distributions (histograms, boxplots)
- Correlation analysis
- Feature importance heuristics
- Key insights and implications for preprocessing

### 2. Training

Run the complete training pipeline to train all 5 models with hyperparameter tuning:

```bash
python scripts/train_pipeline.py
```

**What it does**:
1. Loads and preprocesses ARFF data
2. Splits into train (72.25%), validation (12.75%), and test (15%)
3. Encodes categorical features and scales numeric features
4. Trains 5 algorithms with 3-fold stratified cross-validation for hyperparameter tuning
5. Evaluates on held-out test set
6. Saves trained models and results

**Output files**:
- `models/` — trained model pickles (.pkl files)
- `results/metrics.json` — performance metrics and hyperparameters
- `results/performance_comparison.csv` — comparison table

### 3. Results & Visualization

View performance comparison and model rankings:

```bash
jupyter notebook notebooks/results.ipynb
```

**Contents**:
- Performance comparison table (accuracy, F1, precision, recall)
- Confusion matrix heatmap for best model
- Per-class F1 score comparison across algorithms
- Model ranking by macro F1
- Best model vs. baseline comparison
- Interpretation of results

### 4. Demo & Inference

Run the demo notebook to see practical usage and predictions:

```bash
jupyter notebook notebooks/demo.ipynb
```

**Contents**:
- Load best-performing model
- Sample inference on 5-10 test instances
- Feature importance visualization (if applicable)
- Example of correct and incorrect predictions
- Usage instructions for new data

## Repository Structure

```
BME6938-Spring-2026-Project-1-Group-1/
├── data/
│   ├── dataset_5_arrhythmia.arff          # Raw ARFF dataset
│   └── [preprocessed versions if saved]
│
├── src/
│   ├── __init__.py                        # Package initialization
│   ├── data_loader.py                     # ARFF loading & train/val/test splitting
│   ├── preprocessing.py                   # Feature scaling & encoding
│   ├── models.py                          # Model training with hyperparameter tuning
│   └── evaluation.py                      # Metrics computation & visualization
│
├── notebooks/
│   ├── EDA.ipynb                          # Exploratory data analysis
│   ├── results.ipynb                      # Results aggregation & visualization
│   └── demo.ipynb                         # Demo & inference examples
│
├── scripts/
│   └── train_pipeline.py                  # Main training orchestration script
│
├── models/
│   ├── logistic_regression.pkl            # Trained Logistic Regression
│   ├── random_forest.pkl                  # Trained Random Forest
│   ├── gradient_boosting.pkl              # Trained Gradient Boosting
│   ├── svm.pkl                            # Trained SVM (OneVsRest)
│   └── mlp.pkl                            # Trained Multi-Layer Perceptron
│
├── results/
│   ├── metrics.json                       # Detailed metrics & hyperparameters
│   ├── performance_comparison.csv         # Summary table
│   └── [visualizations if saved]
│
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
└── AGENT_PLAN.md                          # Detailed implementation plan
```

## Methods Summary

### Data Handling
- **Train/Val/Test Split**: 85% (train+val combined), 15% (test) , train+val splits again into 85% train and 15% val (72.25% total and 12.75% total)
- **Stratification**: StratifiedKFold ensures class proportions are preserved in all splits
- **Cross-Validation**: 3-fold stratified CV on combined train+val data (adaptive based on smallest class size)
- **Hyperparameter Tuning**: GridSearchCV with f1_weighted as the scoring metric

### Preprocessing
- **Feature Encoding**: Categorical features encoded with LabelEncoder
- **Feature Scaling**: StandardScaler applied to numeric features (fit on train, applied to all splits)
- **Missing Value Handling**: Rows with missing values are dropped (based on EDA findings)

### Algorithms

All algorithms use class-weighted loss to handle class imbalance.

#### 1. Logistic Regression
- Multi-class classification via one-vs-rest
- Hyperparameters: C (regularization), solver, max_iter, class_weight
- Best for: Fast training, interpretable coefficients

#### 2. Random Forest
- Ensemble of decision trees with bagging
- Hyperparameters: n_estimators, max_depth, min_samples_split, class_weight
- Best for: Feature importance, robustness to outliers

#### 3. Gradient Boosting
- Sequential tree ensemble with residual learning
- Hyperparameters: n_estimators, max_depth, learning_rate, subsample
- Best for: High accuracy, interaction detection

#### 4. Support Vector Machine (SVM)
- One-vs-Rest SVM for multi-class
- Hyperparameters: C, kernel (linear/rbf/poly), gamma, class_weight
- Best for: High-dimensional data (279 features)

#### 5. Multi-Layer Perceptron (MLP)
- Fully connected neural network
- Hyperparameters: hidden_layer_sizes, alpha (L2), learning_rate_init, max_iter
- Early stopping enabled to prevent overfitting
- Best for: Complex non-linear patterns

### Evaluation Metrics
- **Accuracy**: Overall correct classification rate
- **Macro F1**: Unweighted average F1 across all classes (prioritized for imbalanced data)
- **Weighted F1**: F1 weighted by class support
- **Per-class Precision & Recall**: Detailed breakdown by arrhythmia type
- **Confusion Matrix**: Class-wise prediction accuracy
- **Baseline**: Stratified random classifier for comparison

## Results Summary

- **Best-performing algorithm**: Logistic Regression had the highest Macro F1, but Gradient Boosting had highest accuracy.
- **Class imbalance observations**: Left Ventricule hypertrophy, PVC, and Supraventricular Premature Contraction
- **Most discriminative features**: Heart Rate was the most important feature across models
- **Notable misclassifications**: Normal can be misclassified as other arrythmias
- **Clinical implications**: This provides a transparent and reproducible baseline that can help with the clinical translation of classical models, showing that these models can be easily integrated in clinical workflows.

## Model Performance Comparison

After running `scripts/train_pipeline.py`, a performance table will be generated in `results/performance_comparison.csv`:

| Algorithm | Accuracy | Macro F1 | Weighted F1 | Precision | Recall |
|-----------|----------|----------|-------------|-----------|--------|
| Logistic Regression | ..0.746.. | ..0.642.. | ..0.702.. |
| Random Forest | ..0.746.. | .0.564... | ..0.707.. |
| Gradient Boosting | ..0.762.. | ..0.534.. | ..0.730.. |
| SVM | ..0.730.. | ..0.418.. | ..0.671.. |
| MLP | ..0.730.. | ..0.553.. | ..0.671.. |

## Limitations & Future Work

### Current Limitations
1. **Class Imbalance**: Unequal representation of arrhythmia types; minority classes harder to predict
2. **Limited Sample Size**: 452 records may be insufficient for deep learning approaches
3. **No Temporal Information**: Single ECG snapshot per patient; no tracking of arrhythmia progression over time
4. **Feature Interpretation**: 279 features is high-dimensional; multicollinearity possible
5. **No Clinical Validation**: Model trained on historical data; clinical deployment requires additional testing

### Future Improvements
1. **Data Augmentation**: Synthetic data generation for minority classes
2. **Feature Selection**: Reduce dimensionality (PCA, recursive feature elimination)
3. **Temporal Modeling**: Include patient history and ECG trends
4. **Ensemble Methods**: Voting or stacking of top models
5. **External Validation**: Test on independent patient cohort
6. **Explainability**: SHAP values or LIME for model interpretation
7. **Real-time Inference**: Deploy as REST API for clinical use

## Running the Full Reproducible Pipeline

### Step-by-Step Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Explore data
jupyter notebook notebooks/EDA.ipynb

# 3. Train all models (automatic data loading from .arff)
python scripts/train_pipeline.py

# 4. View results
jupyter notebook notebooks/results.ipynb

# 5. Run inference examples
jupyter notebook notebooks/demo.ipynb
```

### Reproducibility Checklist

- Fixed random seeds (RANDOM_STATE=42) in all functions
- Test set held-out during CV/tuning (CV only on train+val combined)
- Scaler fitted on train+val, applied consistently to test
- StratifiedKFold used throughout to preserve class distributions
- All file paths are relative (no hardcoded absolute paths)
- Running pipeline twice produces identical results
- All figures exportable as high-resolution PNG/PDF (300+ DPI)
- Full code documentation (docstrings for all functions)
- All dependencies pinned in requirements.txt

**Expected behavior**: Running `python scripts/train_pipeline.py` on a fresh machine with the same Python/package versions should produce identical metrics and saved models.

## Code Documentation

All modules follow consistent documentation standards:

### Docstring Format
```python
def function_name(param1, param2):
		"""
		One-line summary of what the function does.
    
		Extended description if needed.
    
		Parameters
		----------
		param1 : type
				Description of param1
		param2 : type
				Description of param2
    
		Returns
		-------
		output : type
				Description of return value
		"""
```

### Key Modules

- **`src/data_loader.py`**: ARFF parsing and stratified train/val/test splitting
- **`src/preprocessing.py`**: Categorical encoding and feature scaling
- **`src/models.py`**: Model training pipeline with GridSearchCV
- **`src/evaluation.py`**: Metrics computation and visualization functions
- **`scripts/train_pipeline.py`**: Main orchestration script

## Contributing & Development

### To Add a New Algorithm

1. Implement training function in `src/models.py`:
	 ```python
	 def train_new_algorithm(X_train, X_val, y_train, y_val, random_state=42):
			 # ... training code with GridSearchCV
			 return best_model, cv_results
	 ```

2. Add hyperparameter grid to `get_hyperparameter_grids()` in `src/models.py`

3. Update `train_all_models()` to call your function

4. Results will be automatically included in comparisons

### Running Tests

```bash
python -m pytest tests/ -v  # (if test suite added)
```

## References

- **Dataset**: UCI Machine Learning Repository - Cardiac Arrhythmia Database
- **scikit-learn**: https://scikit-learn.org/
- **ECG Interpretation**: Standard guidelines from American Heart Association

## License

[TBD]

## Contact

For questions or issues, please contact the development team or open an issue in the repository.

## Citation

If you use this project, please cite:

```
@misc{arrhythmia_classification_2026,
	title={Cardiac Arrhythmia Classification with scikit-learn},
	author={BME 6938 Spring 2026 - Group 1},
	year={2026}
}
```

---

**Last Updated**: February 2026  
**Pipeline Status**: ✅ Ready for execution
