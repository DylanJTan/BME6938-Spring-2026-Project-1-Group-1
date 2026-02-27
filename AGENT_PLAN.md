# Implementation Plan: Cardiac Arrhythmia Classification with scikit-learn

## Overview

Build a multi-class ECG classification pipeline handling 279 features across 16 arrhythmia classes using 5 scikit-learn algorithms (Logistic Regression, Random Forest, Gradient Boosting, SVM, MLP). Implement train/val/test split (70%/15%/15%) with 10-fold stratified CV on train/val to tune hyperparameters, then retrain on combined train/val and evaluate on held-out test. Organize repository with EDA notebook, ML pipeline script(s), demo notebook, and report-ready visualizations.

---

## Phase 1: Repository Setup & Data Pipeline

### 1.1 Project Structure
Create organized directories:
- `data/` — raw arff file and preprocessed versions
- `notebooks/` — EDA.ipynb, demo.ipynb, results.ipynb
- `src/` — Python modules for data loading, preprocessing, modeling
- `models/` — trained model artifacts (pickled sklearn models)
- `results/` — outputs (metrics, confusion matrices, ROC curves)
- `requirements.txt` — dependencies (scikit-learn, pandas, numpy, matplotlib, seaborn)

### 1.2 Data Loading Module (`src/data_loader.py`)
- Parse ARFF format (use `arff` library or scipy)
- Load into pandas DataFrame with proper column naming
- Handle missing values (imputation strategy to be determined during EDA)
- Implement train/val/test split (70%/15%/15% with stratification for class imbalance)
- Output: Separate train, val, and test DataFrames

### 1.3 Preprocessing Module (`src/preprocessing.py`)
- Feature scaling (StandardScaler for SVM/MLP, consider others for tree-based models)
- Handle mixed feature types (numeric measurements vs. nominal binary flags)
- Document all transformations for reproducibility
- Ensure proper fit/transform pipeline (fit on train/val, transform all splits with same scaler)

---

## Phase 2: Exploratory Data Analysis

### 2.1 EDA Notebook (`notebooks/EDA.ipynb`)
Generate insights to inform modeling decisions:
- **Dataset overview**: 452 samples, 279 features, 16 classes; show summary statistics
- **Missingness analysis**: Percentage missing per feature, visualize with heatmap
- **Class distribution**: Histogram showing imbalance severity; compute class ratios
- **Feature distributions**: Histograms/boxplots for key ECG measurements (heart rate, QRS duration, PR interval, QT interval)
- **Correlations**: Heatmap of feature correlations; identify potential multicollinearity
- **Feature importance heuristics**: Identify features with high variance/discriminative power (univariate analysis if feasible)
- **Key insights**: Document findings and their implications for preprocessing and modeling
- **Interpretation**: Explain how EDA findings motivate choices in feature engineering, imputation strategy, and algorithm selection

---

## Phase 3: ML Pipeline Development

### 3.1 Preprocessing & Feature Engineering (`src/preprocessing.py`)
- Decide on imputation strategy based on EDA (mean, median, forward-fill, or more sophisticated)
- Apply feature scaling (fit on train/val, transform all splits consistently)
- Consider feature selection if EDA suggests high dimensionality issues
- Document rationale for each preprocessing step

### 3.2 Model Training Module (`src/models.py`)
Build reusable training function with:
- Takes train/val data and algorithm specification
- Applies StratifiedKFold (10 folds) on combined train+val to handle class imbalance
- Uses GridSearchCV or RandomizedSearchCV for hyperparameter tuning
- Returns best model, CV metrics, and hyperparameter search results

**Hyperparameter grids for each algorithm:**

**Logistic Regression:**
- C: [0.001, 0.01, 0.1, 1, 10, 100]
- solver: ['lbfgs', 'saga']
- max_iter: [1000, 5000]
- class_weight: ['balanced', None]

**Random Forest:**
- n_estimators: [50, 100, 200, 300]
- max_depth: [5, 10, 20, None]
- min_samples_split: [2, 5, 10]
- class_weight: ['balanced', 'balanced_subsample']

**Gradient Boosting (sklearn.ensemble.GradientBoostingClassifier):**
- n_estimators: [50, 100, 200]
- max_depth: [3, 5, 7]
- learning_rate: [0.01, 0.05, 0.1]
- subsample: [0.8, 0.9, 1.0]

**Support Vector Machine (OneVsRest with SVC):**
- C: [0.1, 1, 10, 100]
- kernel: ['linear', 'rbf', 'poly']
- gamma: ['scale', 'auto'] (for rbf/poly)
- class_weight: ['balanced', None]

**Multi-Layer Perceptron:**
- hidden_layer_sizes: [(100,), (100,100), (100,100,100), (200,100)]
- alpha: [0.0001, 0.001, 0.01]
- learning_rate_init: [0.001, 0.01, 0.1]
- max_iter: [500, 1000, 2000]
- early_stopping: True

**Notes:**
- Use StratifiedKFold for all cross-validation to preserve class proportions
- Apply class_weight='balanced' where available to mitigate class imbalance
- Set random_state for reproducibility
- Use scoring='f1_weighted' or 'f1_macro' for imbalanced multi-class problem

### 3.3 Evaluation Module (`src/evaluation.py`)
- **Retraining**: After hyperparameter tuning, retrain best model on combined train+val
- **Test evaluation**: Evaluate on held-out test set (never seen during CV/tuning)
- **Metrics computed**:
  - Per-class precision, recall, F1 (critical for imbalanced data)
  - Macro F1 (unweighted average across classes)
  - Weighted F1 (weighted by class support)
  - Overall accuracy
  - Confusion matrix
- **Baseline comparison**: Stratified random classifier (predict class with its true frequency) as lower bound
- **Visualization functions**:
  - Confusion matrix heatmap (normalized by true label for interpretability)
  - Per-class F1 bar chart across all 5 algorithms
  - ROC curves if binary reduction needed (one-vs-rest)
  - Performance summary table

---

## Phase 4: Results & Visualization

### 4.1 Results Notebook (`notebooks/results.ipynb`)
Aggregate and present findings:
- Load all 5 trained models from `models/` directory
- Display **performance comparison table**:
  - Columns: Algorithm, Accuracy, Macro F1, Weighted F1, Precision (macro), Recall (macro)
  - Rows: One per algorithm
  - Highlight best performer
- **Visualizations** (report-ready, high resolution):
  - Confusion matrix for best model (heatmap)
  - Per-class F1 comparison across all algorithms (grouped bar chart)
  - Model ranking by macro F1 score
  - Baseline vs. best model comparison
- **Interpretation** section explaining:
  - Which algorithm performs best and why
  - Class-specific performance patterns (which classes are hardest to predict)
  - Contrast with baseline
  - Implications for clinical use (accuracy sufficient? biases in predictions?)

---

## Phase 5: Demo & Inference

### 5.1 Demo Notebook (`notebooks/demo.ipynb`)
Demonstrate practical usage:
- **Load best-performing model** from `models/`
- **Sample inference**: Show predictions on 5-10 test samples with:
  - Input features (key ECG measurements)
  - Predicted class label
  - Prediction confidence (probability for predicted class)
  - Actual label (if available in test set)
  - Correct/incorrect indicator
- **Feature importance** (if applicable):
  - For Random Forest or Gradient Boosting: top 10 most important features
  - Visualize with bar chart
- **Interpretation**:
  - Example of correct prediction: which features drove the decision?
  - Example of incorrect prediction: what might have caused the misclassification?
- **Usage instructions**: How users can run inference on new data

---

## Phase 6: Documentation

### 6.1 Update README.md
Comprehensive documentation with clear instructions:

**Sections:**
1. **Project Overview**
   - Cardiac arrhythmia classification from ECG data
   - Clinical motivation (early detection of arrhythmias, patient safety)

2. **Problem Statement**
   - Multi-class classification (16 classes: normal + 15 arrhythmia types)
   - Goal: Minimize difference between cardiologist's classification and ML model predictions

3. **Data Source**
   - OpenML dataset #5 (Cardiac Arrhythmia Database)
   - 452 samples, 279 features (206 numeric + 73 nominal)
   - 12-channel ECG measurements + patient demographics
   - 16 target classes (01=normal, 02-15=arrhythmia types, 16=unclassified)

4. **Environment Setup**
   - Python version: 3.9+
   - Installation: `pip install -r requirements.txt`
   - List key dependencies: scikit-learn, pandas, numpy, matplotlib, seaborn, arff

5. **Quick Start**
   - 3-command example to run full pipeline:
     ```bash
     jupyter notebook notebooks/EDA.ipynb
     python scripts/train_pipeline.py
     jupyter notebook notebooks/results.ipynb
     ```

6. **Usage Instructions**
   - **EDA**: `jupyter notebook notebooks/EDA.ipynb` — explore data and preprocessing decisions
   - **Training**: `python scripts/train_pipeline.py` — train all 5 algorithms with CV
   - **Results**: `jupyter notebook notebooks/results.ipynb` — view performance comparison
   - **Inference**: `jupyter notebook notebooks/demo.ipynb` — run model on new data

7. **Repository Structure**
   ```
   ├── data/
   │   ├── dataset_5_arrhythmia.arff (raw dataset)
   │   └── [preprocessed versions if saved]
   ├── src/
   │   ├── data_loader.py
   │   ├── preprocessing.py
   │   ├── models.py
   │   └── evaluation.py
   ├── notebooks/
   │   ├── EDA.ipynb
   │   ├── results.ipynb
   │   └── demo.ipynb
   ├── models/
   │   ├── logistic_regression.pkl
   │   ├── random_forest.pkl
   │   ├── gradient_boosting.pkl
   │   ├── svm.pkl
   │   └── mlp.pkl
   ├── results/
   │   ├── confusion_matrix.png
   │   ├── performance_comparison.png
   │   └── metrics.json
   ├── scripts/
   │   └── train_pipeline.py
   ├── requirements.txt
   ├── README.md
   └── AGENT_PLAN.md
   ```

8. **Key Findings** (to be filled after modeling)
   - Best-performing algorithm and its test-set metrics
   - Class imbalance observations
   - Most discriminative features
   - Notable misclassifications or biases

9. **Methods Summary** (brief overview)
   - Data split: 70% train/val, 15% test
   - Cross-validation: 10-fold stratified on train/val
   - Hyperparameter tuning: GridSearchCV with macro F1 scoring
   - Algorithms: Logistic Regression, Random Forest, Gradient Boosting, SVM, MLP
   - Preprocessing: Feature scaling, missing value imputation

10. **Limitations & Future Work**
    - Class imbalance (unequal class representation)
    - Limited sample size (452 records) — generalizability to larger patient populations
    - No temporal information (single ECG snapshot per patient)
    - Clinical validation needed before deployment
    - Future: Incorporate additional clinical features, test on external validation set

11. **Running the Full Reproducible Pipeline**
    - Step-by-step commands to replicate results
    - Expected runtime for training
    - Output files generated

### 6.2 Code Documentation
- **Docstrings** for all functions/classes:
  ```python
  def load_data(filepath):
      """
      Load ARFF dataset and return preprocessed train/val/test splits.
      
      Parameters
      ----------
      filepath : str
          Path to .arff file
      
      Returns
      -------
      train_X, train_y, val_X, val_y, test_X, test_y : tuple of arrays
          Train/val/test splits with features and labels
      """
  ```
- **Inline comments** for complex logic (e.g., CV strategy, preprocessing rationale)
- **No hard-coded paths**: Use relative paths or config files
- **Consistent naming**: snake_case for functions/variables, PascalCase for classes

---

## Phase 7: Integration & Reproducibility Check

### 7.1 Main Training Script (`scripts/train_pipeline.py`)
Single entry point to:
1. Load data (call data_loader.py)
2. Preprocess (call preprocessing.py)
3. Train all 5 models with CV (call models.py)
4. Evaluate on test set (call evaluation.py)
5. Save trained models to `models/`
6. Save evaluation metrics and visualizations to `results/`
7. Print summary of results to console

### 7.2 Reproducibility Verification Checklist
- [ ] Fixed random seeds in all train/CV/model functions
- [ ] Test set untouched during hyperparameter tuning (CV on train/val only)
- [ ] Scaling fitted on train/val, applied to test
- [ ] StratifiedKFold used for class imbalance
- [ ] All file paths are relative (no absolute paths hardcoded)
- [ ] Running `scripts/train_pipeline.py` twice produces identical results
- [ ] All figures are exportable (PNG/PDF quality) for report inclusion
- [ ] EDA notebook runs without errors and generates expected outputs
- [ ] Demo notebook loads model and runs inference successfully

---

## Team Work Allocation (Optional)

| Phase | Task | Owner |
|-------|------|-------|
| Phase 1 | Data loading + preprocessing modules | **Person A** |
| Phase 2 | EDA notebook + exploratory analysis | **Person B** |
| Phase 3a | Logistic Regression + SVM training | **Person C** |
| Phase 3b | Random Forest + Gradient Boosting training | **Person D** |
| Phase 3c | MLP training + hyperparameter tuning | **Person E** |
| Phase 4 | Results aggregation + visualization | **Person B or F** |
| Phase 5 | Demo notebook | **Person A or C** |
| Phase 6 | README + documentation | **Person G** |
| Phase 7 | Integration script + reproducibility testing | **All** |

---

## Success Criteria

✓ **Code Quality**
- All functions have docstrings
- Modular, reusable code organized by purpose
- No hard-coded paths or magic numbers in source files
- Code runs without errors end-to-end

✓ **Data Handling**
- Train/val/test split properly implemented
- 10-fold CV applied to train+val, never touching test
- Preprocessing fit only on train/val, applied consistently to all splits

✓ **Modeling & Evaluation**
- All 5 algorithms trained and evaluated
- Hyperparameter tuning completed for each
- Metrics computed: per-class F1, macro F1, weighted F1, accuracy, confusion matrix
- Baseline comparison included (stratified random classifier)

✓ **Visualizations**
- Confusion matrix heatmap(s)
- Per-class performance comparison chart
- Algorithm ranking visualization
- High-resolution, labeled, report-ready

✓ **Reproducibility**
- Fixed random seeds → identical results on re-runs
- Clear instructions in README
- All dependencies in requirements.txt

✓ **Documentation**
- EDA notebook with insights and justifications
- Demo notebook demonstrating inference
- Results notebook showing performance comparison
- Comprehensive README with all required sections
- Inline code comments and full docstrings

---

## Timeline Estimate

- **Phase 1 (Data Pipeline)**: 2-3 days
- **Phase 2 (EDA)**: 2-3 days
- **Phase 3 (ML Pipeline)**: 3-5 days (parallelizable by algorithm)
- **Phase 4 (Results)**: 1-2 days
- **Phase 5 (Demo)**: 1 day
- **Phase 6 (Documentation)**: 1-2 days
- **Phase 7 (Integration & Testing)**: 1-2 days

**Total**: 2-3 weeks with concurrent team work; 4-6 weeks sequentially one person.

---

## Notes

1. **Class Imbalance**: Use StratifiedKFold and class_weight='balanced' throughout.
2. **Metric Choice**: For imbalanced multi-class, prioritize macro F1 over accuracy.
3. **Hyperparameter Search**: RandomizedSearchCV may be faster than GridSearchCV for large grids; adjust n_iter based on compute budget.
4. **Feature Scaling**: Required for SVM and MLP; optional but often helpful for tree-based models.
5. **Baseline**: Always compare to stratified random classifier to ensure models learn something meaningful.
6. **Report Figures**: Save all plots as high-res images (300+ DPI) for inclusion in formal report if needed later.

