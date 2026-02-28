"""
Model training module with hyperparameter tuning.
Trains multiple classifiers with cross-validation and hyperparameter search.
Handles extreme class imbalance with adaptive cross-validation strategy.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
import pickle
import json


def get_adaptive_cv_splits(y_combined):
    """
    Calculate adaptive number of CV splits based on class imbalance.
    Handles extreme imbalance gracefully (min class size < n_splits).
    
    Parameters
    ----------
    y_combined : array-like
        Combined target vector (train + val)
    
    Returns
    -------
    int
        Number of CV splits to use
    """
    unique, counts = np.unique(y_combined, return_counts=True)
    min_class_size = counts.min()
    
    # n_splits cannot exceed min_class_size // 2 to ensure each fold has samples from min class
    # Cap at 10 for reasonable computation
    n_splits = min(10, max(3, min_class_size // 2))
    
    print(f"  Class distribution: min={min_class_size}, max={counts.max()}, ratio={counts.max()/min_class_size:.1f}:1")
    print(f"  Using adaptive StratifiedKFold: n_splits={n_splits}")
    
    return n_splits


def get_hyperparameter_grids(use_reduced=True):
    """
    Define hyperparameter search grids for all algorithms.
    Reduced grids prioritize compute efficiency for RandomizedSearchCV.
    
    Parameters
    ----------
    use_reduced : bool, default=True
        Whether to use reduced parameter grids (for RandomizedSearchCV)
    
    Returns
    -------
    dict
        Dictionary with algorithm names as keys and param grids as values
    """
    if use_reduced:
        # Reduced grids optimized for RandomizedSearchCV (n_iter=20)
        param_grids = {
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs', 'saga'],
                'max_iter': [1000, 5000],
                'class_weight': ['balanced', None]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced', 'balanced_subsample']
            },
            'gradient_boosting': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0]
            },
            'svm': {
                'estimator__C': [0.1, 1, 10],
                'estimator__kernel': ['linear', 'rbf'],
                'estimator__gamma': ['scale', 'auto'],
                'estimator__class_weight': ['balanced', None]
            },
            'mlp': {
                'hidden_layer_sizes': [(100,), (100, 100), (200, 100)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01],
                'max_iter': [500, 1000]
            }
        }
    else:
        # Full grids (original, computationally expensive)
        param_grids = {
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs', 'saga'],
                'max_iter': [1000, 5000],
                'class_weight': ['balanced', None]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced', 'balanced_subsample']
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0]
            },
            'svm': {
                'estimator__C': [0.1, 1, 10, 100],
                'estimator__kernel': ['linear', 'rbf', 'poly'],
                'estimator__gamma': ['scale', 'auto'],
                'estimator__class_weight': ['balanced', None]
            },
            'mlp': {
                'hidden_layer_sizes': [(100,), (100, 100), (100, 100, 100), (200, 100)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'max_iter': [500, 1000, 2000]
            }
        }
    return param_grids


def train_logistic_regression(X_train, X_val, y_train, y_val, random_state=42):
    """
    Train Logistic Regression with adaptive hyperparameter tuning.
    Uses RandomizedSearchCV with adaptive CV splits for class imbalance.
    
    Parameters
    ----------
    X_train, X_val, y_train, y_val : array-like
        Train and validation data
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    best_model : LogisticRegression
        Trained model on best hyperparameters
    cv_results : dict
        Cross-validation results
    """
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    
    param_grid = get_hyperparameter_grids(use_reduced=True)['logistic_regression']
    n_splits = get_adaptive_cv_splits(y_combined)
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    model = LogisticRegression(random_state=random_state, max_iter=5000, solver='saga')
    
    search = RandomizedSearchCV(
        model, param_grid, n_iter=20, cv=cv, scoring='f1_weighted', n_jobs=-1, 
        random_state=random_state, verbose=1
    )
    
    search.fit(X_combined, y_combined)
    
    return search.best_estimator_, {
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'cv_results': search.cv_results_
    }


def train_random_forest(X_train, X_val, y_train, y_val, random_state=42):
    """
    Train Random Forest with adaptive hyperparameter tuning.
    Uses RandomizedSearchCV with adaptive CV splits for class imbalance.
    
    Parameters
    ----------
    X_train, X_val, y_train, y_val : array-like
        Train and validation data
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    best_model : RandomForestClassifier
        Trained model on best hyperparameters
    cv_results : dict
        Cross-validation results
    """
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    
    param_grid = get_hyperparameter_grids(use_reduced=True)['random_forest']
    n_splits = get_adaptive_cv_splits(y_combined)
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    
    search = RandomizedSearchCV(
        model, param_grid, n_iter=20, cv=cv, scoring='f1_weighted', n_jobs=-1, 
        random_state=random_state, verbose=1
    )
    
    search.fit(X_combined, y_combined)
    
    return search.best_estimator_, {
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'cv_results': search.cv_results_
    }


def train_gradient_boosting(X_train, X_val, y_train, y_val, random_state=42):
    """
    Train Gradient Boosting with adaptive hyperparameter tuning.
    Uses RandomizedSearchCV with adaptive CV splits for class imbalance.
    
    Parameters
    ----------
    X_train, X_val, y_train, y_val : array-like
        Train and validation data
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    best_model : GradientBoostingClassifier
        Trained model on best hyperparameters
    cv_results : dict
        Cross-validation results
    """
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    
    param_grid = get_hyperparameter_grids(use_reduced=True)['gradient_boosting']
    n_splits = get_adaptive_cv_splits(y_combined)
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    model = GradientBoostingClassifier(random_state=random_state)
    
    search = RandomizedSearchCV(
        model, param_grid, n_iter=20, cv=cv, scoring='f1_weighted', n_jobs=-1, 
        random_state=random_state, verbose=1
    )
    
    search.fit(X_combined, y_combined)
    
    return search.best_estimator_, {
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'cv_results': search.cv_results_
    }


def train_svm(X_train, X_val, y_train, y_val, random_state=42):
    """
    Train Support Vector Machine (OneVsRest) with adaptive hyperparameter tuning.
    Uses RandomizedSearchCV with adaptive CV splits for class imbalance.
    
    Parameters
    ----------
    X_train, X_val, y_train, y_val : array-like
        Train and validation data
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    best_model : OneVsRestClassifier with SVC
        Trained model on best hyperparameters
    cv_results : dict
        Cross-validation results
    """
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    
    param_grid = get_hyperparameter_grids(use_reduced=True)['svm']
    n_splits = get_adaptive_cv_splits(y_combined)
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    base_svm = SVC(random_state=random_state, probability=False)
    model = OneVsRestClassifier(base_svm, n_jobs=-1)
    
    search = RandomizedSearchCV(
        model, param_grid, n_iter=20, cv=cv, scoring='f1_weighted', n_jobs=-1, 
        random_state=random_state, verbose=1
    )
    
    search.fit(X_combined, y_combined)
    
    return search.best_estimator_, {
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'cv_results': search.cv_results_
    }


def train_mlp(X_train, X_val, y_train, y_val, random_state=42):
    """
    Train Multi-Layer Perceptron with adaptive hyperparameter tuning.
    Uses RandomizedSearchCV with adaptive CV splits for class imbalance.
    
    Parameters
    ----------
    X_train, X_val, y_train, y_val : array-like
        Train and validation data
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    best_model : MLPClassifier
        Trained model on best hyperparameters
    cv_results : dict
        Cross-validation results
    """
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    
    param_grid = get_hyperparameter_grids(use_reduced=True)['mlp']
    n_splits = get_adaptive_cv_splits(y_combined)
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    model = MLPClassifier(random_state=random_state, early_stopping=True, n_iter_no_change=10)
    
    search = RandomizedSearchCV(
        model, param_grid, n_iter=20, cv=cv, scoring='f1_weighted', n_jobs=1, 
        random_state=random_state, verbose=1
    )
    
    search.fit(X_combined, y_combined)
    
    return search.best_estimator_, {
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'cv_results': search.cv_results_
    }


def train_all_models(X_train, X_val, y_train, y_val, random_state=42):
    """
    Train all 5 models with cross-validation and hyperparameter tuning.
    
    Parameters
    ----------
    X_train, X_val, y_train, y_val : array-like
        Train and validation data
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    models : dict
        Dictionary with algorithm names and trained models
    cv_results : dict
        Cross-validation results for each model
    """
    print("Training Logistic Regression...")
    lr_model, lr_cv = train_logistic_regression(X_train, X_val, y_train, y_val, random_state)
    print(f"  Best CV Score: {lr_cv['best_score']:.4f}")
    
    print("Training Random Forest...")
    rf_model, rf_cv = train_random_forest(X_train, X_val, y_train, y_val, random_state)
    print(f"  Best CV Score: {rf_cv['best_score']:.4f}")
    
    print("Training Gradient Boosting...")
    gb_model, gb_cv = train_gradient_boosting(X_train, X_val, y_train, y_val, random_state)
    print(f"  Best CV Score: {gb_cv['best_score']:.4f}")
    
    print("Training Support Vector Machine...")
    svm_model, svm_cv = train_svm(X_train, X_val, y_train, y_val, random_state)
    print(f"  Best CV Score: {svm_cv['best_score']:.4f}")
    
    print("Training Multi-Layer Perceptron...")
    mlp_model, mlp_cv = train_mlp(X_train, X_val, y_train, y_val, random_state)
    print(f"  Best CV Score: {mlp_cv['best_score']:.4f}")
    
    models = {
        'logistic_regression': lr_model,
        'random_forest': rf_model,
        'gradient_boosting': gb_model,
        'svm': svm_model,
        'mlp': mlp_model
    }
    
    cv_results = {
        'logistic_regression': lr_cv,
        'random_forest': rf_cv,
        'gradient_boosting': gb_cv,
        'svm': svm_cv,
        'mlp': mlp_cv
    }
    
    return models, cv_results


def save_model(model, filepath):
    """
    Save trained model to file.
    
    Parameters
    ----------
    model : sklearn model
        Trained model
    filepath : str
        Path to save model
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath):
    """
    Load trained model from file.
    
    Parameters
    ----------
    filepath : str
        Path to model file
    
    Returns
    -------
    model : sklearn model
        Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model
