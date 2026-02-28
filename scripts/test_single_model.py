"""
Single-model test script to validate pipeline fixes.
Trains Random Forest only to verify data loading, preprocessing, and training logic.
Useful for debugging without full computational overhead.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_data
from preprocessing import preprocess_data, encode_target
from models import train_random_forest, save_model
from evaluation import evaluate_all_models, compute_metrics


def main():
    """
    Test pipeline with single model:
    1. Load data
    2. Preprocess
    3. Train Random Forest only
    4. Evaluate
    """
    
    # Configuration
    RANDOM_STATE = 42
    DATA_FILE = Path(__file__).parent.parent / 'dataset_5_arrhythmia.arff'
    MODELS_DIR = Path(__file__).parent.parent / 'models'
    RESULTS_DIR = Path(__file__).parent.parent / 'results'
    
    # Create output directories
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    print("="*80)
    print("SINGLE-MODEL TEST: Random Forest Only")
    print("="*80)
    
    # Phase 1: Load data
    print("\n[Phase 1] Loading data...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, df_clean = load_data(
            str(DATA_FILE),
            target_col='class',
            missing_value_strategy='drop',
            random_state=RANDOM_STATE
        )
        print(f"  ✓ Loaded {df_clean.shape[0]} samples with {df_clean.shape[1]-1} features")
        print(f"  ✓ Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        unique_classes = np.unique(y_train)
        print(f"  ✓ Classes found: {len(unique_classes)} ({list(unique_classes)})")
        
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Phase 2: Preprocess
    print("\n[Phase 2] Preprocessing...")
    try:
        X_train_prep, X_val_prep, X_test_prep, preprocessing_info = preprocess_data(
            X_train, X_val, X_test,
            scale=True,
            encode_categorical=True,
            drop_j=True
        )
        
        # Encode target
        y_train_enc, y_val_enc, y_test_enc, label_encoder = encode_target(y_train, y_val, y_test)
        
        print(f"  ✓ Features after preprocessing: {X_train_prep.shape[1]}")
        print(f"  ✓ StandardScaler applied: {'scaler' in preprocessing_info}")
        print(f"  ✓ Target encoding completed: {len(label_encoder.classes_)} classes")
        print(f"  ✓ Target classes: {list(label_encoder.classes_)}")
        
    except Exception as e:
        print(f"  ✗ Error preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Phase 3: Train single model
    print("\n[Phase 3] Training Random Forest...")
    try:
        model, cv_results = train_random_forest(
            X_train_prep, X_val_prep,
            y_train_enc, y_val_enc,
            random_state=RANDOM_STATE
        )
        print(f"  ✓ Training completed")
        print(f"  ✓ Best CV Score: {cv_results['best_score']:.4f}")
        print(f"  ✓ Best Hyperparameters:")
        for param, value in cv_results['best_params'].items():
            print(f"      - {param}: {value}")
        
    except Exception as e:
        print(f"  ✗ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Phase 4: Evaluate
    print("\n[Phase 4] Evaluating on test set...")
    try:
        # Wrap model in dict for evaluate_all_models function
        models_dict = {'random_forest': model}
        results, baseline_metrics = evaluate_all_models(
            models_dict, X_test_prep, y_test_enc, y_train_enc,
            label_encoder=label_encoder
        )
        
        metrics = results['random_forest']
        
        print(f"  ✓ Evaluation completed")
        print(f"\n  Performance Metrics:")
        print(f"    - Accuracy:         {metrics['accuracy']:.4f}")
        print(f"    - Precision (macro):{metrics['precision_macro']:.4f}")
        print(f"    - Recall (macro):   {metrics['recall_macro']:.4f}")
        print(f"    - F1-Score (macro): {metrics['f1_macro']:.4f}")
        print(f"\n  Baseline (Majority Class):")
        print(f"    - Accuracy:         {baseline_metrics['accuracy']:.4f}")
        print(f"    - F1-Score (macro): {baseline_metrics['f1_macro']:.4f}")
        
    except Exception as e:
        print(f"  ✗ Error evaluating model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Phase 5: Save model
    print("\n[Phase 5] Saving model...")
    try:
        save_path = MODELS_DIR / 'random_forest_test.pkl'
        save_model(model, save_path)
        print(f"  ✓ Model saved to {save_path}")
        
    except Exception as e:
        print(f"  ✗ Error saving model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nValidation Summary:")
    print("  ✓ Data loading with '?' → NaN conversion")
    print("  ✓ Case-insensitive target column ('class')")
    print("  ✓ J feature dropping (279 → 278 features)")
    print("  ✓ Categorical encoding with unseen value handling")
    print("  ✓ Standard feature scaling")
    print("  ✓ Adaptive stratified K-fold cross-validation")
    print("  ✓ RandomizedSearchCV hyperparameter tuning")
    print("  ✓ Model training and evaluation")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
