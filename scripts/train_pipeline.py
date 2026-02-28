"""
Main training pipeline script.
Orchestrates data loading, preprocessing, model training, and evaluation.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_data
from preprocessing import preprocess_data, encode_target
from models import train_all_models, save_model
from evaluation import evaluate_all_models, create_comparison_table, print_detailed_report


def main():
    """
    Execute complete training pipeline:
    1. Load data
    2. Preprocess features and targets
    3. Train all models with CV on train+val
    4. Evaluate on test set
    5. Save models and results
    """
    
    # Configuration
    RANDOM_STATE = 42
    DATA_FILE = Path(__file__).parent.parent / 'data' / 'dataset_5_arrhythmia.arff'
    MODELS_DIR = Path(__file__).parent.parent / 'models'
    RESULTS_DIR = Path(__file__).parent.parent / 'results'
    
    # Create output directories if they don't exist
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    print("="*80)
    print("CARDIAC ARRHYTHMIA CLASSIFICATION PIPELINE")
    print("="*80)
    
    # Phase 1: Load data
    print("\n[Phase 1] Loading data...")
    try:
        # Move dataset file to data directory if not already there
        if not DATA_FILE.exists():
            # Try to find it in parent directory
            parent_data_file = Path(__file__).parent.parent / 'dataset_5_arrhythmia.arff'
            if parent_data_file.exists():
                import shutil
                shutil.copy(parent_data_file, DATA_FILE)
                print(f"  Copied dataset to {DATA_FILE}")
            else:
                raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")
        
        X_train, X_val, X_test, y_train, y_val, y_test, df_clean = load_data(
            str(DATA_FILE),
            target_col='class',
            missing_value_strategy='drop',
            random_state=RANDOM_STATE
        )
        print(f"  Loaded data: {df_clean.shape[0]} samples, {df_clean.shape[1]-1} features")
        print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        print(f"  Classes: {np.unique(y_train).size}")
    except Exception as e:
        print(f"  Error loading data: {e}")
        return
    
    # Phase 2: Preprocess
    print("\n[Phase 2] Preprocessing...")
    try:
        X_train_prep, X_val_prep, X_test_prep, preprocessing_info = preprocess_data(
            X_train, X_val, X_test,
            scale=True,
            encode_categorical=True
        )
        
        # Encode target
        y_train_enc, y_val_enc, y_test_enc, label_encoder = encode_target(y_train, y_val, y_test)
        
        print(f"  Features after preprocessing: {X_train_prep.shape[1]}")
        print(f"  Scaling applied: {'scaler' in preprocessing_info}")
        print(f"  Target classes: {label_encoder.classes_}")
    except Exception as e:
        print(f"  Error preprocessing: {e}")
        return
    
    # Phase 3: Train models
    print("\n[Phase 3] Training models with hyperparameter tuning...")
    print("  (10-fold stratified CV on train+val combined)")
    try:
        models, cv_results = train_all_models(
            X_train_prep, X_val_prep,
            y_train_enc, y_val_enc,
            random_state=RANDOM_STATE
        )
        print("  All models trained successfully!")
    except Exception as e:
        print(f"  Error training models: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Phase 4: Evaluate on test set
    print("\n[Phase 4] Evaluating on test set...")
    try:
        results, baseline_metrics = evaluate_all_models(
            models, X_test_prep, y_test_enc, y_train_enc,
            label_encoder=label_encoder
        )
        
        # Print comparison table
        comparison_table = create_comparison_table(results, baseline_metrics)
        print("\n" + comparison_table.to_string(index=False))
        
        # Print detailed report
        print_detailed_report(results, baseline_metrics)
    except Exception as e:
        print(f"  Error evaluating models: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Phase 5: Save models
    print("\n[Phase 5] Saving models...")
    try:
        for model_name, model in models.items():
            save_path = MODELS_DIR / f"{model_name}.pkl"
            save_model(model, save_path)
            print(f"  Saved {model_name} to {save_path}")
    except Exception as e:
        print(f"  Error saving models: {e}")
        return
    
    # Phase 6: Save results
    print("\n[Phase 6] Saving results...")
    try:
        # Save metrics to JSON
        metrics_summary = {}
        for model_name, metrics in results.items():
            metrics_summary[model_name] = {
                'accuracy': float(metrics['accuracy']),
                'precision_macro': float(metrics['precision_macro']),
                'precision_weighted': float(metrics['precision_weighted']),
                'recall_macro': float(metrics['recall_macro']),
                'recall_weighted': float(metrics['recall_weighted']),
                'f1_macro': float(metrics['f1_macro']),
                'f1_weighted': float(metrics['f1_weighted'])
            }
        
        metrics_summary['baseline'] = {
            'accuracy': float(baseline_metrics['accuracy']),
            'precision_macro': float(baseline_metrics['precision_macro']),
            'precision_weighted': float(baseline_metrics['precision_weighted']),
            'recall_macro': float(baseline_metrics['recall_macro']),
            'recall_weighted': float(baseline_metrics['recall_weighted']),
            'f1_macro': float(baseline_metrics['f1_macro']),
            'f1_weighted': float(baseline_metrics['f1_weighted'])
        }
        
        # Save hyperparameter search results
        hyperparams = {}
        for model_name, cv_result in cv_results.items():
            hyperparams[model_name] = {
                'best_params': str(cv_result['best_params']),
                'best_score': float(cv_result['best_score'])
            }
        
        metrics_summary['hyperparameters'] = hyperparams
        
        results_file = RESULTS_DIR / 'metrics.json'
        with open(results_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"  Saved metrics to {results_file}")
        
        # Save comparison table
        table_file = RESULTS_DIR / 'performance_comparison.csv'
        comparison_table.to_csv(table_file, index=False)
        print(f"  Saved comparison table to {table_file}")
    except Exception as e:
        print(f"  Error saving results: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nOutput files saved to:")
    print(f"  Models: {MODELS_DIR}/")
    print(f"  Results: {RESULTS_DIR}/")
    print("\nNext steps:")
    print("  1. Review results/metrics.json for detailed metrics")
    print("  2. Run notebooks/results.ipynb for visualizations")
    print("  3. Run notebooks/demo.ipynb for inference examples")


if __name__ == '__main__':
    main()
