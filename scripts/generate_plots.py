"""
Generate all evaluation plots from trained models and test data.
Produces model comparisons, ROC curves, confusion matrices, class distributions, and feature importance.
"""

import sys
import os
from pathlib import Path
import pickle
import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_data
from preprocessing import preprocess_data, encode_target
from visualization import (
    plot_model_comparison,
    plot_roc_curves,
    plot_confusion_matrix,
    plot_class_distribution,
    plot_feature_importance
)


def main():
    """
    Generate all evaluation visualizations.
    """
    print("=" * 80)
    print("GENERATING EVALUATION PLOTS")
    print("=" * 80)
    
    # Configuration
    RANDOM_STATE = 42
    DATA_FILE = Path(__file__).parent.parent / 'dataset_5_arrhythmia.arff'
    MODELS_DIR = Path(__file__).parent.parent / 'models'
    PLOTS_DIR = Path(__file__).parent.parent / 'results' / 'plots'
    
    # Create plots directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load and preprocess data (same as training)
    print("\n[1/6] Loading and preprocessing data...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, df_clean, class_mapping = load_data(
            str(DATA_FILE),
            target_col='class',
            missing_value_strategy='drop',
            random_state=RANDOM_STATE
        )
        
        # Store original feature names before preprocessing
        original_feature_names = X_train.columns.tolist()
        
        # Preprocess
        X_train_prep, X_val_prep, X_test_prep, preprocessing_info = preprocess_data(
            X_train, X_val, X_test,
            scale=True,
            encode_categorical=True,
            drop_j=True
        )
        
        # Convert preprocessed data to numpy arrays to avoid feature name warnings
        X_train_prep_array = X_train_prep.values if hasattr(X_train_prep, 'values') else X_train_prep
        X_val_prep_array = X_val_prep.values if hasattr(X_val_prep, 'values') else X_val_prep
        X_test_prep_array = X_test_prep.values if hasattr(X_test_prep, 'values') else X_test_prep
        
        # Encode targets
        y_train_enc, y_val_enc, y_test_enc, label_encoder = encode_target(y_train, y_val, y_test)
        
        # Get class names from label encoder, then map to clinical descriptions
        class_names_numeric = [str(c) for c in label_encoder.classes_]
        class_names_clinical = []
        
        # Map numeric class codes to clinical descriptions
        for class_code in class_names_numeric:
            if class_mapping and class_code in class_mapping:
                # Use clinical description from ARFF file
                clinical_desc = class_mapping[class_code]
                class_names_clinical.append(f"{class_code}: {clinical_desc}")
            else:
                # Fallback to numeric code
                class_names_clinical.append(class_code)
        
        print(f"  ✓ Class mapping extracted: {len(class_mapping)} classes found")
        print(f"  ✓ Data loaded: {X_test_prep_array.shape[0]} test samples, {X_test_prep_array.shape[1]} features")
        print(f"  ✓ Classes: {len(class_names_numeric)} ({', '.join(class_names_numeric)})")
        print(f"  ✓ Original features: {len(original_feature_names)}")
        
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. Plot model comparison from metrics.json
    print("\n[2/6] Plotting model comparison...")
    try:
        plot_model_comparison(
            metrics_file='results/metrics.json',
            save_dir=str(PLOTS_DIR)
        )
    except Exception as e:
        print(f"  ✗ Error plotting model comparison: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Plot class distribution
    print("\n[3/6] Plotting class distribution...")
    try:
        plot_class_distribution(y_train_enc, y_test_enc, class_names_clinical, save_dir=str(PLOTS_DIR))
    except Exception as e:
        print(f"  ✗ Error plotting class distribution: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Generate per-model plots
    print("\n[4/6] Generating per-model plots...")
    model_files = {
        'logistic_regression': 'logistic_regression.pkl',
        'random_forest': 'random_forest.pkl',
        'gradient_boosting': 'gradient_boosting.pkl',
        'svm': 'svm.pkl',
        'mlp': 'mlp.pkl'
    }
    
    for model_name, model_file in model_files.items():
        model_path = MODELS_DIR / model_file
        
        if not model_path.exists():
            print(f"  ⚠ Skipping {model_name} (model file not found)")
            continue
        
        print(f"\n  Processing {model_name.replace('_', ' ').title()}...")
        
        try:
            # Load model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Use pre-converted numpy array to avoid feature name warnings
            X_test_array = X_test_prep_array
            
            # Get predictions
            y_pred = model.predict(X_test_array)
            
            # Plot confusion matrix
            plot_confusion_matrix(y_test_enc, y_pred, class_names_clinical, model_name, save_dir=str(PLOTS_DIR))
            
            # Plot ROC curves (if model supports predict_proba)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_array)
                plot_roc_curves(y_test_enc, y_pred_proba, class_names_clinical, model_name, save_dir=str(PLOTS_DIR))
            else:
                print(f"    ⚠ {model_name} does not support predict_proba, skipping ROC curves")
            
            # Plot feature importance (for tree-based models)
            if model_name in ['random_forest', 'gradient_boosting']:
                # Use actual ECG feature names from original dataset
                # Note: Feature count may differ due to J feature dropping
                feature_names = original_feature_names[:X_test_array.shape[1]] if len(original_feature_names) >= X_test_array.shape[1] else [f"Feature_{i}" for i in range(X_test_array.shape[1])]
                plot_feature_importance(model, feature_names, model_name, top_n=20, save_dir=str(PLOTS_DIR))
            
        except Exception as e:
            print(f"  ✗ Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 5. Summary
    print("\n[5/6] Generating summary statistics...")
    try:
        # Count generated plots
        plot_files = list(PLOTS_DIR.glob('*.png'))
        print(f"  ✓ Generated {len(plot_files)} plot files")
        
        # List all plots
        print(f"\n  Generated plots:")
        for plot_file in sorted(plot_files):
            print(f"    - {plot_file.name}")
            
    except Exception as e:
        print(f"  ✗ Error generating summary: {e}")
    
    print("\n" + "=" * 80)
    print("PLOT GENERATION COMPLETED!")
    print("=" * 80)
    print(f"\n📊 All plots saved to: {PLOTS_DIR}/")
    print("\nGenerated visualizations:")
    print("  • model_comparison.png - Side-by-side model performance comparison")
    print("  • class_distribution.png - Class imbalance visualization")
    print("  • {model}_confusion_matrix.png - Confusion matrices for each model")
    print("  • {model}_roc_curves.png - ROC curves for each model")
    print("  • {model}_feature_importance.png - Feature importance (tree-based models)")
    print("\n💡 Next steps:")
    print("  1. Review plots in results/plots/")
    print("  2. Use notebooks/results.ipynb for interactive analysis")
    print("  3. Include plots in your final report")


if __name__ == '__main__':
    main()
