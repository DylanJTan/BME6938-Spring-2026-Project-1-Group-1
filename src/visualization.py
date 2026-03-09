"""
Visualization utilities for model evaluation.
Generates comparison plots, ROC curves, confusion matrices, and feature importance charts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import json
import os

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_model_comparison(metrics_file='results/metrics.json', save_dir='results/plots'):
    """
    Compare all trained models on key metrics.
    
    Parameters
    ----------
    metrics_file : str
        Path to metrics.json file
    save_dir : str
        Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load metrics
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    # Extract model metrics (exclude baseline and hyperparameters)
    models_data = []
    for model_name, metrics in data.items():
        if model_name not in ['baseline', 'hyperparameters']:
            models_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': metrics['accuracy'],
                'Precision (Macro)': metrics['precision_macro'],
                'Precision (Weighted)': metrics['precision_weighted'],
                'Recall (Macro)': metrics['recall_macro'],
                'Recall (Weighted)': metrics['recall_weighted'],
                'F1 (Macro)': metrics['f1_macro'],
                'F1 (Weighted)': metrics['f1_weighted']
            })
    
    df = pd.DataFrame(models_data)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold', y=0.995)
    
    # 1. Accuracy comparison
    ax1 = axes[0, 0]
    x = np.arange(len(df))
    colors = plt.cm.viridis(df['Accuracy'] / df['Accuracy'].max())
    bars = ax1.bar(x, df['Accuracy'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=data['baseline']['accuracy'], color='red', linestyle='--', linewidth=2, label='Baseline')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Test Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.4, linestyle='--')
    ax1.set_ylim([0, 1])
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df['Accuracy'])):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. F1 Score comparison
    ax2 = axes[0, 1]
    width = 0.35
    x = np.arange(len(df))
    bars1 = ax2.bar(x - width/2, df['F1 (Macro)'], width, label='Macro F1', alpha=0.8, color='steelblue', edgecolor='black')
    bars2 = ax2.bar(x + width/2, df['F1 (Weighted)'], width, label='Weighted F1', alpha=0.8, color='coral', edgecolor='black')
    ax2.axhline(y=data['baseline']['f1_macro'], color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Baseline (Macro)')
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1 Scores (Macro vs Weighted)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.4, linestyle='--')
    ax2.set_ylim([0, 1])
    
    # 3. Precision vs Recall (Macro)
    ax3 = axes[1, 0]
    colors_pr = plt.cm.tab10(np.arange(len(df)))
    for i, (model, prec, rec) in enumerate(zip(df['Model'], df['Precision (Macro)'], df['Recall (Macro)'])):
        ax3.scatter(rec, prec, s=200, alpha=0.8, color=colors_pr[i], edgecolor='black', linewidth=1.5, label=model)
    ax3.set_xlabel('Recall (Macro)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Precision (Macro)', fontsize=12, fontweight='bold')
    ax3.set_title('Precision vs Recall (Macro)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(alpha=0.4, linestyle='--')
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # 4. Ranking table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    # Sort by F1 Weighted
    df_sorted = df.sort_values('F1 (Weighted)', ascending=False)
    table_data = df_sorted[['Model', 'Accuracy', 'F1 (Weighted)', 'F1 (Macro)']].round(3)
    table_data.insert(0, 'Rank', range(1, len(table_data) + 1))
    
    table = ax4.table(cellText=table_data.values,
                     colLabels=table_data.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(table_data.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    return df


def plot_roc_curves(y_true, y_pred_proba, class_names, model_name, save_dir='results/plots'):
    """
    Plot ROC curves for multiclass classification (one-vs-rest).
    Handles classes missing from test set gracefully.
    
    Parameters
    ----------
    y_true : array-like
        True labels (encoded as integers)
    y_pred_proba : array-like
        Predicted probabilities (n_samples, n_classes)
    class_names : list
        List of class names (should match n_classes in y_pred_proba)
    model_name : str
        Name of the model
    save_dir : str
        Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_classes = len(class_names)
    
    # Binarize the labels for one-vs-rest
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    classes_with_samples = []  # Track which classes have positive samples
    
    for i in range(n_classes):
        # Skip if no positive samples in this class (occurs with extreme class imbalance)
        if y_true_bin[:, i].sum() == 0:
            roc_auc[i] = np.nan  # Mark as unavailable
            continue
        
        try:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            classes_with_samples.append(i)
        except Exception as e:
            print(f"    ⚠ Could not compute ROC for class {class_names[i]}: {e}")
            roc_auc[i] = np.nan
    
    # Compute micro-average ROC curve (only if we have classes with samples)
    if len(classes_with_samples) > 0:
        try:
            fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        except Exception as e:
            print(f"    ⚠ Could not compute micro-average ROC: {e}")
            roc_auc["micro"] = np.nan
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot ROC for each class with AUC > 0.5 (and not NaN)
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    plotted = 0
    for i, color in zip(range(n_classes), colors):
        if not np.isnan(roc_auc[i]) and roc_auc[i] > 0.5:  # Only plot valid, meaningful AUCs
            ax.plot(fpr[i], tpr[i], color=color, lw=1.5, alpha=0.7,
                   label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.2f})')
            plotted += 1
    
    # Plot micro-average (if available)
    if not np.isnan(roc_auc.get("micro", np.nan)):
        ax.plot(fpr["micro"], tpr["micro"], color='deeppink', lw=3,
               label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})', linestyle='--')
    
    # Plot baseline
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Chance (AUC = 0.50)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title(f'ROC Curves - {model_name.replace("_", " ").title()}', fontsize=16, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add note if some classes were missing
    missing_count = n_classes - len(classes_with_samples)
    if missing_count > 0:
        ax.text(0.5, 0.05, f'{missing_count} classes not in test set', 
               ha='center', fontsize=9, style='italic', alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_roc_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    return roc_auc


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_dir='results/plots'):
    """
    Plot confusion matrix heatmap with actual class labels.
    
    Parameters
    ----------
    y_true : array-like
        True labels (encoded as integers 0 to n_classes-1)
    y_pred : array-like
        Predicted labels (encoded as integers)
    class_names : list
        List of actual class names (e.g., ['1', '10', '14', ...])
    model_name : str
        Name of the model
    save_dir : str
        Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_classes = len(class_names)
    
    # Create confusion matrix with all classes (some may have zero counts)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    
    # Normalize by row (true label)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Raw confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
    axes[0].set_xticks(np.arange(n_classes))
    axes[0].set_yticks(np.arange(n_classes))
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].set_yticklabels(class_names, rotation=0)
    axes[0].set_title(f'Confusion Matrix (Counts) - {model_name.replace("_", " ").title()}', 
                      fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Class', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    
    # Normalized confusion matrix
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1],
                cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1, 
                linewidths=0.5, linecolor='gray')
    axes[1].set_xticks(np.arange(n_classes))
    axes[1].set_yticks(np.arange(n_classes))
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_yticklabels(class_names, rotation=0)
    axes[1].set_title(f'Confusion Matrix (Normalized) - {model_name.replace("_", " ").title()}', 
                      fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Class', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_class_distribution(y_train, y_test, class_names, save_dir='results/plots'):
    """
    Plot class distribution to show imbalance in train and test sets.
    
    Parameters
    ----------
    y_train : array-like
        Training labels
    y_test : array-like
        Test labels
    class_names : list
        List of class names
    save_dir : str
        Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get counts for train and test
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    
    # Align counts (some classes might be missing in test)
    all_classes = np.arange(len(class_names))
    counts_train_full = np.zeros(len(class_names), dtype=int)
    counts_test_full = np.zeros(len(class_names), dtype=int)
    
    for i, count in zip(unique_train, counts_train):
        counts_train_full[i] = count
    for i, count in zip(unique_test, counts_test):
        counts_test_full[i] = count
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Stacked bar chart
    x = np.arange(len(class_names))
    width = 0.6
    
    p1 = axes[0].bar(x, counts_train_full, width, label='Train', alpha=0.8, color='steelblue', edgecolor='black')
    p2 = axes[0].bar(x, counts_test_full, width, bottom=counts_train_full, label='Test', alpha=0.8, 
                     color='coral', edgecolor='black')
    
    axes[0].set_xlabel('Class', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    axes[0].set_title('Class Distribution (Train vs Test)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add total counts on bars
    for i, (train_count, test_count) in enumerate(zip(counts_train_full, counts_test_full)):
        total = train_count + test_count
        if total > 0:
            axes[0].text(i, total + max(counts_train_full + counts_test_full)*0.01, 
                        str(total), ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Imbalance ratio visualization
    total_counts = counts_train_full + counts_test_full
    non_zero_counts = total_counts[total_counts > 0]
    
    if len(non_zero_counts) > 0:
        max_count = non_zero_counts.max()
        ratios = total_counts / max_count
        
        colors_ratio = plt.cm.RdYlGn(ratios)
        bars = axes[1].bar(x, total_counts, color=colors_ratio, alpha=0.8, edgecolor='black')
        axes[1].set_xlabel('Class', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Total Samples', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Class Imbalance (Ratio: {max_count}:{non_zero_counts.min():.0f})', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add percentage labels
        for i, (bar, count) in enumerate(zip(bars, total_counts)):
            if count > 0:
                pct = (count / total_counts.sum()) * 100
                axes[1].text(bar.get_x() + bar.get_width()/2, count + max_count*0.01,
                           f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'class_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names, model_name, top_n=20, save_dir='results/plots'):
    """
    Plot feature importance for tree-based models.
    
    Parameters
    ----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names (string labels)
    model_name : str
        Name of the model
    top_n : int
        Number of top features to display
    save_dir : str
        Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if not hasattr(model, 'feature_importances_'):
        print(f"  ⚠ Model {model_name} does not have feature_importances_")
        return
    
    importances = model.feature_importances_
    
    # Ensure we don't try to display more features than exist
    actual_top_n = min(top_n, len(importances))
    
    indices = np.argsort(importances)[-actual_top_n:]
    
    # Get feature names for top features
    top_feature_names = [feature_names[i] if i < len(feature_names) else f"Feature_{i}" 
                         for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(11, 8))
    colors = plt.cm.viridis(top_importances / top_importances.max())
    bars = plt.barh(range(len(indices)), top_importances, color=colors, alpha=0.8, edgecolor='black')
    
    # Set y-axis labels with actual feature names
    plt.yticks(range(len(indices)), top_feature_names, fontsize=10)
    
    plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.ylabel('ECG Features', fontsize=12, fontweight='bold')
    plt.title(f'Top {actual_top_n} Feature Importances - {model_name.replace("_", " ").title()}', 
              fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top_importances)):
        plt.text(val + max(top_importances)*0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_feature_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
