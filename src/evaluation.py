"""
Evaluation module for model assessment and visualization.
Computes metrics, creates confusion matrices, ROC curves, and comparison charts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.dummy import DummyClassifier


def compute_metrics(y_true, y_pred, labels=None):
    """
    Compute comprehensive metrics for multi-class classification.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : array-like, optional
        List of class labels
    
    Returns
    -------
    metrics : dict
        Dictionary containing computed metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels)
    }
    
    # Per-class metrics
    if labels is not None:
        precision_per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        
        metrics['precision_per_class'] = precision_per_class
        metrics['recall_per_class'] = recall_per_class
        metrics['f1_per_class'] = f1_per_class
    
    return metrics


def compute_baseline_metrics(y_train, y_test, labels=None):
    """
    Compute baseline metrics using stratified random classifier.
    
    Parameters
    ----------
    y_train : array-like
        Training labels (for frequency estimation)
    y_test : array-like
        Test labels
    labels : array-like, optional
        List of class labels
    
    Returns
    -------
    baseline_metrics : dict
        Metrics for baseline classifier
    """
    baseline = DummyClassifier(strategy='stratified', random_state=42)
    baseline.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred_baseline = baseline.predict(np.zeros((len(y_test), 1)))
    
    return compute_metrics(y_test, y_pred_baseline, labels=labels)


def evaluate_all_models(models, X_test, y_test, y_train, label_encoder=None):
    """
    Evaluate all trained models on test set.
    
    Parameters
    ----------
    models : dict
        Dictionary of algorithm name -> trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test labels (encoded as integers)
    y_train : array-like
        Training labels (encoded as integers, for baseline)
    label_encoder : sklearn LabelEncoder, optional
        Encoder for decoding labels (not used for metrics computation)
    
    Returns
    -------
    results : dict
        Dictionary with evaluation results for each model
    baseline_metrics : dict
        Baseline metrics
    """
    # Use encoded integer labels directly (y_test, y_train are already integers)
    labels = np.unique(y_test)
    
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, labels=labels)
        results[name] = metrics
    
    # Compute baseline
    baseline_metrics = compute_baseline_metrics(y_train, y_test, labels=labels)
    
    return results, baseline_metrics


def create_comparison_table(results, baseline_metrics):
    """
    Create performance comparison table for all models.
    
    Parameters
    ----------
    results : dict
        Evaluation results for each model
    baseline_metrics : dict
        Baseline metrics
    
    Returns
    -------
    df_comparison : pd.DataFrame
        Comparison table with key metrics per model
    """
    data = []
    for model_name, metrics in results.items():
        data.append({
            'Algorithm': model_name.replace('_', ' ').title(),
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Macro F1': f"{metrics['f1_macro']:.4f}",
            'Weighted F1': f"{metrics['f1_weighted']:.4f}",
            'Precision (macro)': f"{metrics['precision_macro']:.4f}",
            'Recall (macro)': f"{metrics['recall_macro']:.4f}"
        })
    
    # Add baseline
    data.append({
        'Algorithm': 'Baseline (Stratified Random)',
        'Accuracy': f"{baseline_metrics['accuracy']:.4f}",
        'Macro F1': f"{baseline_metrics['f1_macro']:.4f}",
        'Weighted F1': f"{baseline_metrics['f1_weighted']:.4f}",
        'Precision (macro)': f"{baseline_metrics['precision_macro']:.4f}",
        'Recall (macro)': f"{baseline_metrics['recall_macro']:.4f}"
    })
    
    df_comparison = pd.DataFrame(data)
    return df_comparison


def plot_confusion_matrix(y_true, y_pred, labels, model_name, save_path=None):
    """
    Create and plot confusion matrix heatmap.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : array-like
        List of class labels
    model_name : str
        Name of model for title
    save_path : str, optional
        Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Normalized Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_per_class_f1_comparison(results, save_path=None):
    """
    Create grouped bar chart comparing per-class F1 scores across algorithms.
    
    Parameters
    ----------
    results : dict
        Evaluation results for each model
    save_path : str, optional
        Path to save figure
    """
    # Prepare data
    data_points = []
    for model_name, metrics in results.items():
        if 'f1_per_class' in metrics:
            for class_idx, f1 in enumerate(metrics['f1_per_class']):
                data_points.append({
                    'Algorithm': model_name.replace('_', ' ').title(),
                    'Class': class_idx,
                    'F1 Score': f1
                })
    
    if not data_points:
        print("No per-class F1 data available for comparison")
        return None
    
    df = pd.DataFrame(data_points)
    
    plt.figure(figsize=(16, 6))
    sns.barplot(data=df, x='Class', y='F1 Score', hue='Algorithm')
    plt.title('Per-Class F1 Score Comparison Across Algorithms', fontsize=14, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim([0, 1])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_model_ranking(results, metric='f1_macro', save_path=None):
    """
    Create bar chart ranking models by specified metric.
    
    Parameters
    ----------
    results : dict
        Evaluation results for each model
    metric : str, default='f1_macro'
        Metric to rank by
    save_path : str, optional
        Path to save figure
    """
    scores = [(name.replace('_', ' ').title(), metrics[metric]) 
              for name, metrics in results.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    
    names, values = zip(*scores)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color='steelblue')
    plt.title(f'Model Ranking by {metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.ylim([0, 1])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_baseline_comparison(results, baseline_metrics, save_path=None):
    """
    Create bar chart comparing best model vs baseline.
    
    Parameters
    ----------
    results : dict
        Evaluation results for each model
    baseline_metrics : dict
        Baseline metrics
    save_path : str, optional
        Path to save figure
    """
    # Find best model by macro F1
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_macro'])
    best_metrics = results[best_model_name]
    
    metrics_to_compare = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    best_values = [best_metrics[m] for m in metrics_to_compare]
    baseline_values = [baseline_metrics[m] for m in metrics_to_compare]
    
    x = np.arange(len(metrics_to_compare))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, best_values, width, label=best_model_name.replace('_', ' ').title(), color='steelblue')
    plt.bar(x + width/2, baseline_values, width, label='Baseline', color='coral')
    
    plt.title('Best Model vs Baseline Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Metric', fontsize=12)
    plt.xticks(x, [m.replace('_', ' ').title() for m in metrics_to_compare], rotation=45, ha='right')
    plt.legend()
    plt.ylim([0, 1])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def print_detailed_report(results, baseline_metrics):
    """
    Print detailed evaluation report to console.
    
    Parameters
    ----------
    results : dict
        Evaluation results for each model
    baseline_metrics : dict
        Baseline metrics
    """
    print("\n" + "="*80)
    print("DETAILED EVALUATION REPORT")
    print("="*80)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.replace('_', ' ').upper()}")
        print("-" * 80)
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"Precision (macro):  {metrics['precision_macro']:.4f}")
        print(f"Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"F1 Score (macro):   {metrics['f1_macro']:.4f}")
        print(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
    
    print(f"\n{'BASELINE (STRATIFIED RANDOM)'.upper()}")
    print("-" * 80)
    print(f"Accuracy:           {baseline_metrics['accuracy']:.4f}")
    print(f"Precision (macro):  {baseline_metrics['precision_macro']:.4f}")
    print(f"Recall (macro):     {baseline_metrics['recall_macro']:.4f}")
    print(f"F1 Score (macro):   {baseline_metrics['f1_macro']:.4f}")
    print(f"F1 Score (weighted): {baseline_metrics['f1_weighted']:.4f}")
    
    print("\n" + "="*80)
