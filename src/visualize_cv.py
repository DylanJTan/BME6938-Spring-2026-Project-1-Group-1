"""
Visualization of Cross-Validation Performance with Mean and Standard Deviation.

This module creates visualizations showing:
1. Mean CV scores with error bars (±1 std dev)
2. Individual fold scores plotted as points to show variance
"""

import json
import matplotlib.pyplot as plt
import numpy as np


def visualize_cv_performance(metrics_path='results/metrics.json', output_path='results/figures/cv_performance.png'):
    """
    Create a comprehensive CV performance visualization.
    
    Parameters
    ----------
    metrics_path : str
        Path to the metrics.json file
    output_path : str
        Path where the figure will be saved
    """
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Extract CV data
    models = ['logistic_regression', 'random_forest', 'gradient_boosting', 'svm', 'mlp']
    model_names = ['Logistic\nRegression', 'Random\nForest', 'Gradient\nBoosting', 'SVM', 'MLP']

    means = []
    stds = []
    cv_scores = []

    for model in models:
        hp = metrics['hyperparameters'][model]
        if 'mean_cv_score' in hp:
            means.append(hp['mean_cv_score'])
            stds.append(hp['std_cv_score'])
            cv_scores.append(hp['cv_scores_per_fold'])
        else:
            # Handle missing data gracefully
            means.append(0)
            stds.append(0)
            cv_scores.append([])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Bar chart with error bars
    x_pos = np.arange(len(model_names))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean CV F1 Score (weighted)', fontsize=12, fontweight='bold')
    ax1.set_title('Cross-Validation Performance (3-Fold)\nwith Standard Deviation', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 1)

    # Add value labels on bars
    for i, (m, s) in enumerate(zip(means, stds)):
        if m > 0:  # Only add labels for valid scores
            ax1.text(i, m + s + 0.02, f'{m:.3f}\n±{s:.3f}', 
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Right: Individual fold scores
    positions = []
    all_scores = []
    scatter_colors = []

    for i, scores in enumerate(cv_scores):
        if scores:  # Only add if scores exist
            positions.extend([i] * len(scores))
            all_scores.extend(scores)
            scatter_colors.extend([colors[i]] * len(scores))

    ax2.scatter(positions, all_scores, c=scatter_colors, s=150, alpha=0.6, 
                edgecolors='black', linewidth=1.5, label='Individual Fold Scores')
    ax2.plot(range(len(model_names)), means, 'k--', alpha=0.7, linewidth=2.5, label='Mean CV Score')
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Score (weighted)', fontsize=12, fontweight='bold')
    ax2.set_title('Individual Fold Scores\nAcross 3-Fold CV', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ CV performance visualization saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    visualize_cv_performance()
