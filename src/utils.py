"""
Utility functions for experiment management and reproducibility.
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


def set_random_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_results(results_df, output_dir='results'):
    """
    Save experimental results to CSV file.
    
    Args:
        results_df: DataFrame containing results
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full results
    results_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    
    # Save formatted results without training history
    results_formatted = results_df.drop(columns=['Training Loss History'])
    results_formatted.to_csv(os.path.join(output_dir, 'metrics_formatted.csv'), index=False)


def load_results(results_dir='results'):
    """
    Load experimental results from CSV file.
    
    Args:
        results_dir: Directory containing results
        
    Returns:
        DataFrame with results
    """
    return pd.read_csv(os.path.join(results_dir, 'metrics.csv'))


def create_visualizations(results_df, output_dir='results/plots'):
    """
    Create and save all experimental visualizations.
    
    Args:
        results_df: DataFrame containing experimental results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    
    # 1. Accuracy vs Sequence Length by Model
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='Seq Length', y='Accuracy', hue='Model', marker='o')
    plt.title('Accuracy vs. Sequence Length by Model Type')
    plt.xlabel('Sequence Length')
    plt.ylabel('Accuracy')
    plt.legend(title='Model')
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_seqlength.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. F1-Score vs Sequence Length by Model
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='Seq Length', y='F1-score', hue='Model', marker='o')
    plt.title('F1-score vs. Sequence Length by Model Type')
    plt.xlabel('Sequence Length')
    plt.ylabel('F1-score')
    plt.legend(title='Model')
    plt.savefig(os.path.join(output_dir, 'f1score_vs_seqlength.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Accuracy by Activation and Model
    g = sns.catplot(data=results_df, x='Seq Length', y='Accuracy', hue='Activation', 
                    col='Model', kind='point', height=5, aspect=0.8)
    g.fig.suptitle('Accuracy vs. Sequence Length by Activation and Model Type', y=1.02)
    g.set_axis_labels("Sequence Length", "Accuracy")
    plt.savefig(os.path.join(output_dir, 'accuracy_by_activation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. F1-Score by Optimizer and Model
    g = sns.catplot(data=results_df, x='Seq Length', y='F1-score', hue='Optimizer', 
                    col='Model', kind='point', height=5, aspect=0.8)
    g.fig.suptitle('F1-score vs. Sequence Length by Optimizer and Model Type', y=1.02)
    g.set_axis_labels("Sequence Length", "F1-score")
    plt.savefig(os.path.join(output_dir, 'f1score_by_optimizer.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Accuracy by Gradient Clipping
    g = sns.catplot(data=results_df, x='Seq Length', y='Accuracy', hue='Grad Clipping', 
                    col='Model', kind='point', height=5, aspect=0.8)
    g.fig.suptitle('Accuracy vs. Sequence Length by Gradient Clipping and Model Type', y=1.02)
    g.set_axis_labels("Sequence Length", "Accuracy")
    plt.savefig(os.path.join(output_dir, 'accuracy_by_gradclipping.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/")


def print_summary_statistics(results_df):
    """
    Print summary statistics from experimental results.
    
    Args:
        results_df: DataFrame containing results
    """
    print("\n" + "="*60)
    print("EXPERIMENTAL SUMMARY STATISTICS")
    print("="*60)
    
    print("\nTop 5 Configurations by Accuracy:")
    top5 = results_df.nlargest(5, 'Accuracy')[['Model', 'Activation', 'Optimizer', 
                                                 'Seq Length', 'Accuracy', 'F1-score']]
    print(top5.to_string(index=False))
    
    print("\n\nPerformance by Model Architecture:")
    model_stats = results_df.groupby('Model').agg({
        'Accuracy': 'mean',
        'F1-score': 'mean',
        'Epoch Time': 'mean'
    }).round(4)
    print(model_stats)
    
    print("\n\nPerformance by Optimizer:")
    optimizer_stats = results_df.groupby('Optimizer').agg({
        'Accuracy': 'mean',
        'F1-score': 'mean'
    }).round(4)
    print(optimizer_stats)
    
    print("\n\nPerformance by Sequence Length:")
    seqlength_stats = results_df.groupby('Seq Length').agg({
        'Accuracy': 'mean',
        'F1-score': 'mean'
    }).round(4)
    print(seqlength_stats)
    
    print("="*60 + "\n")
