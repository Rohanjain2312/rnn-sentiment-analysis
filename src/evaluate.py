"""
Model evaluation module for computing performance metrics.
"""

import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix


def evaluate_model(model, x_test, y_test, threshold=0.5):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained Keras model
        x_test: Test sequences
        y_test: Test labels
        threshold: Classification threshold for binary prediction
        
    Returns:
        Tuple of (accuracy, f1_score)
    """
    # Get model predictions
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    # Get probability predictions
    y_pred_proba = model.predict(x_test, verbose=0)
    
    # Convert to binary predictions
    y_pred = (y_pred_proba > threshold).astype("int32")
    
    # Calculate F1 score
    f1 = f1_score(y_test, y_pred)
    
    return accuracy, f1


def get_detailed_metrics(model, x_test, y_test, threshold=0.5):
    """
    Get detailed evaluation metrics including classification report.
    
    Args:
        model: Trained Keras model
        x_test: Test sequences
        y_test: Test labels
        threshold: Classification threshold
        
    Returns:
        Dictionary containing detailed metrics
    """
    # Get predictions
    y_pred_proba = model.predict(x_test, verbose=0)
    y_pred = (y_pred_proba > threshold).astype("int32")
    
    # Calculate metrics
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report
    }


def print_evaluation_summary(metrics):
    """
    Print formatted evaluation summary.
    
    Args:
        metrics: Dictionary from get_detailed_metrics
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION SUMMARY")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(metrics['classification_report'])
    print("="*50 + "\n")
