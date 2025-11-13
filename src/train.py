"""
Training script for RNN sentiment classification experiments.
Runs systematic experiments across all parameter combinations.
"""

import time
import pandas as pd
import os
from preprocess import prepare_data
from models import build_model, get_model_config
from evaluate import evaluate_model
from utils import set_random_seed, save_results


def run_single_experiment(model_type, activation, optimizer_type, sequence_length, 
                          gradient_clipping, data, config):
    """
    Run a single training experiment.
    
    Args:
        model_type: Type of RNN model
        activation: Activation function (not used in this implementation)
        optimizer_type: Optimizer type
        sequence_length: Input sequence length
        gradient_clipping: Gradient clipping strategy
        data: Preprocessed data dictionary
        config: Model configuration dictionary
        
    Returns:
        Dictionary containing experiment results
    """
    # Set gradient clipping parameters
    clipnorm = None
    clipvalue = None
    if gradient_clipping == 'clipnorm':
        clipnorm = 1.0
    elif gradient_clipping == 'clipvalue':
        clipvalue = 0.5
    
    # Build model
    model = build_model(
        model_type=model_type,
        vocabulary_size=config['vocabulary_size'],
        embedding_dim=config['embedding_dim'],
        sequence_length=sequence_length,
        hidden_units=config['hidden_units'],
        dropout_rate=config['dropout_rate'],
        optimizer_type=optimizer_type,
        learning_rate=config['learning_rate'],
        clipnorm=clipnorm,
        clipvalue=clipvalue
    )
    
    # Get appropriate data
    x_train = data['padded_train'][sequence_length]
    x_test = data['padded_test'][sequence_length]
    y_train = data['y_train']
    y_test = data['y_test']
    
    # Train model
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_split=0.2,
        verbose=0
    )
    end_time = time.time()
    epoch_time = (end_time - start_time) / config['epochs']
    
    # Evaluate model
    accuracy, f1_score = evaluate_model(model, x_test, y_test)
    
    return {
        'Model': model_type,
        'Activation': activation,
        'Optimizer': optimizer_type,
        'Seq Length': sequence_length,
        'Grad Clipping': gradient_clipping,
        'Accuracy': accuracy,
        'F1-score': f1_score,
        'Epoch Time': epoch_time,
        'Training Loss History': history.history['loss']
    }


def run_all_experiments(output_dir='results'):
    """
    Run all experimental combinations systematically.
    
    Args:
        output_dir: Directory to save results
        
    Returns:
        DataFrame containing all results
    """
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Prepare data
    print("Preparing data...")
    data = prepare_data(num_words=10000, max_lengths=[25, 50, 100])
    
    # Get model configuration
    config = get_model_config()
    
    # Define parameter grid
    parameter_grid = {
        'model_type': ['RNN', 'LSTM', 'Bidirectional LSTM'],
        'activation': ['sigmoid', 'relu', 'tanh'],
        'optimizer_type': ['Adam', 'SGD', 'RMSprop'],
        'sequence_length': [25, 50, 100],
        'gradient_clipping': ['none', 'clipnorm', 'clipvalue']
    }
    
    # Initialize results list
    results_list = []
    
    # Run experiments
    experiment_counter = 0
    total_experiments = (len(parameter_grid['model_type']) * 
                        len(parameter_grid['activation']) * 
                        len(parameter_grid['optimizer_type']) * 
                        len(parameter_grid['sequence_length']) * 
                        len(parameter_grid['gradient_clipping']))
    
    print(f"Running {total_experiments} experiments...")
    
    for model_type in parameter_grid['model_type']:
        for activation in parameter_grid['activation']:
            for optimizer_type in parameter_grid['optimizer_type']:
                for sequence_length in parameter_grid['sequence_length']:
                    for gradient_clipping in parameter_grid['gradient_clipping']:
                        experiment_counter += 1
                        print(f"Experiment {experiment_counter}/{total_experiments}: "
                              f"{model_type}, {activation}, {optimizer_type}, "
                              f"Seq={sequence_length}, Clip={gradient_clipping}")
                        
                        result = run_single_experiment(
                            model_type, activation, optimizer_type,
                            sequence_length, gradient_clipping, data, config
                        )
                        results_list.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    save_results(results_df, output_dir)
    
    print(f"\nExperiments complete. Results saved to {output_dir}/")
    
    return results_df


if __name__ == '__main__':
    results = run_all_experiments()
    print("\nTop 5 performing configurations:")
    print(results.nlargest(5, 'Accuracy')[['Model', 'Optimizer', 'Seq Length', 'Accuracy', 'F1-score']])
