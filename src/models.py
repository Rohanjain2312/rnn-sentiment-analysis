"""
Model architecture definitions for RNN, LSTM, and Bidirectional LSTM.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop


def build_model(model_type, vocabulary_size, embedding_dim, sequence_length, 
                hidden_units=64, dropout_rate=0.5, optimizer_type='Adam', 
                learning_rate=0.001, clipnorm=None, clipvalue=None):
    """
    Build RNN-based model for sentiment classification.
    
    Args:
        model_type: Type of model ('RNN', 'LSTM', 'Bidirectional LSTM')
        vocabulary_size: Size of vocabulary
        embedding_dim: Dimension of embedding layer
        sequence_length: Maximum sequence length
        hidden_units: Number of units in recurrent layer
        dropout_rate: Dropout rate for regularization
        optimizer_type: Optimizer type ('Adam', 'SGD', 'RMSprop')
        learning_rate: Learning rate for optimizer
        clipnorm: Gradient clipping by norm (optional)
        clipvalue: Gradient clipping by value (optional)
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # Embedding layer
    model.add(Embedding(input_dim=vocabulary_size, 
                       output_dim=embedding_dim, 
                       input_length=sequence_length))
    
    # Recurrent layer
    if model_type == 'RNN':
        model.add(SimpleRNN(units=hidden_units, return_sequences=False))
    elif model_type == 'LSTM':
        model.add(LSTM(units=hidden_units, return_sequences=False))
    elif model_type == 'Bidirectional LSTM':
        model.add(Bidirectional(LSTM(units=hidden_units, return_sequences=False)))
    else:
        raise ValueError("Invalid model_type. Choose from 'RNN', 'LSTM', 'Bidirectional LSTM'.")
    
    # Dropout for regularization
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(units=1, activation='sigmoid'))
    
    # Configure optimizer
    optimizer = get_optimizer(optimizer_type, learning_rate, clipnorm, clipvalue)
    
    # Compile model
    model.compile(optimizer=optimizer, 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    
    return model


def get_optimizer(optimizer_type, learning_rate, clipnorm=None, clipvalue=None):
    """
    Create optimizer with optional gradient clipping.
    
    Args:
        optimizer_type: Type of optimizer ('Adam', 'SGD', 'RMSprop')
        learning_rate: Learning rate
        clipnorm: Gradient clipping by norm
        clipvalue: Gradient clipping by value
        
    Returns:
        Keras optimizer instance
    """
    optimizer_kwargs = {'learning_rate': learning_rate}
    
    if clipnorm is not None:
        optimizer_kwargs['clipnorm'] = clipnorm
    if clipvalue is not None:
        optimizer_kwargs['clipvalue'] = clipvalue
    
    if optimizer_type == 'Adam':
        return Adam(**optimizer_kwargs)
    elif optimizer_type == 'SGD':
        return SGD(**optimizer_kwargs)
    elif optimizer_type == 'RMSprop':
        return RMSprop(**optimizer_kwargs)
    else:
        raise ValueError("Invalid optimizer_type. Choose from 'Adam', 'SGD', 'RMSprop'.")


def get_model_config():
    """
    Get default model configuration parameters.
    
    Returns:
        Dictionary of model configuration
    """
    return {
        'vocabulary_size': 10000,
        'embedding_dim': 100,
        'hidden_units': 64,
        'dropout_rate': 0.5,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 5
    }
