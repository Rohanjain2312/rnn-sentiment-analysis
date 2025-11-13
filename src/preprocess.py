"""
Data preprocessing module for IMDb sentiment classification.
Handles text cleaning, tokenization, and sequence padding.
"""

import re
import string
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_imdb_data(num_words=10000):
    """
    Load IMDb dataset with vocabulary limit.
    
    Args:
        num_words: Maximum number of words to keep
        
    Returns:
        Tuple of (x_train, y_train), (x_test, y_test) and word_index
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)
    word_index = tf.keras.datasets.imdb.get_word_index()
    
    return (x_train, y_train), (x_test, y_test), word_index


def preprocess_review(review_indices, reverse_word_index):
    """
    Preprocess a single review by converting to text, cleaning, and returning word list.
    
    Args:
        review_indices: List of word indices
        reverse_word_index: Dictionary mapping indices to words
        
    Returns:
        List of cleaned words
    """
    # Convert indices to words (IMDb uses offset of 3)
    review_words = [reverse_word_index.get(i - 3, '?') for i in review_indices]
    review_string = ' '.join(review_words)
    
    # Clean text
    review_string = review_string.lower()
    review_string = re.sub(r'[%s]' % re.escape(string.punctuation), '', review_string)
    review_string = re.sub(r'\s+', ' ', review_string).strip()
    
    cleaned_words = review_string.split()
    return cleaned_words


def words_to_sequences(word_list, word_index, num_words):
    """
    Convert word list back to sequence of indices.
    
    Args:
        word_list: List of words
        word_index: Dictionary mapping words to indices
        num_words: Vocabulary size limit
        
    Returns:
        List of word indices
    """
    sequence = []
    for word in word_list:
        index = word_index.get(word)
        if index is not None and index < num_words:
            sequence.append(index)
        else:
            sequence.append(2)  # Unknown word token
    return sequence


def preprocess_dataset(x_train, x_test, word_index, num_words=10000):
    """
    Preprocess entire dataset.
    
    Args:
        x_train: Training data indices
        x_test: Test data indices
        word_index: Word to index mapping
        num_words: Vocabulary size
        
    Returns:
        Tuple of preprocessed (x_train_sequences, x_test_sequences)
    """
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    # Preprocess training data
    x_train_processed = [preprocess_review(review, reverse_word_index) for review in x_train]
    x_train_sequences = [words_to_sequences(review, word_index, num_words) for review in x_train_processed]
    
    # Preprocess test data
    x_test_processed = [preprocess_review(review, reverse_word_index) for review in x_test]
    x_test_sequences = [words_to_sequences(review, word_index, num_words) for review in x_test_processed]
    
    return x_train_sequences, x_test_sequences


def create_padded_sequences(sequences, max_lengths):
    """
    Create padded sequences for multiple sequence lengths.
    
    Args:
        sequences: List of sequence lists
        max_lengths: List of maximum sequence lengths to create
        
    Returns:
        Dictionary with keys 'train_{length}' or 'test_{length}' and padded arrays as values
    """
    padded_sequences = {}
    
    for length in max_lengths:
        padded = pad_sequences(sequences, maxlen=length, padding='post', truncating='post')
        padded_sequences[length] = padded
    
    return padded_sequences


def prepare_data(num_words=10000, max_lengths=[25, 50, 100]):
    """
    Complete data preparation pipeline.
    
    Args:
        num_words: Vocabulary size
        max_lengths: List of sequence lengths to prepare
        
    Returns:
        Dictionary containing:
            - padded_train: Dict of padded training sequences by length
            - padded_test: Dict of padded test sequences by length
            - y_train: Training labels
            - y_test: Test labels
            - word_index: Word to index mapping
    """
    # Load data
    (x_train, y_train), (x_test, y_test), word_index = load_imdb_data(num_words)
    
    # Preprocess
    x_train_sequences, x_test_sequences = preprocess_dataset(x_train, x_test, word_index, num_words)
    
    # Create padded sequences
    padded_train = create_padded_sequences(x_train_sequences, max_lengths)
    padded_test = create_padded_sequences(x_test_sequences, max_lengths)
    
    return {
        'padded_train': padded_train,
        'padded_test': padded_test,
        'y_train': y_train,
        'y_test': y_test,
        'word_index': word_index
    }
