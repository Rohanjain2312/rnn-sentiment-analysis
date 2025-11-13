# Comparative Analysis of RNN Architectures for Sentiment Classification

**Author:** Rohan Jain  
**Student ID:** 121012081  
**Course:**  MSML 641 - Deep Learning

## Project Overview

This project implements and evaluates multiple Recurrent Neural Network architectures for binary sentiment classification on the IMDb Movie Review Dataset. The study systematically compares 243 configurations across RNN, LSTM, and Bidirectional LSTM models.

## Repository Structure

```
.
├── data/                       # Data directory (auto-downloaded by Keras)
├── src/                        # Source code
│   ├── preprocess.py          # Data preprocessing and cleaning
│   ├── models.py              # Model architecture definitions
│   ├── train.py               # Training and experiment execution
│   ├── evaluate.py            # Model evaluation metrics
│   └── utils.py               # Utility functions and visualization
├── results/                    # Experimental results
│   ├── metrics.csv            # Complete results table
│   └── plots/                 # Generated visualizations
├── report.pdf                  # Project report
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Requirements

- Python 3.10 or higher
- TensorFlow 2.15.0
- See `requirements.txt` for complete dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rnn-sentiment-analysis.git
cd rnn-sentiment-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the IMDb Movie Review Dataset:
- 50,000 reviews (25,000 train / 25,000 test)
- Binary labels (positive/negative)
- Vocabulary: Top 10,000 words
- Sequence lengths: 25, 50, 100 words

The dataset is automatically downloaded by TensorFlow Keras.

## Usage

### Run Complete Experiment Suite

Execute all 243 experiments:
```bash
cd src
python train.py
```

Expected runtime: 4-5 hours with GPU (NVIDIA T4)

### Run Individual Components

Preprocess data:
```python
from src.preprocess import prepare_data
data = prepare_data(num_words=10000, max_lengths=[25, 50, 100])
```

Build and train a model:
```python
from src.models import build_model
from src.evaluate import evaluate_model

model = build_model(
    model_type='LSTM',
    vocabulary_size=10000,
    embedding_dim=100,
    sequence_length=100,
    optimizer_type='Adam'
)

model.fit(x_train, y_train, epochs=5, batch_size=32)
accuracy, f1 = evaluate_model(model, x_test, y_test)
```

Generate visualizations:
```python
from src.utils import load_results, create_visualizations
results_df = load_results('results')
create_visualizations(results_df, 'results/plots')
```

## Experimental Configuration

### Model Architectures
- Simple RNN (64 units)
- LSTM (64 units)
- Bidirectional LSTM (64 units per direction)

### Hyperparameters
- Embedding dimension: 100
- Dropout rate: 0.5
- Batch size: 32
- Epochs: 5
- Learning rate: 0.001

### Experimental Variations
- Activation functions: Sigmoid, ReLU, Tanh
- Optimizers: Adam, SGD, RMSprop
- Sequence lengths: 25, 50, 100
- Gradient clipping: None, Clipnorm (1.0), Clipvalue (0.5)

## Results Summary

### Best Configuration
- Model: LSTM
- Activation: Tanh
- Optimizer: Adam
- Sequence Length: 100
- Performance: 76.5% accuracy, 0.767 F1-score

### Key Findings
- LSTM outperforms Simple RNN by 7.7%
- Longer sequences improve LSTM performance by 10.3%
- Adam optimizer achieves 4.6% higher accuracy than SGD
- Gradient clipping shows minimal impact (< 0.2%)

### Recommended Configuration (CPU)
- Model: LSTM
- Sequence Length: 50 (optimal speed/accuracy trade-off)
- Optimizer: Adam
- Expected: 71.3% accuracy with reasonable training time

## Output Files

After running experiments, the following files are generated:

- `results/metrics.csv`: Complete experimental results
- `results/plots/accuracy_vs_seqlength.png`: Accuracy comparison
- `results/plots/f1score_vs_seqlength.png`: F1-score comparison
- `results/plots/accuracy_by_activation.png`: Performance by activation function
- `results/plots/f1score_by_optimizer.png`: Performance by optimizer
- `results/plots/accuracy_by_gradclipping.png`: Impact of gradient clipping

## Reproducibility

All random seeds are fixed to 42 for reproducibility:
```python
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

### Hardware Used
- Platform: Google Colab
- GPU: NVIDIA Tesla T4 (16GB VRAM)
- RAM: ~15GB
- Python: 3.10
- TensorFlow: 2.15

## Project Report

Complete analysis and findings are available in `report.pdf`.

## Contact

Rohan Jain  
Student ID: 121012081
