**Legal Clause Similarity Detection System**

## Overview
This project implements a deep learning system for detecting similarity between legal clauses using two different neural network architectures: BiLSTM Siamese and Attention Encoder models. The system processes legal text data to classify whether two clauses belong to the same category or not.

The dataset used is Legal Clause Dataset: https://www.kaggle.com/datasets/bahushruth/legalclausedataset

## Project Structure
```
DL-Assignment2.ipynb - Main Jupyter notebook containing all code
datasets/ - Directory containing legal clause CSV files
```

## Features
- Text preprocessing and cleaning pipeline
- Clause pair generation (positive and negative pairs)
- Two model architectures for comparison
- Comprehensive evaluation metrics
- Performance visualization

## Requirements
### Python Libraries
- tensorflow
- pandas
- numpy
- matplotlib
- scikit-learn
- nltk

### Dataset Requirements
The system expects CSV files in the `datasets/` directory, where each file represents a different clause category. Each CSV should contain:
- `clause_text`: The actual legal clause text
- `clause_type`: Type of the clause
- Additional metadata columns

## Usage

### 1. Data Loading and Preprocessing
```python
# Load datasets from the datasets folder
data = load_legal_datasets('datasets/')

# Preprocess text data
data = preprocess_dataset(data, text_column='clause_text')
```

### 2. Generate Clause Pairs
```python
# Create positive and negative clause pairs
pairs_df = create_clause_pairs(data, max_pairs=50000)
```

### 3. Tokenization and Data Preparation
```python
# Tokenize and pad sequences
X1, X2, y, tokenizer = tokenize_and_pad(pairs_df, max_vocab=20000, max_len=150)

# Split into train/validation/test sets
X1_train, X2_train, y_train, X1_val, X2_val, y_val, X1_test, X2_test, y_test = split_dataset(X1, X2, y)
```

### 4. Model Training
#### BiLSTM Siamese Model
```python
model = build_bilstm_model(tokenizer=tokenizer, max_len=150)
history = model.fit([X1_train, X2_train], y_train, validation_data=([X1_val, X2_val], y_val), epochs=10, batch_size=64)
```

#### Attention Encoder Model
```python
attn_model = build_attention_model(max_len=150, max_vocab=20000)
history2 = attn_model.fit([X1_train, X2_train], y_train, validation_data=([X1_val, X2_val], y_val), epochs=10, batch_size=32)
```

### 5. Model Evaluation
```python
# Evaluate model performance
results = evaluate_model(model, X1_test, X2_test, y_test, model_name="BiLSTM Siamese")
```

## Models Implemented

### 1. BiLSTM Siamese Model
- Architecture: Siamese network with shared bidirectional LSTM layers
- Embedding: 128 dimensions, vocabulary size 20,000
- Features: Shared weights, concatenated clause representations
- Output: Binary classification (similar/not similar)

### 2. Attention Encoder Model
- Architecture: Transformer-style with multi-head self-attention
- Embedding: 128 dimensions, vocabulary size 20,000
- Features: 4 attention heads, residual connections, layer normalization
- Output: Binary classification (similar/not similar)

## Performance
Based on experimental results:
- **BiLSTM Siamese**: 98.2% accuracy, 0.997 ROC-AUC
- **Attention Encoder**: 82.2% accuracy, 0.891 ROC-AUC

The BiLSTM Siamese model demonstrated superior performance for this legal text similarity task.

## Key Functions

### Data Processing
- `load_legal_datasets()`: Loads all CSV files from specified directory
- `clean_text()`: Text preprocessing including lowercasing, tokenization, stopword removal, and lemmatization
- `create_clause_pairs()`: Generates positive and negative clause pairs for training

### Model Building
- `build_bilstm_model()`: Constructs the BiLSTM Siamese architecture
- `build_attention_model()`: Constructs the Attention Encoder architecture
- `tokenize_and_pad()`: Text tokenization and sequence padding

### Evaluation
- `evaluate_model()`: Comprehensive model evaluation with metrics and visualizations
- `split_dataset()`: Data splitting into train/validation/test sets

## Outputs
- Training history plots (accuracy, loss, AUC)
- Confusion matrices
- ROC curves
- Performance metrics (accuracy, precision, recall, F1-score, AUC)

## Notes
- The system is optimized for legal text with specific preprocessing tailored for legal language
- Default parameters are set for 150 sequence length and 20,000 vocabulary size
- Early stopping and model checkpointing are implemented for training optimization

## Future Improvements
- Hyperparameter tuning
- Additional model architectures
- Cross-validation
- Deployment as a web service
- Integration with legal document management systems
