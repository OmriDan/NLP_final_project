# README for Text Classification Module

## Overview
This module is designed for training and evaluating text classification models. It supports retrieval-augmented generation (RAG) for enhanced context-based classification and includes functionality for preprocessing, training, and inference.

## Features
- **Model Training**: Train classification models using datasets with or without RAG.
- **Inference**: Predict the difficulty of text-based problems and solutions.
- **Metrics**: Evaluate models using accuracy, F1 scores, and per-class metrics.
- **Precomputed Retrievals**: Use precomputed RAG retrievals for faster training and inference.

## Directory Structure
- `numina_classification.py`: Main script for training and evaluating models with RAG.
- `numina_inference.py`: Script for running inference on new datasets.
- `leetcode_classifier.py`: Specialized script for classifying LeetCode problems.
- `RAG/`: Contains utilities for retrieval-augmented generation.

## Usage

### Training
Run the `numina_classification.py` script to train a model:
```bash
python numina_classification.py
```
### Inference
Use the numina_inference.py script to predict difficulty levels:
python numina_inference.py --model_folder <path_to_model> --csv_path <path_to_csv>

### LeetCode Classification
Run the leetcode_classifier.py script for LeetCode-specific datasets:
```
python leetcode_classifier.py
```

### Requirements
All required libraries are listed in the `requirements.txt` file. Install them using bash

## Notes
Ensure that the required datasets are downloaded and preprocessed before training.
For RAG, precompute retrievals using the provided scripts.