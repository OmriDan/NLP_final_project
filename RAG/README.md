# README

## Overview
The `RAG` folder contains utilities and scripts for implementing Retrieval-Augmented Generation (RAG) in the project. RAG is used to enhance the model's ability to retrieve relevant context from a knowledge corpus, improving the accuracy and interpretability of predictions.

## Contents
- **`knowledge_corpus.txt`**: A text file containing the preprocessed knowledge corpus used for retrieval.
- **`retrieval_utils.py`**: Utility functions for managing and querying the knowledge corpus.
- **`rag_pipeline.py`**: Implements the RAG pipeline, including retrieval and integration with the model.
- **`precompute_retrievals.py`**: Script for precomputing retrievals from the knowledge corpus to speed up training and inference.

## Features
- **Knowledge Corpus Management**: Tools for loading, preprocessing, and querying the knowledge corpus.
- **RAG Pipeline**: Combines retrieval with model predictions for context-aware inference.
- **Precomputed Retrievals**: Allows for faster training and inference by caching retrieval results.

## Usage

### Precomputing Retrievals
Run the `precompute_retrievals.py` script to generate precomputed retrievals:
```bash
python precompute_retrievals.py --corpus_path <path_to_corpus> --output_path <path_to_output>
```
### Using the RAG Pipeline
The rag_pipeline.py script integrates retrieval with the model. Import and use it in your training or inference scripts:
```
from RAG.rag_pipeline import RAGPipeline

rag_pipeline = RAGPipeline(corpus_path="RAG/knowledge_corpus.txt")
retrieved_context = rag_pipeline.retrieve_context(question="What is a binary tree?")```
```
### Knowledge Corpus

Ensure the knowledge_corpus.txt file is updated with relevant documents. Use the retrieval_utils.py script to preprocess and manage the corpus.

## Notes
The knowledge corpus should be preprocessed and cleaned before use.
Precomputing retrievals is recommended for large datasets to improve performance.
Ensure all dependencies are installed as listed in the requirements.txt file.
Requirements can be installed using:
```bash
pip install -r requirements.txt
```