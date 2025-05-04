# NLP Final Project: Question Difficulty Estimation

Created by Omri Dan, Idan Drori and Tom Rapoport

## Overview
This project focuses on developing a model to predict the difficulty of exam questions based on the text of the question and a corresponding teacher-provided answer. The primary goal is to assess whether large language models (LLMs) require subject-specific proficiency to accurately estimate question difficulty. The project leverages a dataset from the Data Structures course (0368-2158), which includes historical exam questions, teacher answers, and student performance metrics.

The project involves:
- Extracting linguistic, structural, and domain-specific features from question-answer pairs.
- Training a model using actual average student scores as labels.
- Evaluating the performance of various pre-trained LLMs on difficulty estimation tasks.

## Structure
The project is organized into several components, each with its own dedicated folder and `README.md` file for detailed documentation:

### 1. **Data**
Contains the dataset used for training and evaluation, including:
- Historical exam questions.
- Teacher-provided answers.
- Student performance metrics.


### 2. **Models**
Includes the implementation of various models used for difficulty estimation, such as:
- Fine-tuned LLMs.
- Baseline models for comparison.

The models are saved at the writer's repository and are not shared.

### 3. **Text Classification**
Contains scripts and models for text classification tasks related to the project.

Refer to the `text_classification/README.md` file for more information.

### 4. **RAG (Retrieval-Augmented Generation)**
Explores the use of retrieval-augmented generation techniques for question difficulty estimation.

Refer to the `RAG/README.md` file for details.

### 5. **Zero-Shot**
Contains experiments and scripts for zero-shot learning approaches to question difficulty estimation.

Refer to the `Zero-Shot/README.md` file for more information.

### 6. **QA_LOSS**
This folder contains extensions to the [*EDM-Question-Difficulty*](https://github.com/readerbench/EDM-Question-Difficulty) repository, tailored for our project on question difficulty estimation. The modifications include scripts for calculating QA loss with transformer-based models (Flan-T5 and Qwen), as well as regression-based evaluation for predicting difficulty scores.

Refer to the `QA_LOSS/README.md` file for details.

### 7. **text2props**
Explores the text2props method for question difficulty estimation.

Refer to the `text2props/README.md` file for more information.
- **Link to the original framework**: [*text2props* GitHub Repository](https://github.com/lucabenedetto/text2props).
- **Link to the original paper**: [Introducing a Framework to Assess Newly Created Questions with Natural Language Processing](https://doi.org/10.1007/978-3-030-52237-7_4).


## Key Features
- **Feature Extraction**: Inspired by Ha et al. (2019), the project extracts linguistic, structural, and domain-specific features from question-answer pairs.
- **Model Evaluation**: Models are evaluated based on their alignment with actual student scores using MAE, MSE, RMSE (regression tasks) and ACC, F1 (classification tasks).
- **LLM Comparison**: Experiments assess the performance of domain-specific vs. general-purpose LLMs.

## Getting Started
To get started with the project:
1. Clone the repository.
2. Follow the setup instructions in the `README.md` files located in each folder.

## Acknowledgments
We thank the staff of the Workshop on Applying Large Language Models to Education, Dr. Michal Kleinbort, Dr. Amir Rubinstein, Prof. Hanoch Levy, and Adi Haviv, for providing access to the Data Structures Exam dataset and for their guidance. And we would like to thank Maor Ivgy, lecturer for the Natural Language Processing course, for his support and direction throughout this project.
