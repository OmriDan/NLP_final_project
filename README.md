# NLP_final_project

## Description
This project aims to develop a model that predicts the difficulty of an exam question based on
the text of the question and a corresponding teacher-provided answer. We would also like to
know if a LLM needs proficiency in the exam’s subject in order to assess the difficulty of its
questions correctly. We will use a dataset from the Data Structures course (0368-2158)
containing historical exam questions, teacher answers, and student performance metrics.
Access to this database is generously provided to us by the staff of the Workshop on Applying
Large Language Models for Education.

The model will extract linguistic, structural, and domain-specific features from the
question-answer pairs and use them in order to predict the question’s difficulty. Our feature
extraction methodology will draw inspiration from the approach proposed by Ha et al. (2019).
During training, the model will have access to the actual average scores as labels. During
inference, the predicted scores will serve as a proxy for difficulty.
As an experiment, we would like to assess the performance of different pre-trained LLMs on
various datasets. For example, is a model proficient in coding and math (like DeepSeek-v3) can
produce better assessment than a LLM that was pre-trained on general knowledge text? We will
perform the assessment on a portion of our given dataset. Specifically, how closely its
predictions align with actual student scores.

## Data


## Models

## Experiments