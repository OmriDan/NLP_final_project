
# Extensions to the *EDM-Question-Difficulty* Framework

This folder contains extensions to the [*EDM-Question-Difficulty*](https://github.com/readerbench/EDM-Question-Difficulty) repository, tailored for our project on question difficulty estimation. The modifications include scripts for calculating QA loss with transformer-based models (Flan-T5 and Qwen), as well as regression-based evaluation for predicting difficulty scores.

## Key Features
- **QA Loss Calculation**: The scripts `calculate_qa_loss_sequence_flan_t5.py` and `calculate_qa_loss_sequence_qwen.py` compute QA loss for question-answer pairs using pre-trained transformer models.
- **Regression**: The `regression.py` script trains and evaluates regression models to predict question difficulty based on computed QA losses. The models include Random Forest, Decision Trees, Linear Regression, and Support Vector Regression.
- **Data Parsing**: The `parse_data.py` file handles the parsing of datasets to the required format for QA loss calculation and regression.
- **Evaluation**: The `evaluation.py` file computes error metrics for evaluating the performance of difficulty prediction.

## Installation

1. **Create a virtual environment**:
   ```bash
   conda create -n edmdifficulty_env python=3.8
   conda activate edmdifficulty_env
   ```

2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run QA Loss Calculation and Regression

The QA loss calculation and regression processes are interconnected. You can calculate the QA loss for question-answer pairs and run the regression evaluation in one seamless step by running the appropriate script.

1. **Calculate QA Loss for Question-Answer Pairs and Run Regression (Flan-T5)**:
   ```bash
   python scripts/calculate_qa_loss_sequence_flan_t5.py
   ```

2. **Calculate QA Loss for Question-Answer Pairs and Run Regression (Qwen)**:
   ```bash
   python scripts/calculate_qa_loss_sequence_qwen.py
   ```

These scripts will:
- Load the dataset, compute QA cross-entropy losses using the selected model (Flan-T5 or Qwen).
- Automatically run regression using `regression.py` to predict difficulty based on the computed QA losses.

### Data Format

The input data should contain the following columns:

- **question_text**: The text of the question.
- **answer_text**: The text of the correct answer.

You can load your data using the provided utility functions in `parse_data.py`. The data is then processed into the required format for QA loss calculation and regression.

## Citation

If you use this framework, please cite the original *EDM-Question-Difficulty* paper:

```bibtex
@inproceedings{2024.EDM-posters.90,
 abstract = {Assessing the difficulty of reading comprehension questions is crucial to educational methodologies and language understanding technologies. Traditional methods of assessing question difficulty frequently rely on human judgments or shallow metrics, often failing to accurately capture the intricate cognitive demands of answering a question. This study delves into the task of automated question difficulty assessment, exploring the potential of leveraging Large Language Models (LLMs) to enhance the comprehension of the context and interconnections required to address a question. Our method incorporates multiple LLM-based difficulty measures and compares their performance on the FairytaleQA educational dataset with the ground truth benchmarks of human-annotated difficulty labels. Besides comparing different computational methods, this study also bridges the gap between machine and human understanding of question difficulty by analyzing the correlation between LLM-based measures and human perceptions. Our results provide valuable insights into the capabilities of LLMs in educational settings, particularly in the context of reading comprehension.},
 address = {Atlanta, Georgia, USA},
 author = {Andreea Dutulescu and Stefan Ruseti and Mihai Dascalu and Danielle Mcnamara},
 booktitle = {Proceedings of the 17th International Conference on Educational Data Mining},
 doi = {10.5281/zenodo.12729956},
 editor = {Benjamin PaaÃŸen and Carrie Demmans Epp},
 isbn = {978-1-7336736-5-5},
 month = {July},
 pages = {802--808},
 publisher = {International Educational Data Mining Society},
 title = {How Hard can this Question be? An Exploratory Analysis of Features Assessing Question Difficulty using LLMs},
 year = {2024}
}
```

