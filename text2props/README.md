# Extensions to the *text2props* Framework

This repository extends the [*text2props*](https://github.com/lucabenedetto/text2props) framework, originally developed by [Benedetto et al.](https://doi.org/10.1007/978-3-030-52237-7_4), to enhance question difficulty estimation using custom vectorizers, fine-tuning of transformer-based embeddings, and batch experimentation. These extensions provide new capabilities for analyzing question difficulty across different domains, such as programming and mathematics.

- **Link to the original framework**: [*text2props* GitHub Repository](https://github.com/lucabenedetto/text2props).
- **Link to the original paper**: [Introducing a Framework to Assess Newly Created Questions with Natural Language Processing](https://doi.org/10.1007/978-3-030-52237-7_4).

## Key Features
- **Custom Vectorizers**: Extended the original vectorization methods with advanced embeddings like BERT, CodeBERT, and word2vec.
- **Fine-Tuning**: Scripts to fine-tune BERT and CodeBERT embeddings on domain-specific datasets (e.g., NovaSky).
- **Batch Experiments**: Automates running multiple experiments with different configurations, and saves evaluation metrics.
- **Data Processing**: Includes utility functions for parsing datasets into the *text2props* required format.

## Installation

To set up the environment for this project:

1. **Create a new virtual environment**:
   ```bash
   conda create -n my_project_env python=3.8
   conda activate my_project_env
   ```

2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the *text2props* framework**:
   ```bash
   python setup.py install
   ```

## Project Structure

- `scripts/`: Contains custom Python scripts for vectorization, fine-tuning, and batch experiments.
- `text2props/`: The original *text2props* framework and its source code.
- `requirements.txt`: Lists the dependencies for this project.

## Usage

### Running Fine-Tuning

To fine-tune transformer models (BERT or CodeBERT) on your dataset, run the following command:

```bash
python scripts/finetune.py
```

This script fine-tunes the selected transformer models on the **NovaSky** dataset (or other datasets you have in your `../data/raw/` folder) and saves the fine-tuned models to `../models/`.

### Running Batch Experiments

To run a batch of experiments with various vectorizers (TF-IDF, word2vec, BERT, CodeBERT), and compare different configurations, use:

```bash
python scripts/difficulty_prediction_batch_experiments.py
```

This script will:
- Load the dataset.
- Process the data using different vectorizers and models.
- Run the experiments with different pre-processing and fine-tuning options.
- Output the predictions and evaluation metrics to files.

### Data Format

The data should be formatted as follows:

- **Questions DataFrame**: Contains `[Q_ID, Q_TEXT, CORRECT_TEXTS, WRONG_TEXTS]`.
- **Answers DataFrame**: Contains `[S_ID, TIMESTAMP, CORRECT, Q_ID]`.
- **Latent Traits DataFrame**: Contains `[Q_ID, LATENT_TRAIT]`.

You can process your datasets using the utility functions provided in the `parse_data.py` file or use your own data parsing methods.


## Citation

If you use this framework in your work, please cite the original *text2props* paper:

```bibtex
@InProceedings{10.1007/978-3-030-52237-7_4,
  author="Benedetto, Luca and Cappelli, Andrea and Turrin, Roberto and Cremonesi, Paolo",
  editor="Bittencourt, Ig Ibert and Cukurova, Mutlu and Muldner, Kasia and Luckin, Rose and Mill{'a}n, Eva",
  title="Introducing a Framework to Assess Newly Created Questions with Natural Language Processing",
  booktitle="Artificial Intelligence in Education",
  year="2020",
  publisher="Springer International Publishing",
  address="Cham",
  pages="43--54",
  isbn="978-3-030-52237-7"
}
```

