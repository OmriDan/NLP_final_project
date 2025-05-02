# Notebook Overview

This folder contains the `Llama7B_testing.ipynb` notebook, which runs a Google Colab workflow to estimate question difficulty using the Llama-2-7b-chat model on a Data Structures exam dataset.

## Purpose

* **Dataset Loading:** Mount Google Drive and load `DS_tests_with_difficulty.csv`.
* **Filtering:** Select only open-ended questions for evaluation.
* **Prompt Engineering:** Define a template that feeds each question and answer into Llama.
* **Inference:** Generate an “Estimated Difficulty” score via the Hugging Face pipeline.
* **Evaluation:** Compute MSE, RMSE, and MAE against ground-truth difficulty labels.

## Prerequisites

* Python ≥ 3.7 (Colab runtime is fine)
* A Hugging Face account with a valid token (for model download)
* Access to Google Drive containing your dataset

## Dependencies

```bash
pip install transformers accelerate datasets scikit-learn matplotlib huggingface-hub
```

*(Or pin versions in `requirements-notebook.txt` via `pip freeze > requirements-notebook.txt`.)*

## Colab Setup

1. **Mount Google Drive**

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. **Login to Hugging Face**

   ```python
   from huggingface_hub import login
   login(token="YOUR_TOKEN_HERE")
   ```
3. **Install Libraries**

   ```bash
   !pip install -q transformers accelerate datasets scikit-learn matplotlib huggingface-hub
   ```

## Running the Notebook

1. Open `Llama7B_testing.ipynb` in Colab.
2. Update the `file_path` variable to point at your CSV in Drive (default: `/content/drive/MyDrive/NLP_PROJECT/DS_tests_with_difficulty.csv`).
3. Run cells in order: setup, data loading, prompt definition, model loading, inference loop, and evaluation.

## Notebook Structure

1. **Install & Import** – Installs packages and imports all required modules.
2. **Mount & Load Data** – Mounts Drive and reads the CSV into a DataFrame.
3. **Filter Questions** – Keeps only open-ended questions and fills missing values.
4. **Prompt Template & Extraction** – Sets `PROMPT_TEMPLATE` and implements `extract_estimated_difficulty(text)`.
5. **Load Model & Pipeline** – Loads Llama-2-7b-chat via `AutoModelForCausalLM` and creates a `pipeline("text-generation")`.
6. **Inference Loop** – Iterates over each question, generates text, parses out the numeric estimate.
7. **Metrics Computation** – Calculates MSE, RMSE, MAE with scikit-learn and displays results.

## Data Requirements

* **CSV file:** Must include `question_type`, `question_translated_to_English`, `answer_translated_to_English`, and `Difficulty` columns.
* **File path:** Adjust the `file_path` variable in the data-loading cell as needed.

## Customization

* Change the prompt template or regex in the extraction function to match your desired output format.
* Modify inference parameters (e.g., `max_new_tokens`) in the pipeline call.
* Swap in a different model checkpoint by updating `model_name`.

## Troubleshooting

* **Model Download Errors:** Ensure `huggingface_hub.login()` is called with a valid token.
* **FileNotFoundError:** Verify the CSV path and Drive mount point.
* **Kernel/Runtime Issues:** Restart the Colab runtime and rerun all cells.
