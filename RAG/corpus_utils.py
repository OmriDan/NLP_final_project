import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from datasets import load_dataset


def prepare_knowledge_corpus(file_path=None, dataset_name=None, split=None):
    """Prepare a knowledge corpus from local files or Hugging Face datasets."""
    documents = []

    if file_path and os.path.exists(file_path):
        with open(file_path, "r") as f:
            content = f.read()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        docs = text_splitter.split_text(content)
        documents.extend([Document(page_content=doc) for doc in docs])
        print(f"Loaded {len(documents)} documents from local file")

    if dataset_name:
        print(f"Loading dataset: {dataset_name} ({split})")
        try:
            hf_dataset = load_dataset(dataset_name, split=split or "train", trust_remote_code=True)
            text_splitter = get_text_splitter_for_dataset(dataset_name)

            # Inspect first item to understand structure
            if len(hf_dataset) > 0:
                first_item = hf_dataset[0]
                print(f"Dataset structure sample: {list(first_item.keys())}")

            if dataset_name == "codeparrot/apps":
                # Fixed APPS dataset handling
                for item in hf_dataset:
                    # Check actual keys in the dataset
                    if "problem_statement" in item:
                        problem = item["problem_statement"]
                    elif "question" in item:
                        problem = item["question"]
                    else:
                        # Use the first long text field found
                        text_fields = [k for k, v in item.items() if isinstance(v, str) and len(v) > 50]
                        problem = item.get(text_fields[0], "") if text_fields else ""

                    # Try to get solutions
                    solutions = ""
                    if "solutions" in item and isinstance(item["solutions"], list) and item["solutions"]:
                        solutions = item["solutions"][0]
                    elif "answer" in item:
                        solutions = item["answer"]

                    content = f"Problem: {problem}\n\nSolutions: {solutions}"
                    chunks = text_splitter.split_text(content)
                    documents.extend([Document(
                        page_content=chunk,
                        metadata={"difficulty": item.get("difficulty", "")}
                    ) for chunk in chunks])

            elif dataset_name == "squad":
                for item in hf_dataset:
                    if "context" in item:
                        doc = Document(
                            page_content=item["context"],
                            metadata={
                                "title": item.get("title", ""),
                                "id": item.get("id", ""),
                                "question": item.get("question", ""),
                                "answers": item.get("answers", {})
                            }
                        )
                        documents.append(doc)

            # Handle other datasets...
            # (keeping the existing implementations for other datasets)

            print(f"Loaded {len(documents)} documents from {dataset_name}")

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {str(e)}")

    return documents


def get_text_splitter_for_dataset(dataset_name):
    """Get an appropriate text splitter based on dataset type"""
    if "code" in dataset_name.lower() or dataset_name in ["codeparrot/apps", "codeparrot/github-jupyter-code-to-text",
                                                          "open-r1/verifiable-coding-problems-python-10k"]:
        # Code-specific text splitter with larger chunks
        return RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=300,
            separators=[
                "\n\n\n", "\n\n",
                "\nclass ", "\ndef ", "\nfunction ", "\nasync def ", "\n@",
                "\nimport ", "\nfrom ", "\n```", "\n#", "\n//", "\n/*",
                "\nif ", "\nfor ", "\nwhile ",
                "\n", " ", ""
            ],
            keep_separator=True
        )
    else:
        # General purpose text splitter
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )