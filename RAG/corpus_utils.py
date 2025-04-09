import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from datasets import load_dataset


def prepare_knowledge_corpus(file_path=None, dataset_name=None, split=None):
    """Prepare a knowledge corpus from local files or Hugging Face datasets.

    Handles specific dataset formats for SQuAD, code repositories, stack exchange, etc.
    """
    documents = []

    # Load local knowledge corpus if provided
    if file_path and os.path.exists(file_path):
        with open(file_path, "r") as f:
            content = f.read()

        # Use LangChain's text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        docs = text_splitter.split_text(content)
        documents.extend([Document(page_content=doc) for doc in docs])
        print(f"Loaded {len(documents)} documents from local file")

    # Load HuggingFace dataset if provided
    if dataset_name:
        print(f"Loading dataset: {dataset_name} ({split})")

        # Load dataset from Hugging Face
        hf_dataset = load_dataset(dataset_name, split=split or "train", trust_remote_code=True)
        text_splitter = get_text_splitter_for_dataset(dataset_name)

        # Handle specific dataset formats
        if dataset_name == "squad":
            # SQuAD dataset: use context as the document content
            for item in hf_dataset:
                doc = Document(
                    page_content=item["context"],
                    metadata={
                        "title": item.get("title", ""),
                        "id": item["id"] if "id" in item else "",
                        "question": item.get("question", ""),
                        "answers": item.get("answers", {})
                    }
                )
                documents.append(doc)

        elif dataset_name == "codeparrot/apps":
            # APPS dataset: combine problem statement and solutions
            for item in hf_dataset:
                content = f"Problem: {item['problem']}\n\nSolutions:\n{item.get('solutions', [''])[0]}"
                chunks = text_splitter.split_text(content)
                documents.extend([Document(
                    page_content=chunk,
                    metadata={"difficulty": item.get("difficulty", "")}
                ) for chunk in chunks])

        elif dataset_name == "codeparrot/github-jupyter-code-to-text":
            # Jupyter code to text dataset
            for item in hf_dataset:
                if "code" in item and "text" in item:
                    content = f"Documentation: {item['text']}\n\nCode:\n{item['code']}"
                    chunks = text_splitter.split_text(content)
                    documents.extend([Document(page_content=chunk) for chunk in chunks])

        elif dataset_name == "open-r1/verifiable-coding-problems-python-10k":
            # Python coding problems
            for item in hf_dataset:
                content = f"Question: {item.get('question', '')}\n\nResponse: {item.get('response', '')}"
                chunks = text_splitter.split_text(content)
                documents.extend([Document(page_content=chunk) for chunk in chunks])

        elif dataset_name == "sahil2801/CodeAlpaca-20k":
            # Code Alpaca dataset
            for item in hf_dataset:
                content = f"Instruction: {item.get('instruction', '')}\n\nInput: {item.get('input', '')}\n\nResponse: {item.get('output', '')}"
                chunks = text_splitter.split_text(content)
                documents.extend([Document(page_content=chunk) for chunk in chunks])

        elif dataset_name == "habedi/stack-exchange-dataset":
            # Stack Exchange dataset
            for item in hf_dataset:
                if "text" in item:
                    content = item["text"]
                    chunks = text_splitter.split_text(content)
                    documents.extend([Document(
                        page_content=chunk,
                        metadata={"tags": item.get("tags", [])}
                    ) for chunk in chunks])

        elif dataset_name == "ajibawa-2023/WikiHow":
            # WikiHow dataset
            for item in hf_dataset:
                title = item.get("title", "")
                steps = item.get("steps", [])
                methods = item.get("methods", [])

                content = f"Title: {title}\n\n"
                if methods:
                    for i, method in enumerate(methods):
                        content += f"Method {i + 1}: {method}\n"

                if steps:
                    content += "\nSteps:\n"
                    for i, step in enumerate(steps):
                        content += f"{i + 1}. {step}\n"

                chunks = text_splitter.split_text(content)
                documents.extend([Document(page_content=chunk) for chunk in chunks])

        else:
            # Generic handling for other datasets
            # Try to find text fields
            for item in hf_dataset:
                text_fields = [k for k, v in item.items()
                               if isinstance(v, str) and len(v) > 50]

                if text_fields:
                    # Use the longest text field
                    text_field = max(text_fields, key=lambda f: len(item[f]))
                    chunks = text_splitter.split_text(item[text_field])
                    documents.extend([Document(page_content=chunk) for chunk in chunks])

        print(f"Loaded {len(documents)} documents from {dataset_name}")

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