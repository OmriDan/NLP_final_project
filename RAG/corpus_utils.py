import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from datasets import load_dataset

def prepare_knowledge_corpus(file_path=None, dataset_name=None, split=None):
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

    # Load HuggingFace dataset if provided
    if dataset_name:
        # Load dataset from Hugging Face
        hf_dataset = load_dataset(dataset_name, split=split or "train", trust_remote_code=True)

        # Extract text from dataset (adjust field name as needed)
        text_field = next(field for field in hf_dataset.features
                          if hf_dataset.features[field].dtype == 'string'
                          and field != 'label')

        # Split long texts
        text_splitter = RecursiveCharacterTextSplitter(
            # Increase chunk size for more complete code examples
            chunk_size=2000,
            chunk_overlap=300,
            # Add code-specific separators to preserve structure
            separators=[
                # Major section breaks
                "\n\n\n", "\n\n",
                # Code-specific separators
                "\nclass ", "\ndef ", "\nfunction ", "\nasync def ", "\n@",
                "\nimport ", "\nfrom ", "\n```", "\n#", "\n//", "\n/*",
                "\nif ", "\nfor ", "\nwhile ",
                # Then by paragraphs/lines
                "\n", " ", ""
            ],
            # Keep special tokens together
            keep_separator=True
        )
        for item in hf_dataset:
            if text_field in item and item[text_field]:
                chunks = text_splitter.split_text(item[text_field])
                documents.extend([Document(page_content=chunk) for chunk in chunks])

    return documents