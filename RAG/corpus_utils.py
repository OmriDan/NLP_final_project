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
                for item in hf_dataset:
                    problem = item.get("problem", "")
                    solution = ""
                    if "solutions" in item and isinstance(item["solutions"], list) and item["solutions"]:
                        solution = item["solutions"][0]
                    content = f"Problem: {problem}\n\nSolution: {solution}"
                    chunks = text_splitter.split_text(content)
                    documents.extend([Document(
                        page_content=chunk,
                    ) for chunk in chunks])

            elif dataset_name == "open-r1/github-python-code-to-desc":
                for item in hf_dataset:
                    code = item.get("code", "")
                    description = item.get("desc", "")
                    content = f"Description: {description}\n\nCode: {code}"
                    chunks = text_splitter.split_text(content)
                    documents.extend([Document(page_content=chunk) for chunk in chunks])

            elif dataset_name == "math_dataset":
                for item in hf_dataset:
                    question = item.get("question", "")
                    answer = item.get("answer", "")
                    type_field = item.get("type", "")
                    content = f"Math problem type: {type_field}\nQuestion: {question}\nAnswer: {answer}"
                    chunks = text_splitter.split_text(content)
                    documents.extend([Document(page_content=chunk) for chunk in chunks])

            elif dataset_name == "allenai/science-qa":
                for item in hf_dataset:
                    question = item.get("question", "")
                    answer = item.get("answer", "")
                    lecture = item.get("lecture", "")
                    solution = item.get("solution", "")
                    content = f"Context question: {question}\nContext answer: {answer}\n Context explanation: {solution}\nAdditional Context: {lecture}"
                    chunks = text_splitter.split_text(content)
                    documents.extend([Document(page_content=chunk) for chunk in chunks])

            elif dataset_name == "EleutherAI/pile":
                for item in hf_dataset:
                    text = item.get("text", "")
                    chunks = text_splitter.split_text(text)
                    documents.extend([Document(page_content=chunk) for chunk in chunks])

            elif dataset_name == "gsm8k":
                for item in hf_dataset:
                    question = item.get("question", "")
                    answer = item.get("answer", "")
                    content = f"Math Problem: {question}\n\nSolution: {answer}"
                    chunks = text_splitter.split_text(content)
                    documents.extend([Document(page_content=chunk) for chunk in chunks])

            elif dataset_name == "wikihow/wikihow":
                for item in hf_dataset:
                    title = item.get("title", "")
                    text = item.get("text", "")
                    content = f"Topic: {title}\n{text}"
                    chunks = text_splitter.split_text(content)
                    documents.extend([Document(page_content=chunk) for chunk in chunks])

            elif dataset_name == "competition_math":
                for item in hf_dataset:
                    problem = item.get("problem", "")
                    solution = item.get("solution", "")
                    level = item.get("level", "")
                    source = item.get("source", "")
                    content = f"Math Competition: {source}, Level: {level}\nProblem: {problem}\nSolution: {solution}"
                    chunks = text_splitter.split_text(content)
                    documents.extend([Document(page_content=chunk) for chunk in chunks])

            else:
                # Generic handling for other datasets
                candidate_fields = [
                    'problem', 'question', 'answer', 'solution', 'solutions', 'gold_standard_solution',
                    'title', 'body', 'text', 'content', 'document', 'instruction', 'input', 'context',
                    'difficulty', 'difficulty_level', 'complexity', 'level', 'code', 'desc',
                    # Add dataset-specific fields
                    'problem_statement', 'lecture', 'explanation', 'source', 'type',
                    'mathematics_problem', 'mathematics_solution', 'proof',
                    'problem_id', 'input_output', 'url', 'starter_code',
                    # Math dataset fields
                    'module', 'level', 'category',
                    # Science fields
                    'theory', 'hypothesis', 'experiment',
                    # Generic fields
                    'python', 'function', 'implementation'
                ]

                for item in hf_dataset:
                    available_fields = [f for f in candidate_fields if f in item and item[f] is not None]

                    if available_fields:
                        text_fields = [f for f in available_fields if isinstance(item[f], str) and len(item[f]) > 10]

                        if text_fields:
                            # Define question and answer related field groups
                            question_fields = ['problem', 'question', 'instruction', 'input', 'problem_statement',
                                               'title', 'body', 'mathematics_problem']
                            answer_fields = ['answer', 'answers', 'correct_answer', 'solution', 'solutions', 'output', 'explanation',
                                             'gold_standard_solution', 'mathematics_solution', 'proof']

                            # Extract question content
                            question_content = []
                            for field in question_fields:
                                if field in item and isinstance(item[field], str) and len(item[field]) > 10:
                                    question_content.append(f"{field.capitalize()}: {item[field]}")

                            # Extract answer content
                            answer_content = []
                            for field in answer_fields:
                                if field in item and isinstance(item[field], str) and len(item[field]) > 10:
                                    answer_content.append(f"{field.capitalize()}: {item[field]}")

                            # Combine question and answer content
                            if question_content or answer_content:
                                full_content = "\n\n".join(question_content + answer_content)
                            else:
                                # Fallback to longest field if no question/answer fields identified
                                longest_field = max(text_fields, key=lambda f: len(item[f]))
                                full_content = item[longest_field]

                            chunks = text_splitter.split_text(full_content)
                            for chunk in chunks:
                                documents.append(Document(
                                    page_content=chunk,
                                    metadata={
                                        "source": dataset_name,
                                    }
                                ))

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