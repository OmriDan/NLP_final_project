import torch
from torch.utils.data import Dataset

class RAGAugmentedDataset(Dataset):
    def __init__(self, questions, answers, retriever, labels=None, tokenizer=None, max_length=512, k=3):
        self.questions = questions
        self.answers = answers
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.retriever = retriever
        self.k = k

        # Pre-compute augmented inputs
        self.augmented_inputs = self._create_augmented_inputs()

        # Convert categorical labels to continuous values if provided
        self.continuous_labels = None
        if self.labels is not None:
            label_mapping = {0: 0.16, 1: 0.5, 2: 0.83}
            self.continuous_labels = [label_mapping.get(label, label) for label in self.labels]

    def _create_augmented_inputs(self):
        augmented_inputs = []
        # Add few-shot examples - these provide calibration points for the model
        examples = [
            {"q": "Print 'Hello World'",
             "a": "print('Hello World')",
             "diff": 0.1,
             "reason": "Very basic syntax with no algorithmic complexity"},

            {"q": "Write a function to check if a string is a palindrome",
             "a": "def is_palindrome(s): return s == s[::-1]",
             "diff": 0.3,
             "reason": "Simple algorithm using basic string operations"},

            {"q": "Implement binary search on a sorted array",
             "a": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1",
             "diff": 0.6,
             "reason": "Medium difficulty with divide-and-conquer approach"},

            {"q": "Implement a balanced binary search tree with insertion and deletion",
             "a": "class TreeNode:\n    def __init__(self, key):\n        self.left = None\n        self.right = None\n        self.val = key\n\nclass BST:\n    def insert(self, root, key):\n        if root is None:\n            return TreeNode(key)\n        else:\n            if root.val < key:\n                root.right = self.insert(root.right, key)\n            else:\n                root.left = self.insert(root.left, key)\n        return root",
             "diff": 0.9,
             "reason": "Complex data structure requiring careful balance maintenance"}
        ]

        # Create example demonstrations text
        examples_text = "Example difficulty ratings:\n"
        for ex in examples:
            examples_text += f"Question: {ex['q']}\nSolution: {ex['a']}\nDifficulty: {ex['diff']} - {ex['reason']}\n\n"

        for question, answer in zip(self.questions, self.answers):
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(question, k=self.k)

            # Extract context from retrieved documents
            context = " ".join([doc.page_content for doc in retrieved_docs])
            # Add structured prompt
            task_prompt = (
                "Rate the difficulty of the following programming question on a scale from 0.0 (very easy) to 1.0 (very difficult). "
                "Consider these factors: algorithm complexity, time/space efficiency, code length, required knowledge, "
                "and implementation challenges. "
                "Easy problems (0.0-0.3) typically use basic syntax and simple operations. "
                "Medium problems (0.3-0.7) require data structures, loops, or standard algorithms. "
                "Hard problems (0.7-1.0) requires expertise, involve complex algorithms, optimizations, or advanced concepts."
            )
            # Create augmented input
            augmented_input = {
                "question": question,
                "answer": answer,
                "context": f"{task_prompt}\n\n{examples_text}\nRelevant Information:\n{context}"
            }

            augmented_inputs.append(augmented_input)

        return augmented_inputs

    def __len__(self):
        return len(self.augmented_inputs)

    def __getitem__(self, idx):
        augmented_input = self.augmented_inputs[idx]

        # Step 1: First prioritize the question (most important)
        question_text = f"Question: {augmented_input['question']}"
        question_encoding = self.tokenizer(
            question_text,
            truncation=True,
            max_length=self.max_length // 2,  # Allocate up to half for question
            return_length=True
        )
        question_length = question_encoding["length"][0]
        question_truncated = self.tokenizer.decode(
            question_encoding["input_ids"],
            skip_special_tokens=True
        )

        # Step 2: Then allocate space for answer
        remaining_tokens = self.max_length - question_length - 2  # Account for separator
        answer_text = f"Answer: {augmented_input['answer']}"
        answer_encoding = self.tokenizer(
            answer_text,
            truncation=True,
            max_length=remaining_tokens // 2,  # Allocate up to half of remaining for answer
            return_length=True
        )
        answer_length = answer_encoding["length"][0]
        answer_truncated = self.tokenizer.decode(
            answer_encoding["input_ids"],
            skip_special_tokens=True
        )

        # Step 3: Use any remaining tokens for context
        remaining_tokens = self.max_length - question_length - answer_length - 3  # Account for separators
        context_text = f"Context: {augmented_input['context']}"
        context_encoding = self.tokenizer(
            context_text,
            truncation=True,
            max_length=remaining_tokens,
            return_length=True
        )
        context_truncated = self.tokenizer.decode(
            context_encoding["input_ids"],
            skip_special_tokens=True
        )

        # Step 4: Combine all components in final format
        text = f"{context_truncated} {question_truncated} {answer_truncated}"

        # Final encoding
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Convert dict of tensors to tensors and remove batch dimension
        item = {k: v.squeeze(0) for k, v in encoding.items()}

        if self.continuous_labels is not None:
            item['labels'] = torch.tensor(self.continuous_labels[idx], dtype=torch.float)

        return item