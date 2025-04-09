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

        for question, answer in zip(self.questions, self.answers):
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(question, k=self.k)

            # Extract context from retrieved documents
            context = " ".join([doc.page_content for doc in retrieved_docs])

            # Create augmented input
            augmented_input = {
                "question": question,
                "answer": answer,
                "context": context
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