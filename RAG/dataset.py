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

        # Combine question, retrieved context, and answer
        text = f"Context: {augmented_input['context']} Question: {augmented_input['question']} Answer: {augmented_input['answer']}"

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