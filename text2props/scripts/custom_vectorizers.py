from sklearn.base import TransformerMixin
import numpy as np
from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel,AutoTokenizer, AutoModel
import gensim.downloader as api
import torch
from tqdm import tqdm


class BERTVectorizer(TransformerMixin):
    def __init__(self, model_name="bert-base-uncased", preprocessor=None, model_path=None):
        """
        Custom vectorizer for BERT embeddings.
        :param model_name: Name of the pre-trained BERT model (from Hugging Face).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_path:
            print(f"[INFO] Loading fine-tuned BERT model from: {model_path}")
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertModel.from_pretrained(model_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
        self.preprocessor = preprocessor
        self.device = device
        self.model.to(self.device)

    def fit(self, X, y=None):
        # No fitting required for pre-trained embeddings
        return self

    def transform(self, X, batch_size=24):
        """
        Transforms input text into BERT embeddings.
        :param X: List of text inputs.
        :return: Numpy array of BERT [CLS] token embeddings.
        """
        embeddings = []
        for i in tqdm(range(0, len(X), batch_size), desc="Processing batches"):
            batch_texts = X[i:i + batch_size].tolist()
            inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)


class CodeBERTVectorizer(TransformerMixin):
    def __init__(self, model_name="microsoft/codebert-base", preprocessor=None, model_path=None):
        """
        Custom vectorizer for CodeBERT embeddings.
        :param model_name: Name of the pre-trained CodeBERT model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_path:
            print(f"[INFO] Loading fine-tuned CodeBERT model from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        self.preprocessor = preprocessor
        self.device = device
        self.model.to(self.device)

    def fit(self, X, y=None):
        # No fitting required for pre-trained embeddings
        return self

    def transform(self, X, batch_size=24):
        """
        Transforms input text into CodeBERT embeddings.
        :param X: List of text inputs.
        :return: Numpy array of CodeBERT [CLS] token embeddings.
        """
        embeddings = []
        for i in tqdm(range(0, len(X), batch_size), desc="Processing batches"):
            batch_texts = X[i:i + batch_size].tolist()
            inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)

    
class Word2VecVectorizer(TransformerMixin):
    def __init__(self, model_name="word2vec-google-news-300"):
        """
        Custom vectorizer for Word2Vec embeddings.
        :param model_name: Name of the pre-trained Word2Vec model (from Gensim's API).
        """
        print(f"[INFO] Loading Word2Vec model: {model_name}")
        self.model = api.load(model_name)  # Automatically downloads and loads the model

    def fit(self, X, y=None):
        # No fitting required for pre-trained embeddings
        return self

    def transform(self, X):
        """
        Transforms input text into Word2Vec embeddings.
        :param X: List of text inputs.
        :return: Numpy array of averaged Word2Vec embeddings.
        """
        return np.array([self._embed_text(text) for text in X])

    def _embed_text(self, text):
        """
        Converts a single text input into an embedding by averaging word embeddings.
        :param text: Input text.
        :return: Averaged Word2Vec embedding vector.
        """
        words = text.split()  # Tokenize the text
        embeddings = [self.model[word] for word in words if word in self.model]
        if embeddings:
            return np.mean(embeddings, axis=0)  # Average the embeddings for all words in the text
        else:
            return np.zeros(self.model.vector_size)  # Return a zero vector if no words are in the model
