from sklearn.base import TransformerMixin
import numpy as np
from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel
import torch


class BERTVectorizer(TransformerMixin):
    def __init__(self, model_name="bert-base-uncased"):
        """
        Custom vectorizer for BERT embeddings.
        :param model_name: Name of the pre-trained BERT model (from Hugging Face).
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def fit(self, X, y=None):
        # No fitting required for pre-trained embeddings
        return self

    def transform(self, X):
        """
        Transforms input text into BERT embeddings.
        :param X: List of text inputs.
        :return: Numpy array of BERT [CLS] token embeddings.
        """
        embeddings = []
        for text in X:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()  # CLS token embedding
            embeddings.append(cls_embedding)
        return np.vstack(embeddings)
    
class Word2VecVectorizer(TransformerMixin):
    def __init__(self, model_path):
        """
        Custom vectorizer for Word2Vec embeddings.
        :param model_path: Path to the pre-trained Word2Vec model (in Gensim format).
        """
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)

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