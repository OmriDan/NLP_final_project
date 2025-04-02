import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS

class RAGRetriever:
    def __init__(self, documents, embedding_model="sentence-transformers/all-mpnet-base-v2"):
        # Initialize HuggingFace embeddings through LangChain
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Convert to LangChain documents if not already
        if not isinstance(documents[0], Document):
            self.documents = [
                Document(page_content=doc if isinstance(doc, str) else doc.text,
                         metadata=getattr(doc, 'metadata', {}) if not isinstance(doc, str) else {})
                for doc in documents
            ]
        else:
            self.documents = documents

        # Create FAISS index using LangChain
        self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)

    def retrieve(self, query, k=3):
        # Retrieve relevant documents
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        return [doc for doc, _ in docs_and_scores]