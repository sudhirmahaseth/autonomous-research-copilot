import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorStoreManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.db = None

    def initialize_with_texts(self, texts: list):
        """Documents ko chunk karke FAISS mein index karne ke liye"""
        docs = self.text_splitter.create_documents(texts)
        self.db = FAISS.from_documents(docs, self.embeddings)
        return self.db

    def retrieve_docs(self, query: str, k: int = 3):
        """Relevant documents search karne ke liye"""
        if not self.db:
            return "No documents indexed yet."
        retriever = self.db.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])
