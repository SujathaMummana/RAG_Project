import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

def load_and_split_pdfs(pdf_folder):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            split_docs = text_splitter.split_documents(docs)
            documents.extend(split_docs)
    
    return documents

def create_vectorstore(do