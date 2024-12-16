import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

def load_and_split_pdfs(uploaded_files):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for uploaded_file in uploaded_files:
        # Save uploaded file to a temporary path
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Load PDF and split content
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        split_docs = text_splitter.split_documents(docs)
        documents.extend(split_docs)
        
        # Clean up temporary file
        os.remove(temp_path)
    
    return documents

def create_vectorstore(documents, persist_directory="db"):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    return vectorstore

def retrieve_answers(vectorstore, query):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
    docs = retriever.get_relevant_documents(query)
    answers = []
    for doc in docs:
      if doc.page_content not in answers:
           answers.append(doc.page_content)
    return answers

# Streamlit app
def main():
    st.title("Chat with PDFs (RAG)")

    # Upload PDFs
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Processing documents..."):
            documents = load_and_split_pdfs(uploaded_files)
            vectorstore = create_vectorstore(documents)
            st.success("Documents processed successfully!")

        # Query input
        query = st.text_input("Ask a question about the uploaded PDFs:")
        if query:
            with st.spinner("Retrieving answers..."):
                response = retrieve_answers(vectorstore, query)
                st.write("### Answer:")
                st.write(response)

if __name__ == "__main__":
    main()