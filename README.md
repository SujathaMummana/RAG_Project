Chat with PDFs (RAG)
This project demonstrates a Retrieval-Augmented Generation (RAG) application for querying PDF documents using a simple chatbot interface built with Streamlit. The application extracts content from PDFs, splits the text into manageable chunks, indexes it for similarity-based retrieval, and then answers user queries by retrieving relevant information from the uploaded documents.

Key Features
Upload Multiple PDFs: Users can upload one or more PDFs to process.
Document Chunking: Large documents are split into smaller chunks to ensure efficient indexing and retrieval.
Vector Store: Uses Chroma to store and retrieve document embeddings.
Question-Answering: Users can ask questions about the uploaded PDFs, and the app retrieves the most relevant chunks of text.
Code Walkthrough
1. Uploading and Processing PDFs
The app allows users to upload multiple PDFs. These files are processed and split into smaller chunks using RecursiveCharacterTextSplitter, ensuring that each chunk remains coherent while accommodating large documents.

python
Copy code
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
documents = load_and_split_pdfs(uploaded_files)
2. Temporary File Handling
Each uploaded file is temporarily saved, processed, and then deleted to optimize memory usage.

python
Copy code
temp_path = f"temp_{uploaded_file.name}"
with open(temp_path, "wb") as f:
    f.write(uploaded_file.read())
os.remove(temp_path)
3. Embedding and Storing Document Data
After splitting the documents, embeddings are created using the SentenceTransformerEmbeddings model (all-MiniLM-L6-v2). These embeddings are stored in a Chroma vector store for quick and efficient similarity-based retrieval.

python
Copy code
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="db")
4. Query Handling
Users can input questions about the uploaded documents. The app retrieves the top 3 most similar document chunks using the Chroma vector store and displays the content.

python
Copy code
query = st.text_input("Ask a question about the uploaded PDFs:")
response = retrieve_answers(vectorstore, query)
st.write("### Answer:", response)
5. Streamlit UI
The app uses Streamlit to create an intuitive and interactive user interface. Upload buttons, progress spinners, and text input fields provide a seamless user experience.

python
Copy code
st.title("Chat with PDFs (RAG)")
st.spinner("Processing documents...")
st.success("Documents processed successfully!")
How It Works
File Upload: Upload one or more PDFs via the Streamlit interface.
PDF Processing: Extract content, split it into chunks, and generate vector embeddings.
Vector Store: Save embeddings in a Chroma vector store for similarity-based search.
Ask Questions: Input a query, and the app retrieves and displays the most relevant answers.
Requirements
Python 3.8+
Streamlit
LangChain
Chroma
SentenceTransformers
Flask
encorewebsm
Install the dependencies using:

bash
Copy code
pip install streamlit langchain chromadb sentence-transformers flask encorewebsm
How to Run
Clone this repository:

bash
Copy code
git clone <repository-url>
cd <repository-folder>
Install the required Python packages:

bash
Copy code
pip install -r requirements.txt
Run the app using Streamlit:

bash
Copy code
streamlit run app.py
Open the URL provided by Streamlit (typically http://localhost:8501) in your browser.

Folder Structure
bash
Copy code
.
├── app.py               # Main application file
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
Future Improvements
Support for additional document formats (e.g., Word, TXT).
Integration with larger language models for more robust question answering.
Deployment to cloud platforms for public access.
Enhanced UI with additional features like file preview and results filtering.
Acknowledgments
This project uses the following libraries and frameworks:

LangChain for document processing and text splitting.
Chroma for vector storage and retrieval.
Sentence Transformers for generating text embeddings.
Streamlit for creating the user interface.
Flask and encorewebsm for future backend integration.