import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import streamlit as st

# Define URLs for Crawling
urls = [
    "https://www.uchicago.edu/",
    "https://www.washington.edu/",
    "https://www.stanford.edu/",
    "https://und.edu/"
]

# Step 1: Crawl and Scrape Content
def scrape_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    else:
        return ""

# Step 2: Text Splitting
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Load Models and Set Up Index
@st.cache_resource
def setup_pipeline():
    all_texts = [scrape_website(url) for url in urls if scrape_website(url)]
    all_chunks = []
    for text in all_texts:
        chunks = split_text(text)
        all_chunks.extend(chunks)

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(all_chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype='float32')

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    qa_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    return embedding_model, index, all_chunks, qa_pipeline

embedding_model, index, all_chunks, qa_pipeline = setup_pipeline()

# Step 3: Query Handling
def query_pipeline(user_query):
    query_embedding = embedding_model.encode([user_query], show_progress_bar=False)
    distances, indices = index.search(np.array(query_embedding, dtype='float32'), k=3)
    retrieved_chunks = [all_chunks[idx] for idx in indices[0]]
    return retrieved_chunks

# Step 4: Generate Response
def generate_response(retrieved_texts, user_query):
    context = " ".join(retrieved_texts)[:1024]
    prompt = f"Context: {context}\nQuery: {user_query}"
    response = qa_pipeline(prompt, max_length=150, min_length=40, do_sample=False)
    return response[0]['summary_text']

# Streamlit UI
st.title("University Information Query System")
st.write("Enter your query below to search across selected university websites.")

user_query = st.text_input("Your Query:")
if st.button("Submit"):
    if user_query.strip():
        st.write("Processing your query...")
        retrieved_texts = query_pipeline(user_query)
        response = generate_response(retrieved_texts, user_query)
        st.subheader("Response")
        st.write(response)
    else:
        st.error("Please enter a valid query!")