import pytesseract
from PIL import Image
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from perplexity import ask_perplexity

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

DIM = 384
index = faiss.IndexFlatL2(DIM)

documents = []


# ---------------------------
# Text Extraction
# ---------------------------
def extract_text(file):

    if file.type == "application/pdf":
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text

    elif "image" in file.type:
        img = Image.open(file)
        return pytesseract.image_to_string(img)

    else:
        return file.read().decode(errors="ignore")


# ---------------------------
# Chunking
# ---------------------------
def chunk_text(text, chunk_size=400, overlap=50):

    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks


# ---------------------------
# Processing Files
# ---------------------------
def process_files(files):

    global documents
    documents.clear()
    index.reset()
    all_chunks = []
    for file in files:
        text = extract_text(file)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    documents = all_chunks
    embeddings = model.encode(all_chunks)
    embeddings = np.array(embeddings).astype("float32")
    index.add(embeddings)


# ---------------------------
# Query RAG
# ---------------------------
def query_rag(query):

    q_emb = model.encode([query])
    q_emb = np.array(q_emb).astype("float32")
    D, I = index.search(q_emb, k=5)

    context = ""
    for i in I[0]:
        if i < len(documents):
            context += documents[i] + "\n\n"

    return ask_perplexity(query, context)
