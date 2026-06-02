from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import os
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

model = SentenceTransformer('all-MiniLM-L6-v2')

index = None
stored_chunks = []

def create_vector_store(chunks):
    global index, stored_chunks

    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    stored_chunks = chunks

def retrieve(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    results = [stored_chunks[i] for i in indices[0] if i != -1]
    return "\n".join(results)

def generate_answer(context, question):
    prompt = f"""
    Answer professionally using this resume context: 
    {context}

    Question: {question}
    """

    response = requests.post(
        "https://router.huggingface.co/v1/responses",
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={"inputs": prompt}
    )
    return response.json()
    # return response.json()[0]["generated_text"]
