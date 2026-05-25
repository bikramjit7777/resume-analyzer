import streamlit as st
from utils import extract_text_from_pdf, split_text
from rag_pipeline import create_vector_store, retrieve, generate_answer

st.title("Resume Analyzer + Interview AI Assistant")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = split_text(text)

    create_vector_store(chunks)
    st.success("Resume processed successfully!")

question = st.text_input("Ask interview questions:")

