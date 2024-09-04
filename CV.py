import os
import streamlit as st
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from docx import Document

# Set environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the summarization model
@st.cache_resource
def load_summarizer():
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        st.error(f"Error loading summarization model: {e}")
        return None

summarizer = load_summarizer()

def summarize_text(text, max_chunk_length=1000):
    if not summarizer:
        st.error("Summarizer model is not available.")
        return ""
    
    # Split text into chunks
    text_chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    
    # Summarize each chunk
    summaries = []
    for chunk in text_chunks:
        chunk_length = len(chunk.split())
        max_length = min(150, chunk_length // 2)  # Adjust max_length based on chunk length
        min_length = max(30, chunk_length // 4)   # Adjust min_length based on chunk length
        try:
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            st.error(f"Error summarizing text: {e}")
    
    # Combine summaries
    return " ".join(summaries)

def vectorize_and_match(job_desc_summary, resume_summary):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([job_desc_summary, resume_summary])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])
        return similarity[0][0]
    except Exception as e:
        st.error(f"Error in vectorizing and matching: {e}")
        return 0.0

def extract_text_from_pdf(file):
    try:
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(file):
    try:
        doc = Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text(file):
    try:
        if file.type == "application/pdf":
            return extract_text_from_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return extract_text_from_docx(file)
        else:
            return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

def main():
    st.title("AI Resume Matcher")

    st.header("Upload Job Description and Resumes")
    job_desc_files = st.file_uploader("Upload Job Description", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    resume_files = st.file_uploader("Upload Resumes", type=["txt", "pdf", "docx"], accept_multiple_files=True)

    if job_desc_files and resume_files:
        job_desc_text = ""
        for file in job_desc_files:
            job_desc_text += extract_text(file) + "\n"

        st.subheader("Processing Job Description...")
        job_desc_summary = summarize_text(job_desc_text)

        resume_match_scores = []
        for file in resume_files:
            st.write(f"Processing {file.name}...")
            resume_text = extract_text(file)
            resume_summary = summarize_text(resume_text)
            match_score = vectorize_and_match(job_desc_summary, resume_summary)
            resume_match_scores.append((file.name, match_score))

        # Sort resumes by match score in descending order
        resume_match_scores.sort(key=lambda x: x[1], reverse=True)

        st.subheader("Top Matches for Resumes")
        for i, (file_name, match_score) in enumerate(resume_match_scores):
            st.write(f"{i + 1}. {file_name} - Match Score: {match_score:.2f}")

if __name__ == "__main__":
    main()
