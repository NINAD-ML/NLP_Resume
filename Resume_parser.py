import streamlit as st
import spacy
import re
from pdfminer.high_level import extract_text
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Load the SpaCy model
nlp = spacy.load("en_core_web_trf")

# Predefined skill list
skills_list = [
    "Python", "Java", "C++", "JavaScript", "C#", "SQL", "R", "Scala", "HTML", "CSS", "React", "Angular",
    "Django", "Flask", "Node.js", "TensorFlow", "Keras", "PyTorch", "AWS", "Azure", "Google Cloud", "Linux",
    "Docker", "Kubernetes", "MongoDB", "PostgreSQL", "MySQL", "Hadoop", "Spark", "Git", "CI/CD", "Machine Learning",
    "Deep Learning", "Data Science", "NLP", "Computer Vision", "Big Data", "Tableau", "Power BI", "Excel", "Data Analysis"
]

# Function to extract named entities using SpaCy
def extract_entities(text):
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities

# Function to extract skills from the resume using the predefined skills list
def extract_skills(text, skill_list):
    found_skills = []
    for skill in skill_list:
        if skill.lower() in text.lower():
            found_skills.append(skill)
    return found_skills

# Function to extract email addresses using regex
def extract_email(text):
    match = re.findall(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', text, re.IGNORECASE)
    return match[0] if match else None

# Function to extract phone numbers using regex
def extract_phone(text):
    match = re.findall(r'\b\d{10}\b', text)
    return match[0] if match else None

# Function to match resume to job description using cosine similarity
def match_resume_to_job(resume_text, job_desc):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
    similarity_score = cosine_similarity(tfidf_matrix)[0][1]
    return similarity_score

# Streamlit app interface
st.title("Resume Parser & Analyzer")

# File uploader for PDF or DOCX file
uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx"])

if uploaded_file:
    # Extract text based on the file type
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(uploaded_file)

    # Display extracted text
    st.text_area("Extracted Text", text, height=300)

    # Extract email and phone number from the text
    email = extract_email(text)
    phone = extract_phone(text)

    st.write(f"Email: {email}")
    st.write(f"Phone: {phone}")

    # Extract named entities from the text
    entities = extract_entities(text)
    st.write("Extracted Entities:")
    st.json(entities)

    # Extract skills using the predefined list
    skills = extract_skills(text, skills_list)
    st.write("Extracted Skills:")
    st.write(skills)

    # Example job description for matching
    job_desc = "Looking for a data scientist skilled in Python, NLP, and SQL."
    similarity = match_resume_to_job(text, job_desc)
    st.write(f"Resume matches the job description with a similarity score of: {similarity:.2f}")
