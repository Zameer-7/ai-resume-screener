import streamlit as st
import fitz  # PyMuPDF
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Curated concept-related keywords to check relevance
TECH_KEYWORDS = set([
    'python', 'pandas', 'numpy', 'scikit', 'tensorflow', 'pytorch', 'mlflow', 'flask',
    'fastapi', 'sql', 'nlp', 'nltk', 'huggingface', 'transformers', 'docker', 'aws', 'azure',
    'kubeflow', 'keras', 'precision', 'recall', 'f1', 'metrics', 'evaluation', 'deployment',
    'mlops', 'pipelines', 'model', 'machine', 'learning', 'regression', 'classification'
])

LEARNING_SUGGESTIONS = {
    "nltk": "Learn text processing techniques with NLTK for NLP tasks.",
    "spacy": "Understand entity recognition and POS tagging using spaCy.",
    "logistic": "Revise Logistic Regression — a fundamental ML algorithm.",
    "random": "Explore Random Forest for ensemble learning.",
    "regression": "Study Linear and Logistic Regression methods.",
    "nlp": "Understand NLP: tokenization, stemming, TF-IDF, etc.",
    "metrics": "Learn evaluation metrics like precision, recall, F1-score.",
    "text": "Understand text preprocessing like stopword removal and TF-IDF.",
    "scikit": "Practice training models and pipelines using scikit-learn.",
    "analysis": "Perform EDA with Pandas and Matplotlib.",
    "mlflow": "Track experiments and manage ML lifecycle with MLflow.",
    "huggingface": "Learn to use pre-trained transformer models for NLP.",
    "tensorflow": "Build deep learning models using TensorFlow.",
    "fastapi": "Deploy ML models with FastAPI backend.",
    "flask": "Create web apps for ML using Flask.",
    "sql": "Learn SQL for querying structured data.",
    "pytorch": "Build deep learning networks using PyTorch.",
    "mlops": "Understand the CI/CD pipeline for deploying ML models.",
    "docker": "Learn to use Docker for containerization.",
    "azure": "Explore Microsoft Azure for cloud computing solutions.",
    "aws": "Understand Amazon Web Services for cloud infrastructure.",
    "classification": "Study classification algorithms and techniques.",
    "pipelines": "Learn about data pipelines in machine learning."
}

def extract_text_from_pdf(uploaded_file):
    try:
        text = ""
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_entities(text):
    doc = nlp(text)
    name = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    email = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone = re.findall(r"\b\d{10}\b", text)
    return {
        "name": name[0] if name else "Not Found",
        "email": email[0] if email else "Not Found",
        "phone": phone[0] if phone else "Not Found"
    }

def extract_keywords(text):
    doc = nlp(text)
    tokens = [
        token.text.lower() for token in doc 
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and len(token.text) > 2
    ]
    filtered = list(set([t for t in tokens if t in TECH_KEYWORDS]))
    return filtered

def calculate_match_score(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectors[0], vectors[1])
    return round(score[0][0] * 100, 2)

def compare_concepts(resume_text, jd_keywords):
    resume_keywords = extract_keywords(resume_text)
    matched = [kw for kw in jd_keywords if kw in resume_keywords]
    missing = [kw for kw in jd_keywords if kw not in matched]
    return matched, missing

# ---- STREAMLIT APP UI ----

st.set_page_config(page_title="AI Resume Screener", page_icon="🧠", layout="centered")
st.title("🧠 AI Resume Screener")
st.markdown("Upload a **Resume (PDF)** and paste a **Job Description** to evaluate how well you match.")

resume_text = ""
jd_keywords = []

resume_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
if resume_file:
    resume_text = extract_text_from_pdf(resume_file)

jd_input = st.text_area("📋 Paste Job Description Here", height=200)
if jd_input:
    jd_keywords = extract_keywords(jd_input)

if resume_file and jd_input:
    if st.button("🚀 Start Screening"):
        with st.spinner("Analyzing your resume..."):
            info = extract_entities(resume_text)
            score = calculate_match_score(resume_text, jd_input)
            matched, missing = compare_concepts(resume_text, jd_keywords)

        st.success("✅ Resume analyzed successfully!")

        # Resume Info
        st.subheader("👤 Resume Info")
        st.write(f"**Name**: {info['name']}")
        st.write(f"**Email**: {info['email']}")
        st.write(f"**Phone**: {info['phone']}")

        # Match Score
        st.subheader("📊 Match Score")
        st.metric(label="Overall Match", value=f"{score} %")
        if score < 60:
            st.error("Better luck next time! Try improving your resume.")
        
        # Matched Concepts
        st.subheader("✅ Matched Concepts")
        st.write(", ".join(matched) if matched else "No matched concepts found.")

        # Missing Concepts
        st.subheader("❌ Missing Concepts in Resume")
        if missing:
            st.warning(", ".join(missing))
        else:
            st.success("You have covered all required concepts!")

        # Suggestions to Add
        st.subheader("📌 Suggestions to Improve Your Resume")
        if missing:
            suggestion_keywords = missing[:10]  # Limit to top 10 suggestions
            st.info("Consider adding these important keywords:\n" + ", ".join(suggestion_keywords))

            # Study Suggestions
            st.subheader("📚 Study These Topics to Improve")
            for kw in suggestion_keywords:
                if kw in LEARNING_SUGGESTIONS:
                    st.info(f"📖 *{kw}*: {LEARNING_SUGGESTIONS[kw]}")
                else:
                    st.warning(f"No study suggestion available for: {kw}")
        else:
            st.success("You have covered all required concepts!")
