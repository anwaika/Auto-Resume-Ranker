from flask import Flask, render_template, request
import os
import PyPDF2
import docx
import nltk
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Ridge
from sentence_transformers import SentenceTransformer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------
# LOAD AI MODEL
# ---------------------------

print("Loading semantic AI model...")
ai_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# ---------------------------
# SKILL DATABASE
# ---------------------------

SKILLS = {
    "python": 5, "sql": 5, "statistics": 4, "machine learning": 5,
    "data analysis": 4, "data visualization": 3, "data cleaning": 3,
    "tableau": 3, "power bi": 3, "excel": 3,
    "pandas": 4, "numpy": 4, "matplotlib": 3, "seaborn": 3, "scikit-learn": 4,
    "deep learning": 5, "tensorflow": 4, "pytorch": 4, "nlp": 4,
    "postgresql": 4, "mysql": 3, "mongodb": 3,
    "aws": 3, "azure": 3, "gcp": 3,
    "spark": 4, "hadoop": 3, "git": 2, "docker": 3,
    "a/b testing": 4, "hypothesis testing": 4, "time series": 4
}

SKILL_MAX = sum(SKILLS.values())

# ---------------------------
# TEXT EXTRACTION
# BUG FIX: only lowercase the extension check, not the file path itself
# ---------------------------

def extract_text(file_path):

    ext = file_path.lower()

    if ext.endswith(".pdf"):
        text = ""
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
        except Exception as e:
            print(f"PDF read error for {file_path}: {e}")
        return text

    elif ext.endswith(".docx"):
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"DOCX read error for {file_path}: {e}")
            return ""

    elif ext.endswith(".txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"TXT read error for {file_path}: {e}")
            return ""

    return ""


# ---------------------------
# TEXT PREPROCESSING
# ---------------------------

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    tokens = nltk.word_tokenize(text)
    return " ".join(tokens)


# ---------------------------
# SKILL SCORE
# ---------------------------

def skill_score(text):
    score = 0
    for skill, weight in SKILLS.items():
        if skill in text:
            score += weight
    return score / SKILL_MAX


# ---------------------------
# EXPERIENCE EXTRACTION
# ---------------------------

def extract_experience_years(text):
    matches = re.findall(r'(\d+)\+?\s*years?', text.lower())
    if matches:
        return min(int(max(matches, key=int)), 15)
    return 0


# ---------------------------
# RANKING FUNCTION
# BUG FIX 1: removed backtick syntax error around experience_scores
# BUG FIX 2: replaced self-fitted Ridge with a transparent weighted sum
#            Ridge.fit(X, semantic_scores) then predict(X) was just
#            reproducing semantic_scores — not combining signals at all.
#            Weights below are explicit and honest:
#              45% semantic  (best single signal for meaning)
#              35% TF-IDF    (keyword/domain overlap)
#              15% skill     (explicit skill match)
#               5% experience (years as a soft tiebreaker)
# ---------------------------

WEIGHTS = {
    "tfidf":      0.35,
    "semantic":   0.45,
    "skill":      0.15,
    "experience": 0.05,
}

def rank_resumes(job_desc_raw, job_desc_processed, resumes_processed, resumes_raw):

    documents = [job_desc_processed] + resumes_processed

    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(documents)

    # SVD dimensionality reduction
    max_components = min(100, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)

    if max_components >= 2:
        svd = TruncatedSVD(n_components=max_components)
        reduced_matrix = svd.fit_transform(tfidf_matrix)
        job_vector = reduced_matrix[0].reshape(1, -1)
        resume_vectors = reduced_matrix[1:]
        tfidf_scores = cosine_similarity(job_vector, resume_vectors)[0]
    else:
        job_vector = tfidf_matrix[0]
        resume_vectors = tfidf_matrix[1:]
        tfidf_scores = cosine_similarity(job_vector, resume_vectors)[0]

    # Semantic similarity
    job_embedding = ai_model.encode([job_desc_raw])[0]
    resume_embeddings = ai_model.encode(resumes_raw)
    semantic_scores = cosine_similarity([job_embedding], resume_embeddings)[0]

    # Skill scores
    skill_scores = np.array([skill_score(r) for r in resumes_processed])

    # Experience scores  (BUG FIX: backtick syntax error removed)
    experience_scores = np.array([
        extract_experience_years(r) / 15 for r in resumes_raw
    ])

    # Weighted combination  (BUG FIX: replaced self-fitted Ridge)
    final_scores = (
        WEIGHTS["tfidf"]      * tfidf_scores +
        WEIGHTS["semantic"]   * semantic_scores +
        WEIGHTS["skill"]      * skill_scores +
        WEIGHTS["experience"] * experience_scores
    )

    # Score breakdown per resume
    breakdown = []
    for i in range(len(resumes_raw)):
        breakdown.append({
            "tfidf_svd":  round(float(tfidf_scores[i]) * 100, 1),
            "semantic":   round(float(semantic_scores[i]) * 100, 1),
            "skill":      round(float(skill_scores[i]) * 100, 1),
            "experience": int(experience_scores[i] * 15),
        })

    return final_scores, breakdown


# ---------------------------
# MAIN ROUTE
# ---------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    results = []

    if request.method == "POST":

        job_desc_raw = request.form.get("job_description", "")
        if not job_desc_raw:
            return render_template("index.html", results=[])

        job_desc_processed = preprocess(job_desc_raw)

        files = request.files.getlist("resumes")

        resumes_processed = []
        resumes_raw       = []
        names             = []

        for file in files:
            if file.filename == "":
                continue

            # BUG FIX: save with original filename (not lowercased path)
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            raw_text = extract_text(path)

            # Clean up uploaded file after reading
            try:
                os.remove(path)
            except Exception:
                pass

            if not raw_text.strip():
                continue

            processed_text = preprocess(raw_text)
            resumes_raw.append(raw_text)
            resumes_processed.append(processed_text)
            names.append(file.filename)

        if len(resumes_raw) > 0:

            final_scores, breakdown = rank_resumes(
                job_desc_raw,
                job_desc_processed,
                resumes_processed,
                resumes_raw
            )

            combined = list(zip(names, final_scores, breakdown))
            combined.sort(key=lambda x: x[1], reverse=True)

            results = [
                {
                    "rank":        i + 1,
                    "name":        item[0],
                    "final_score": round(float(item[1]) * 100, 2),
                    "tfidf_svd":   item[2]["tfidf_svd"],
                    "semantic":    item[2]["semantic"],
                    "skill":       item[2]["skill"],
                    "experience":  item[2]["experience"],
                }
                for i, item in enumerate(combined)
            ]

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)