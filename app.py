from flask import Flask, render_template, request
import os
import PyPDF2
import docx
import nltk
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------
# SKILL DATABASE (Weighted)
# ---------------------------

SKILLS = {
    "python":5,
    "sql":5,
    "statistics":4,
    "machine learning":5,
    "data analysis":4,
    "tableau":3,
    "power bi":3,
    "excel":3,
    "pandas":4,
    "numpy":4,
    "deep learning":5,
    "tensorflow":4,
    "pytorch":4
}


# ---------------------------
# TEXT EXTRACTION
# ---------------------------

def extract_text(file_path):

    if file_path.endswith(".pdf"):
        text = ""
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text()
        return text

    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf8") as f:
            return f.read()

    return ""


# ---------------------------
# PREPROCESSING
# ---------------------------

def preprocess(text):

    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    tokens = nltk.word_tokenize(text)

    return " ".join(tokens)


# ---------------------------
# SKILL SCORING
# ---------------------------

def skill_score(resume_text):

    score = 0

    for skill, weight in SKILLS.items():
        if skill in resume_text:
            score += weight

    return score


# ---------------------------
# RANKING FUNCTION
# ---------------------------

def rank_resumes(job_desc, resumes):

    documents = [job_desc] + resumes

    tfidf = TfidfVectorizer(stop_words="english")

    tfidf_matrix = tfidf.fit_transform(documents)

    # ---- SVD (Math syllabus) ----
    svd = TruncatedSVD(n_components=100)

    reduced_matrix = svd.fit_transform(tfidf_matrix)

    job_vector = reduced_matrix[0].reshape(1, -1)
    resume_vectors = reduced_matrix[1:]

    similarity_scores = cosine_similarity(job_vector, resume_vectors)[0]

    return similarity_scores


# ---------------------------
# MAIN ROUTE
# ---------------------------

@app.route("/", methods=["GET", "POST"])

def index():

    results = []

    if request.method == "POST":

        job_desc = request.form.get("job_description","")
        job_desc = preprocess(job_desc)

        files = request.files.getlist("resumes")

        resumes = []
        names = []
        skill_scores = []

        for file in files:

            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            text = extract_text(path)
            text = preprocess(text)

            resumes.append(text)
            names.append(file.filename)

            skill_scores.append(skill_score(text))

        similarity_scores = rank_resumes(job_desc, resumes)

        # ----- FINAL SCORE -----

        final_scores = []

        for i in range(len(names)):

            skill_component = skill_scores[i] / 50   # normalize
            similarity_component = similarity_scores[i]

            final_score = (0.7 * similarity_component) + (0.3 * skill_component)

            final_scores.append(final_score)

        results = sorted(zip(names, final_scores), key=lambda x: x[1], reverse=True)

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)