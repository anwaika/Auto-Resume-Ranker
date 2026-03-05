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
# TEXT EXTRACTION
# ---------------------------

def extract_text(file_path):

    if file_path.endswith(".pdf"):
        text = ""
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
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
    text = re.sub(r'[^a-zA-Z ]', '', text)

    tokens = nltk.word_tokenize(text)

    return " ".join(tokens)


# ---------------------------
# RANKING FUNCTION
# ---------------------------

def rank_resumes(job_desc, resumes):

    documents = [job_desc] + resumes

    tfidf = TfidfVectorizer(stop_words="english")

    tfidf_matrix = tfidf.fit_transform(documents)

    # Safe SVD size
    n_components = min(50, tfidf_matrix.shape[1] - 1)

    if n_components > 1:
        svd = TruncatedSVD(n_components=n_components)
        reduced_matrix = svd.fit_transform(tfidf_matrix)
    else:
        reduced_matrix = tfidf_matrix.toarray()

    job_vector = reduced_matrix[0].reshape(1, -1)
    resume_vectors = reduced_matrix[1:]

    similarity = cosine_similarity(job_vector, resume_vectors)[0]

    return similarity


# ---------------------------
# MAIN ROUTE
# ---------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    results = []

    if request.method == "POST":

        job_desc = request.form["job_desc"]
        job_desc = preprocess(job_desc)

        files = request.files.getlist("resumes")

        resumes = []
        names = []

        for file in files:

            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            text = extract_text(path)
            text = preprocess(text)

            resumes.append(text)
            names.append(file.filename)

        scores = rank_resumes(job_desc, resumes)

        results = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)