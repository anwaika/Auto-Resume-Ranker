from flask import Flask, render_template, request, make_response
import os
import uuid
import io
import csv
import json
import urllib.request
import urllib.error
import PyPDF2
import docx
import nltk
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────
# SEMANTIC MODEL
# ─────────────────────────────────────────

print("Loading semantic AI model...")
ai_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# ─────────────────────────────────────────
# SKILL DATABASE
# ─────────────────────────────────────────

SKILLS = {
    "python": 5, "sql": 5, "statistics": 4, "machine learning": 5,
    "data analysis": 4, "data visualization": 3, "data cleaning": 3,
    "tableau": 3, "power bi": 3, "excel": 3,
    "pandas": 4, "numpy": 4, "matplotlib": 3, "seaborn": 3, "scikit-learn": 4,
    "deep learning": 5, "tensorflow": 4, "pytorch": 4, "nlp": 4,
    "postgresql": 4, "mysql": 3, "mongodb": 3,
    "aws": 3, "azure": 3, "gcp": 3,
    "spark": 4, "hadoop": 3, "git": 2, "docker": 3,
    "a/b testing": 4, "hypothesis testing": 4, "time series": 4,
}

SKILL_MAX = sum(SKILLS.values())

# ─────────────────────────────────────────
# SECTION PATTERNS & WEIGHTS
# ─────────────────────────────────────────

SECTION_PATTERNS = {
    "skills": [
        r"skills?", r"technical skills?", r"core competenc",
        r"tools? &? ?technologies?", r"key skills?",
        r"professional skills?", r"competencies", r"technologies",
    ],
    "experience": [
        r"(work\s)?experience", r"employment( history)?",
        r"professional experience", r"work history",
        r"career history", r"positions? held", r"internship",
    ],
    "education": [
        r"education", r"academic background", r"qualifications?",
        r"degrees?", r"certifications?", r"training",
    ],
    "summary": [
        r"summary", r"profile", r"objective", r"about me",
        r"professional summary", r"career objective", r"overview",
    ],
}

SECTION_WEIGHTS = {
    "skills":     0.35,
    "experience": 0.40,
    "education":  0.10,
    "summary":    0.10,
    "other":      0.05,
}

FEATURE_NAMES = ["TF-IDF/SVD", "Semantic", "Skill Match", "Experience"]

# ─────────────────────────────────────────
# TRAINING DATA
# 20 synthetic candidate profiles covering the full spectrum of
# archetypes: keyword stuffers, domain mismatches, fresh graduates,
# over-experienced candidates, and genuine strong matches.
# Labels are human-assigned quality scores in [0, 1].
# ─────────────────────────────────────────

TRAINING_X = np.array([
    # tfidf  semantic  skill   exp
    [0.85,   0.88,     0.90,   0.80],   # 1.  perfect all-rounder, senior
    [0.80,   0.85,     0.85,   0.60],   # 2.  strong match, mid-level
    [0.75,   0.82,     0.80,   0.40],   # 3.  good match, junior-mid
    [0.70,   0.78,     0.75,   0.70],   # 4.  solid, experienced
    [0.65,   0.80,     0.70,   0.50],   # 5.  decent semantic, avg skills
    [0.60,   0.75,     0.65,   0.30],   # 6.  moderate fit, low exp
    [0.55,   0.70,     0.60,   0.60],   # 7.  average across the board
    [0.50,   0.65,     0.55,   0.40],   # 8.  below average overall
    [0.40,   0.60,     0.45,   0.20],   # 9.  weak keyword match
    [0.30,   0.55,     0.35,   0.10],   # 10. poor fit, low skills
    [0.20,   0.45,     0.20,   0.05],   # 11. very poor fit
    [0.10,   0.35,     0.10,   0.00],   # 12. irrelevant resume
    [0.90,   0.50,     0.85,   0.70],   # 13. keyword stuffer
    [0.20,   0.88,     0.30,   0.10],   # 14. good writer, wrong skills
    [0.15,   0.40,     0.90,   0.80],   # 15. wrong domain expert
    [0.75,   0.80,     0.40,   0.90],   # 16. experienced, skills gap
    [0.60,   0.72,     0.80,   0.00],   # 17. great skills, fresh grad
    [0.50,   0.68,     0.70,   0.20],   # 18. decent skills, little exp
    [0.85,   0.83,     0.75,   0.30],   # 19. strong match, early career
    [0.45,   0.60,     0.50,   0.50],   # 20. middling all round
])

TRAINING_Y = np.array([
    0.95, 0.88, 0.80, 0.78, 0.72,
    0.63, 0.58, 0.50, 0.40, 0.30,
    0.20, 0.10, 0.55, 0.45, 0.38,
    0.60, 0.70, 0.62, 0.82, 0.48,
])

# ─────────────────────────────────────────
# PSEUDOINVERSE REGRESSION  (Unit 1 — SVD, Pseudoinverse)
#
# Instead of Ridge.fit() we solve:
#   w = X⁺ y   (Moore-Penrose pseudoinverse)
#   X⁺ = V Σ⁺ Uᵀ  — computed internally by np.linalg.lstsq
#
# This gives the minimum-norm least-squares solution with no
# hyperparameter to tune. Ridge needs alpha; lstsq does not.
# We still scale features so each dimension contributes equally.
# ─────────────────────────────────────────

scaler        = StandardScaler()
X_train_scaled = scaler.fit_transform(TRAINING_X)

# Solve w = X⁺ y
pseudo_weights, residuals, rank_Xt, sv = np.linalg.lstsq(
    X_train_scaled, TRAINING_Y, rcond=None
)

print("\n── Pseudoinverse Regression (X⁺y) ──")
print(f"  Matrix rank : {rank_Xt}  (full rank = {X_train_scaled.shape[1]})")
print(f"  Singular values : {np.round(sv, 4)}")
for name, w in zip(FEATURE_NAMES, pseudo_weights):
    print(f"  {name:<14}: {w:+.4f}")
print("─────────────────────────────────────\n")

# ─────────────────────────────────────────
# FEATURE CORRELATION MATRIX  (Unit 3 — Covariance / Correlation)
#
# Printed at startup on training data so you can verify feature
# independence. Recomputed on live data each ranking call and
# stored in session for display. If tfidf ↔ semantic r > 0.8
# they're largely redundant — good demo talking point.
# ─────────────────────────────────────────

_corr_train = np.corrcoef(TRAINING_X.T)

print("── Training-data feature correlation matrix ──")
header = f"{'':14}" + "".join(f"{n:>15}" for n in FEATURE_NAMES)
print(header)
for i, row in enumerate(_corr_train):
    row_str = f"{FEATURE_NAMES[i]:<14}" + "".join(f"{v:>15.3f}" for v in row)
    print(row_str)
print("──────────────────────────────────────────────\n")


# ─────────────────────────────────────────
# TEXT EXTRACTION
# ─────────────────────────────────────────

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
            print(f"PDF read error: {e}")
        return text

    elif ext.endswith(".docx"):
        try:
            doc = docx.Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            print(f"DOCX read error: {e}")
            return ""

    elif ext.endswith(".txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"TXT read error: {e}")
            return ""

    return ""


# ─────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    return " ".join(nltk.word_tokenize(text))


# ─────────────────────────────────────────
# SECTION PARSER
# ─────────────────────────────────────────

def parse_sections(text):
    lines           = text.split("\n")
    sections        = {k: "" for k in SECTION_PATTERNS}
    sections["other"] = ""
    current_section = "other"

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        matched = None
        for sec_name, patterns in SECTION_PATTERNS.items():
            for pat in patterns:
                if re.fullmatch(pat + r"[\s:.\-]*", stripped.lower()) or (
                    len(stripped) <= 60
                    and re.search(pat, stripped.lower())
                    and not re.search(r'[,;]', stripped)
                ):
                    matched = sec_name
                    break
            if matched:
                break

        if matched:
            current_section = matched
        else:
            sections[current_section] += " " + stripped

    return sections


# ─────────────────────────────────────────
# SKILL UTILITIES
# ─────────────────────────────────────────

def _skill_present(skill, text):
    pat = r"(?<![a-zA-Z0-9])" + re.escape(skill) + r"(?![a-zA-Z0-9])"
    return bool(re.search(pat, text, re.IGNORECASE))


def section_aware_skill_score(sections):
    score      = 0
    skills_txt = sections.get("skills", "").lower()
    rest_txt   = " ".join([
        sections.get("experience", ""),
        sections.get("summary",    ""),
        sections.get("other",      ""),
    ]).lower()

    for skill, weight in SKILLS.items():
        if _skill_present(skill, skills_txt):
            score += weight * 1.5
        elif _skill_present(skill, rest_txt):
            score += weight * 0.8

    return min(score / SKILL_MAX, 1.0)


def extract_skill_gap(job_desc_raw, resume_raw):
    required = [s for s in SKILLS if _skill_present(s, job_desc_raw)]
    missing  = [s for s in required if not _skill_present(s, resume_raw)]
    missing.sort(key=lambda s: SKILLS[s], reverse=True)
    return missing


# ─────────────────────────────────────────
# TF-IDF + SVD SECTION SCORER
# ─────────────────────────────────────────

def section_aware_tfidf_score(job_desc_processed, sections):
    section_scores = {}

    for sec_name in SECTION_WEIGHTS:
        sec_text = sections.get(sec_name, "").strip()
        if not sec_text:
            section_scores[sec_name] = 0.0
            continue

        docs = [job_desc_processed, preprocess(sec_text)]
        try:
            tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            mat   = tfidf.fit_transform(docs)
            n_comp = min(10, mat.shape[0] - 1, mat.shape[1] - 1)
            if n_comp >= 2:
                mat = TruncatedSVD(n_components=n_comp).fit_transform(mat)
            score = cosine_similarity(mat[0].reshape(1, -1), mat[1].reshape(1, -1))[0][0]
        except Exception:
            score = 0.0

        section_scores[sec_name] = float(score)

    return sum(SECTION_WEIGHTS[s] * section_scores[s] for s in SECTION_WEIGHTS), section_scores


# ─────────────────────────────────────────
# EXPERIENCE EXTRACTION
# ─────────────────────────────────────────

def extract_experience_years(text):
    matches = re.findall(r'(\d+)\+?\s*years?', text.lower())
    return min(int(max(matches, key=int)), 15) if matches else 0


# ─────────────────────────────────────────
# AI SUMMARY  (one sentence per candidate via Claude API)
# Falls back gracefully if no API key is set or call fails.
# Set ANTHROPIC_API_KEY in your environment to enable this.
# ─────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

def generate_ai_summary(name, rank, final_score, tfidf, semantic, skill, experience, skill_gap):
    """
    Call Claude to generate a one-sentence recruiter summary for this candidate.
    Returns an empty string if the API key is missing or the call fails.
    """
    if not ANTHROPIC_API_KEY:
        return ""

    gap_str = ", ".join(skill_gap[:3]) if skill_gap else "none"
    prompt  = (
        f"You are a professional recruiter writing a brief candidate summary. "
        f"Candidate: {name} | Rank: #{rank} | Final Score: {final_score:.0f}% | "
        f"TF-IDF/SVD: {tfidf:.0f}% | Semantic: {semantic:.0f}% | "
        f"Skill Match: {skill:.0f}% | Experience: {experience} yrs | "
        f"Top skill gaps: {gap_str}. "
        f"Write exactly ONE sentence (max 25 words) summarising this candidate's fit. "
        f"Be specific and professional. No preamble."
    )

    payload = json.dumps({
        "model":      "claude-sonnet-4-20250514",
        "max_tokens": 80,
        "messages":   [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data    = payload,
        headers = {
            "Content-Type":      "application/json",
            "x-api-key":         ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method = "POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"].strip()
    except Exception as e:
        print(f"AI summary error for {name}: {e}")
        return ""


# ─────────────────────────────────────────
# MAIN RANKING FUNCTION
# ─────────────────────────────────────────

def rank_resumes(job_desc_raw, job_desc_processed, resumes_raw):
    n                 = len(resumes_raw)
    tfidf_scores      = np.zeros(n)
    skill_scores      = np.zeros(n)
    experience_scores = np.zeros(n)
    sections_list     = []

    # Semantic similarity (batched)
    job_emb           = ai_model.encode([job_desc_raw])[0]
    res_embs          = ai_model.encode(resumes_raw)
    semantic_scores   = cosine_similarity([job_emb], res_embs)[0]

    for i, raw in enumerate(resumes_raw):
        sections              = parse_sections(raw)
        sections_list.append(sections)
        tfidf_scores[i], _    = section_aware_tfidf_score(job_desc_processed, sections)
        skill_scores[i]       = section_aware_skill_score(sections)
        experience_scores[i]  = extract_experience_years(raw) / 15

    # Feature matrix  [n × 4]
    X        = np.column_stack([tfidf_scores, semantic_scores, skill_scores, experience_scores])
    X_scaled = scaler.transform(X)

    # ── Score using pseudoinverse weights ──────────────────────────────
    # final = X_scaled @ w   (linear combination with learned coefficients)
    raw_scores   = X_scaled @ pseudo_weights + (TRAINING_Y.mean() - X_train_scaled.mean(axis=0) @ pseudo_weights)
    final_scores = np.clip(raw_scores, 0.0, 1.0)

    # ── Feature gradient per candidate  (Unit 2 — Gradient) ───────────
    # For a linear model f(x) = wᵀx, ∂f/∂xᵢ = wᵢ (scaled back to [0,1])
    # We compute the contribution of each feature to the final score:
    #   contribution_i = w_i * x_scaled_i   (signed, shows direction)
    # Then normalise to percentages so they sum to 100% of the abs total.
    gradients_all = []
    for i in range(n):
        contribs    = pseudo_weights * X_scaled[i]                  # signed contributions
        abs_total   = np.abs(contribs).sum() or 1.0
        pct_contribs = (np.abs(contribs) / abs_total * 100).round(1)
        gradients_all.append({
            name: float(pct)
            for name, pct in zip(FEATURE_NAMES, pct_contribs)
        })

    # ── Feature correlation on live data  (Unit 3 — Covariance) ───────
    if n >= 2:
        corr_matrix = np.corrcoef(X.T)
        corr_pairs  = []
        for i in range(len(FEATURE_NAMES)):
            for j in range(i + 1, len(FEATURE_NAMES)):
                r_val = round(float(corr_matrix[i, j]), 3)
                corr_pairs.append({
                    "a":       FEATURE_NAMES[i],
                    "b":       FEATURE_NAMES[j],
                    "r":       r_val,
                    "r_class": "corr-high" if abs(r_val) > 0.7 else "corr-mid" if abs(r_val) > 0.4 else "corr-low",
                    })
        print("\n── Live feature correlation ──")
        for p in corr_pairs:
            print(f"  {p['a']:14} ↔ {p['b']:14} : r = {p['r']:+.3f}")
        print()
    else:
        corr_pairs = []

    # ── Breakdown per resume ───────────────────────────────────────────
    breakdown = []
    for i in range(n):
        secs = sections_list[i]
        raw  = resumes_raw[i]
        breakdown.append({
            "tfidf_svd":      round(float(tfidf_scores[i])    * 100, 1),
            "semantic":       round(float(semantic_scores[i])  * 100, 1),
            "skill":          round(float(skill_scores[i])     * 100, 1),
            "experience":     int(experience_scores[i] * 15),
            "sections_found": [k for k, v in secs.items() if v.strip() and k != "other"],
            "skill_gap":      extract_skill_gap(job_desc_raw, raw),
            "gradient":       gradients_all[i],
        })

    return final_scores, breakdown, corr_pairs


# ─────────────────────────────────────────
# CSV EXPORT ROUTE
# ─────────────────────────────────────────

@app.route("/export-csv", methods=["POST"])
def export_csv():
    """
    Receives the ranked results as JSON from the frontend form
    and returns them as a downloadable CSV file.
    """
    raw = request.form.get("results_json", "[]")
    try:
        results = json.loads(raw)
    except Exception:
        results = []

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Rank", "Name", "Final Score (%)",
        "TF-IDF/SVD (%)", "Semantic AI (%)", "Skill Match (%)",
        "Experience (yrs)", "Skill Gaps", "AI Summary",
        "Gradient: TF-IDF/SVD (%)", "Gradient: Semantic (%)",
        "Gradient: Skill Match (%)", "Gradient: Experience (%)",
    ])

    for r in results:
        grad = r.get("gradient", {})
        writer.writerow([
            r.get("rank",        ""),
            r.get("name",        ""),
            r.get("final_score", ""),
            r.get("tfidf_svd",   ""),
            r.get("semantic",    ""),
            r.get("skill",       ""),
            r.get("experience",  ""),
            "; ".join(r.get("skill_gap", [])),
            r.get("ai_summary",  ""),
            grad.get("TF-IDF/SVD",   ""),
            grad.get("Semantic",     ""),
            grad.get("Skill Match",  ""),
            grad.get("Experience",   ""),
        ])

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=resume_rankings.csv"
    response.headers["Content-Type"]        = "text/csv"
    return response


# ─────────────────────────────────────────
# MAIN ROUTE
# ─────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    results    = []
    corr_pairs = []

    if request.method == "POST":

        job_desc_raw = request.form.get("job_description", "")
        if not job_desc_raw:
            return render_template("index.html", results=[], corr_pairs=[])

        job_desc_processed = preprocess(job_desc_raw)
        files              = request.files.getlist("resumes")

        resumes_raw = []
        names       = []

        for file in files:
            if file.filename == "":
                continue

            ext       = os.path.splitext(file.filename)[1].lower()
            temp_name = f"{uuid.uuid4().hex}{ext}"
            path      = os.path.join(UPLOAD_FOLDER, temp_name)
            file.save(path)
            raw_text = extract_text(path)
            try:
                os.remove(path)
            except Exception:
                pass

            if raw_text.strip():
                resumes_raw.append(raw_text)
                names.append(file.filename)

        if resumes_raw:
            final_scores, breakdown, corr_pairs = rank_resumes(
                job_desc_raw, job_desc_processed, resumes_raw
            )

            combined = sorted(
                zip(names, final_scores, breakdown),
                key=lambda x: x[1], reverse=True
            )

            results = []
            for i, (name, score, bd) in enumerate(combined):
                ai_summary = generate_ai_summary(
                    name         = name,
                    rank         = i + 1,
                    final_score  = score * 100,
                    tfidf        = bd["tfidf_svd"],
                    semantic     = bd["semantic"],
                    skill        = bd["skill"],
                    experience   = bd["experience"],
                    skill_gap    = bd["skill_gap"],
                )
                results.append({
                    "rank":           i + 1,
                    "name":           name,
                    "final_score":    round(float(score) * 100, 2),
                    "tfidf_svd":      bd["tfidf_svd"],
                    "semantic":       bd["semantic"],
                    "skill":          bd["skill"],
                    "experience":     bd["experience"],
                    "sections_found": bd["sections_found"],
                    "skill_gap":      bd["skill_gap"],
                    "gradient":       bd["gradient"],
                    "ai_summary":     ai_summary,
                })

    return render_template("index.html", results=results, corr_pairs=corr_pairs)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)