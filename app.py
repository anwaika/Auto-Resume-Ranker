from flask import Flask, render_template, request, make_response, jsonify
import os, uuid, io, csv, json, re
import urllib.request
import PyPDF2
import docx
import nltk
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

# ── GROQ (optional) ───────────────────────────────────────────────
# set GROQ_API_KEY=gsk_...  (Windows, same terminal as python app.py)
# export GROQ_API_KEY=gsk_... (Mac/Linux)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.1-8b-instant"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
print("Groq:", "ENABLED" if GROQ_API_KEY else "DISABLED (set GROQ_API_KEY to enable)")

def groq_call(messages, max_tokens=200, temperature=0.3):
    if not GROQ_API_KEY:
        return "", "GROQ_API_KEY not set"
    payload = json.dumps({"model":GROQ_MODEL,"messages":messages,
                          "max_tokens":max_tokens,"temperature":temperature}).encode()
    req = urllib.request.Request(GROQ_URL, data=payload, method="POST",
          headers={"Content-Type":"application/json",
                   "Authorization":f"Bearer {GROQ_API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip(), ""
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8","replace")
        return "", f"HTTP {e.code}: {body[:200]}"
    except Exception as e:
        return "", str(e)

# ── SEMANTIC MODEL ────────────────────────────────────────────────
print("Loading semantic model...")
try:
    ai_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    BGE_MODE = True
    print("BGE model loaded.")
except Exception:
    ai_model = SentenceTransformer("all-MiniLM-L6-v2")
    BGE_MODE = False
    print("Fallback: MiniLM loaded.")

def encode_text(texts, is_query=False):
    if BGE_MODE and is_query:
        texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
    return ai_model.encode(texts, normalize_embeddings=True)

# ── spaCy (optional) ──────────────────────────────────────────────
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
    print("spaCy loaded.")
except Exception:
    SPACY_OK = False
    print("spaCy unavailable.")

def extract_entities(text):
    if not SPACY_OK:
        return {"orgs":[], "titles":[], "degrees":[]}
    doc = nlp(text[:50000])
    orgs   = list({e.text for e in doc.ents if e.label_=="ORG"})[:8]
    title_kw = re.compile(
        r"engineer|analyst|developer|manager|designer|scientist|"
        r"consultant|architect|specialist|director|lead|intern", re.I)
    titles = list(dict.fromkeys(
        c.text.strip() for c in doc.noun_chunks
        if title_kw.search(c.text) and len(c.text.split())<=5))[:5]
    deg_kw = re.compile(
        r"bachelor|master|phd|b\.?s\.?|m\.?s\.?|b\.?e\.?|m\.?e\.?|"
        r"b\.?tech|m\.?tech|mba|bca|mca", re.I)
    degrees = [s.text.strip()[:80] for s in doc.sents if deg_kw.search(s.text)][:3]
    return {"orgs":orgs, "titles":titles, "degrees":degrees}

# ── ROLE PREDICTION ───────────────────────────────────────────────
ROLE_LABELS = [
    "Software Engineer / Developer","Data Analyst",
    "Data Scientist / ML Engineer","Product Manager",
    "UX / UI Designer","DevOps / Cloud Engineer",
    "Marketing Specialist","Finance / Accounting",
    "HR / People Operations","Sales / Business Development",
    "Content Writer / Journalist","Healthcare / Medical",
    "Legal","Operations / Supply Chain","Research Scientist",
]
print("Encoding role labels...")
_role_embs = encode_text(ROLE_LABELS)
print("Role labels ready.")

def predict_role(text):
    emb  = encode_text([text[:3000]])[0]
    sims = cosine_similarity([emb], _role_embs)[0]
    idx  = int(np.argmax(sims))
    return ROLE_LABELS[idx], round(float(sims[idx])*100, 1)

# ── SKILL DATABASE ────────────────────────────────────────────────
SKILLS = {
    "python":5,"java":5,"javascript":5,"typescript":4,"c++":5,
    "c#":4,"go":4,"rust":4,"ruby":3,"swift":4,"kotlin":4,
    "php":3,"scala":4,"r":3,"matlab":3,"bash":3,"shell scripting":3,
    "react":4,"angular":4,"vue":3,"node.js":4,"django":3,
    "flask":3,"spring boot":4,"fastapi":3,"graphql":3,"rest api":4,
    "microservices":4,"system design":5,"object oriented programming":4,
    "data structures":4,"algorithms":4,
    "sql":5,"statistics":4,"machine learning":5,"deep learning":5,
    "data analysis":4,"data visualization":3,"data cleaning":3,
    "pandas":4,"numpy":4,"matplotlib":3,"seaborn":3,"scikit-learn":4,
    "tensorflow":4,"pytorch":4,"nlp":4,"computer vision":4,
    "a/b testing":4,"hypothesis testing":4,"time series":4,
    "tableau":3,"power bi":3,"looker":3,"excel":3,
    "large language models":4,"generative ai":4,"prompt engineering":3,
    "feature engineering":4,"model deployment":4,"mlops":4,
    "postgresql":4,"mysql":3,"mongodb":3,"redis":3,
    "snowflake":4,"bigquery":4,"elasticsearch":3,"cassandra":3,
    "oracle":3,"sqlite":2,"dynamodb":3,
    "aws":4,"azure":4,"gcp":4,"git":3,"docker":4,
    "kubernetes":4,"terraform":4,"ci/cd":4,"jenkins":3,
    "github actions":3,"ansible":3,"linux":3,"networking":3,
    "spark":4,"hadoop":3,"kafka":4,"airflow":3,
    "devops":4,"sre":4,"cloud architecture":4,
    "product management":5,"product roadmap":4,"user research":4,
    "wireframing":3,"prototyping":3,"figma":4,"sketch":3,
    "adobe xd":3,"ux design":4,"ui design":4,"usability testing":4,
    "design thinking":3,"agile":4,"scrum":4,"kanban":3,"jira":3,
    "digital marketing":5,"seo":4,"sem":4,"google ads":4,
    "content marketing":4,"email marketing":3,"social media":3,
    "hubspot":3,"salesforce":4,"google analytics":4,"copywriting":4,
    "brand strategy":4,"market research":4,"campaign management":4,
    "financial modeling":5,"accounting":4,"financial analysis":5,
    "budgeting":4,"forecasting":4,"valuation":4,
    "sap":4,"erp":3,"audit":4,"gaap":4,"ifrs":4,
    "investment banking":4,"risk management":4,"compliance":4,
    "portfolio management":4,"equity research":4,
    "sales":4,"business development":4,"account management":4,
    "lead generation":4,"negotiation":4,"pipeline management":4,
    "b2b sales":4,"customer success":4,"client relations":4,
    "operations management":5,"supply chain":5,"logistics":4,
    "procurement":4,"inventory management":4,"lean":4,"six sigma":4,
    "process improvement":4,"project management":5,
    "recruiting":5,"talent acquisition":5,"onboarding":3,
    "performance management":4,"employee relations":4,
    "compensation":4,"hr analytics":4,"workforce planning":4,
    "contract drafting":5,"legal research":5,"litigation":4,
    "corporate law":4,"intellectual property":4,
    "regulatory":4,"due diligence":4,
    "clinical research":5,"patient care":5,
    "electronic health records":4,"ehr":4,"fhir":3,
    "healthcare analytics":4,"hipaa":4,"clinical trials":5,
    "nursing":4,"diagnosis":4,"radiology":4,
    "curriculum development":5,"instructional design":5,"teaching":4,
    "e-learning":3,"training delivery":4,"mentoring":3,"coaching":3,
    "technical writing":5,"content writing":4,"editing":4,
    "journalism":4,"grant writing":4,"documentation":3,
    "research methodology":5,"qualitative research":4,
    "quantitative research":4,"experimental design":4,
    "stata":3,"spss":3,
    "communication":3,"leadership":3,"teamwork":2,
    "problem solving":3,"critical thinking":3,
    "stakeholder management":4,"presentation":3,"strategic planning":4,
}
SKILL_MAX = sum(SKILLS.values())

# ── SECTION PATTERNS ──────────────────────────────────────────────
SECTION_PATTERNS = {
    "skills":["skills?","technical skills?","core competenc",
              "tools? &? ?technologies?","key skills?",
              "professional skills?","competencies","technologies"],
    "experience":["(work\s)?experience","employment( history)?",
                  "professional experience","work history",
                  "career history","positions? held","internship"],
    "education":["education","academic background","qualifications?",
                 "degrees?","certifications?","training"],
    "summary":["summary","profile","objective","about me",
               "professional summary","career objective","overview"],
    "projects":["projects?","personal projects?","academic projects?",
                "open source","portfolio"],
}
SECTION_WEIGHTS = {
    "skills":0.30,"experience":0.35,"education":0.10,
    "summary":0.10,"projects":0.10,"other":0.05,
}
FEATURE_NAMES = ["TF-IDF/SVD","Semantic","Skill Match","Experience"]

# ── PSEUDOINVERSE REGRESSION ──────────────────────────────────────
TRAINING_X = np.array([
    [0.85,0.88,0.90,0.80],[0.80,0.85,0.85,0.60],
    [0.75,0.82,0.80,0.40],[0.70,0.78,0.75,0.70],
    [0.65,0.80,0.70,0.50],[0.60,0.75,0.65,0.30],
    [0.55,0.70,0.60,0.60],[0.50,0.65,0.55,0.40],
    [0.40,0.60,0.45,0.20],[0.30,0.55,0.35,0.10],
    [0.20,0.45,0.20,0.05],[0.10,0.35,0.10,0.00],
    [0.90,0.50,0.85,0.70],[0.20,0.88,0.30,0.10],
    [0.15,0.40,0.90,0.80],[0.75,0.80,0.40,0.90],
    [0.60,0.72,0.80,0.00],[0.50,0.68,0.70,0.20],
    [0.85,0.83,0.75,0.30],[0.45,0.60,0.50,0.50],
])
TRAINING_Y = np.array([
    0.95,0.88,0.80,0.78,0.72,0.63,0.58,0.50,
    0.40,0.30,0.20,0.10,0.55,0.45,0.38,0.60,
    0.70,0.62,0.82,0.48,
])
scaler         = StandardScaler()
X_train_sc     = scaler.fit_transform(TRAINING_X)
pseudo_w, _, _, _ = np.linalg.lstsq(X_train_sc, TRAINING_Y, rcond=None)
print("\n── Pseudoinverse weights ──")
for n,w in zip(FEATURE_NAMES, pseudo_w): print(f"  {n:<14}: {w:+.4f}")
print()

# ── GROQ: JD ANALYSIS ─────────────────────────────────────────────
def _compress_jd(jd, max_chars=600):
    kw = re.compile(
        r"require|must|proficien|experienc|skill|knowledge|famili|"
        r"responsibil|qualif|preferred|tool|technolog|degree|certif|year", re.I)
    lines = [l.strip() for l in jd.split("\n") if l.strip() and kw.search(l)]
    return (" | ".join(lines) if lines else jd)[:max_chars]

def analyse_jd(jd):
    empty = {"role_type":"","experience_level":"","must_have":[],"nice_to_have":[],"summary":""}
    if not GROQ_API_KEY: return empty
    prompt = (
        'Extract job info as JSON only, no markdown.\n'
        '{"role_type":"<2-3 word role>","experience_level":"entry|mid|senior",'
        '"must_have":["s1","s2"],"nice_to_have":["s1"],"summary":"<15 words>"}\n'
        "must_have<=6, nice_to_have<=4, all lowercase.\nJD: " + _compress_jd(jd)
    )
    reply, err = groq_call([{"role":"user","content":prompt}], max_tokens=200, temperature=0.1)
    if err: print(f"JD analysis: {err}"); return empty
    try:
        clean  = re.sub(r"```json|```","",reply).strip()
        m      = re.search(r"\{.*\}", clean, re.DOTALL)
        result = json.loads(m.group() if m else clean)
        for k in empty:
            if k not in result: result[k] = [] if "have" in k else ""
        return result
    except Exception as e:
        print(f"JD parse error: {e}"); return empty

# ── GROQ: BATCH SUMMARIES ─────────────────────────────────────────
def generate_summaries(candidates, jd_analysis):
    if not GROQ_API_KEY: return {}
    role  = jd_analysis.get("role_type","this role")
    level = jd_analysis.get("experience_level","")
    lines = []
    for c in candidates:
        gap = ",".join(c["skill_gap"][:2]) if c["skill_gap"] else "none"
        lines.append(f"{c['rank']}|{c['name']}|{c['final_score']:.0f}%|"
                     f"sem:{c['semantic']:.0f}%|sk:{c['skill']:.0f}%|"
                     f"exp:{c['experience']}yr|edu:{c['education']}|gap:{gap}")
    prompt = (
        f"Role:{role}({level}). ONE sentence per candidate (max 15 words). "
        f"Return ONLY JSON {{name:summary}}. No markdown.\n" + "\n".join(lines)
    )
    reply, err = groq_call([{"role":"user","content":prompt}],
                           max_tokens=max(80,len(candidates)*25), temperature=0.4)
    if err: print(f"Summaries: {err}"); return {}
    try:
        clean  = re.sub(r"```json|```","",reply).strip()
        m      = re.search(r"\{.*\}", clean, re.DOTALL)
        result = json.loads(m.group() if m else clean)
        return {str(k):str(v) for k,v in result.items()}
    except Exception as e:
        print(f"Summary parse: {e}"); return {}

# ── TEXT EXTRACTION ───────────────────────────────────────────────
def extract_text(path):
    ext = path.lower()
    if ext.endswith(".pdf"):
        text = ""
        try:
            with open(path,"rb") as f:
                for page in PyPDF2.PdfReader(f).pages:
                    t = page.extract_text()
                    if t: text += t
        except Exception as e: print(f"PDF error: {e}")
        return text
    if ext.endswith(".docx"):
        try: return "\n".join(p.text for p in docx.Document(path).paragraphs)
        except Exception as e: print(f"DOCX error: {e}"); return ""
    if ext.endswith(".txt"):
        try:
            with open(path,"r",encoding="utf-8") as f: return f.read()
        except Exception as e: print(f"TXT error: {e}"); return ""
    return ""

def extract_jd_file(file):
    if not file or not file.filename: return ""
    ext  = os.path.splitext(file.filename)[1].lower()
    tmp  = os.path.join(UPLOAD_FOLDER, f"jd_{uuid.uuid4().hex}{ext}")
    file.save(tmp)
    text = extract_text(tmp)
    try: os.remove(tmp)
    except: pass
    return text.strip()

# ── PREPROCESSING ─────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]',' ',text)
    return " ".join(nltk.word_tokenize(text))

# ── SECTION PARSER ────────────────────────────────────────────────
def parse_sections(text):
    secs    = {k:"" for k in SECTION_PATTERNS}
    secs["other"] = ""
    current = "other"
    for line in text.split("\n"):
        s = line.strip()
        if not s: continue
        matched = None
        for sname, pats in SECTION_PATTERNS.items():
            for pat in pats:
                if re.fullmatch(pat+r"[\s:.\-]*", s.lower()) or (
                    len(s)<=60 and re.search(pat,s.lower())
                    and not re.search(r'[,;]',s)):
                    matched = sname; break
            if matched: break
        current = matched if matched else current
        if not matched: secs[current] += " "+s
    return secs

# ── SKILL UTILITIES ───────────────────────────────────────────────
def _skill_in(skill, text):
    return bool(re.search(
        r"(?<![a-zA-Z0-9])"+re.escape(skill)+r"(?![a-zA-Z0-9])",
        text, re.IGNORECASE))

def skill_score(secs, jd_skills, must_have=None):
    if not jd_skills: return 0.0
    max_s = sum(SKILLS.get(s,3) for s in jd_skills) or 1
    skills_txt = secs.get("skills","").lower()
    rest_txt   = " ".join([secs.get("experience",""), secs.get("projects",""),
                            secs.get("summary",""), secs.get("other","")]).lower()
    boost = set(s.lower() for s in (must_have or []))
    score = 0
    for s in jd_skills:
        w = SKILLS.get(s,3)
        m = 2.0 if s in boost else 1.0
        if   _skill_in(s, skills_txt): score += w*1.5*m
        elif _skill_in(s, rest_txt):   score += w*0.8*m
    return min(score/max_s, 1.0)

def skill_gap(jd_skills, resume_raw, must_have=None):
    missing  = [s for s in jd_skills if not _skill_in(s, resume_raw)]
    must_set = set(s.lower() for s in (must_have or []))
    missing.sort(key=lambda s: (s not in must_set, -SKILLS.get(s,1)))
    return missing

# ── TF-IDF + SVD ─────────────────────────────────────────────────
def tfidf_score(jd_proc, secs):
    sec_s = {}
    for sname in SECTION_WEIGHTS:
        txt = secs.get(sname,"").strip()
        if not txt: sec_s[sname]=0.0; continue
        docs = [jd_proc, preprocess(txt)]
        try:
            mat = TfidfVectorizer(stop_words="english",ngram_range=(1,2)).fit_transform(docs)
            nc  = min(10, mat.shape[0]-1, mat.shape[1]-1)
            if nc>=2: mat = TruncatedSVD(n_components=nc).fit_transform(mat)
            sec_s[sname] = float(cosine_similarity(mat[0].reshape(1,-1),mat[1].reshape(1,-1))[0][0])
        except: sec_s[sname]=0.0
    return sum(SECTION_WEIGHTS[s]*sec_s[s] for s in SECTION_WEIGHTS), sec_s

# ── EXPERIENCE EXTRACTION ─────────────────────────────────────────
def extract_exp_years(text):
    m = re.findall(r'(\d+)\+?\s*years?', text.lower())
    if m: return min(int(max(m, key=int)), 20)
    years = sorted(set(int(y) for y in re.findall(r'\b(20\d{2}|19\d{2})\b', text)))
    if len(years)>=2:
        span = years[-1]-years[0]
        if 0<span<=25: return min(span,20)
    return 0

# ── EDUCATION EXTRACTION ──────────────────────────────────────────
def extract_education(text):
    """
    Extract the highest degree and institution from resume text.
    Returns a short string like 'M.Tech — IIT Delhi' or 'B.Sc Computer Science'.
    """
    deg_patterns = [
        (r'\bph\.?d\b',                        "PhD"),
        (r'\bm\.?tech\b|\bmaster of technology','M.Tech'),
        (r'\bm\.?e\b|\bmaster of engineering',  'M.E'),
        (r'\bmba\b',                             'MBA'),
        (r'\bm\.?sc\b|\bmaster of science',      'M.Sc'),
        (r'\bm\.?s\b|\bmaster of',               'M.S'),
        (r'\bb\.?tech\b|\bbachelor of technology','B.Tech'),
        (r'\bb\.?e\b|\bbachelor of engineering',  'B.E'),
        (r'\bbca\b',                              'BCA'),
        (r'\bmca\b',                              'MCA'),
        (r'\bb\.?sc\b|\bbachelor of science',     'B.Sc'),
        (r'\bb\.?a\b|\bbachelor of arts',         'B.A'),
        (r'\bbachelor',                           'Bachelor\'s'),
        (r'\bmaster',                             'Master\'s'),
        (r'\bassociate',                          'Associate\'s'),
    ]
    text_low = text.lower()
    degree = ""
    for pat, label in deg_patterns:
        if re.search(pat, text_low, re.I):
            degree = label
            break

    # Try to find institution name near the degree mention
    inst = ""
    if SPACY_OK:
        doc  = nlp(text[:20000])
        orgs = [e.text for e in doc.ents if e.label_=="ORG"]
        edu_kw = re.compile(
            r"university|college|institute|school|iit|nit|bits|"
            r"amrita|vit|manipal|anna|caltech|mit|stanford|oxford", re.I)
        for org in orgs:
            if edu_kw.search(org):
                inst = org[:40]
                break

    if degree and inst: return f"{degree} — {inst}"
    if degree:          return degree
    return ""

# ── NORMALISE ─────────────────────────────────────────────────────
def _norm(arr):
    lo,hi = arr.min(), arr.max()
    if hi-lo < 1e-9: return np.ones_like(arr)*0.5
    return (arr-lo)/(hi-lo)

# ── SCORE CALIBRATION ─────────────────────────────────────────────
def calibrate(blended):
    """
    Map raw blended scores (typically 0.20–0.65) to human-readable range.
    Good candidate  → 75-90%
    Average         → 45-65%
    Poor            → 18-35%

    Uses batch-relative rescaling then sigmoid with midpoint at 0.45
    (tuned to BGE similarity distribution) rather than 0.50.
    """
    lo, hi = blended.min(), blended.max()
    rel = (blended-lo)/(hi-lo) if (hi-lo)>0.02 else np.full_like(blended, 0.5)
    return np.clip(0.15 + 0.80/(1.0+np.exp(-6.0*(rel-0.45))), 0.0, 1.0)

# ── MAIN RANKING ──────────────────────────────────────────────────
def rank_resumes(jd_raw, jd_proc, resumes_raw, jd_analysis):
    n         = len(resumes_raw)
    tf_raw    = np.zeros(n)
    sk_raw    = np.zeros(n)
    exp_raw   = np.zeros(n)
    secs_list = []
    must_have = jd_analysis.get("must_have",[])
    jd_skills = [s for s in SKILLS if _skill_in(s, jd_raw)]

    jd_emb   = encode_text([jd_raw], is_query=True)[0]
    res_embs = encode_text(resumes_raw)
    sem_raw  = cosine_similarity([jd_emb], res_embs)[0]

    for i, raw in enumerate(resumes_raw):
        secs = parse_sections(raw)
        secs_list.append(secs)
        tf_raw[i],_ = tfidf_score(jd_proc, secs)
        sk_raw[i]   = skill_score(secs, jd_skills, must_have)
        exp_raw[i]  = extract_exp_years(raw)/20.0

    tf_n  = _norm(tf_raw)
    sem_n = _norm(sem_raw)
    sk_n  = _norm(sk_raw)
    exp_n = _norm(exp_raw)

    X      = np.column_stack([tf_n, sem_n, sk_n, exp_n])
    Xs     = scaler.transform(X)
    bias   = TRAINING_Y.mean() - X_train_sc.mean(axis=0)@pseudo_w
    pred   = Xs@pseudo_w + bias
    w_sum  = 0.40*sem_n + 0.30*sk_n + 0.20*tf_n + 0.10*exp_n
    blended = np.clip(0.55*pred + 0.45*w_sum, 0.0, 1.0)
    final  = calibrate(blended)

    grads = []
    for i in range(n):
        c = np.abs(pseudo_w * Xs[i])
        t = c.sum() or 1.0
        grads.append({nm:float(round(v/t*100,1)) for nm,v in zip(FEATURE_NAMES,c)})

    corr_pairs = []
    if n>=2:
        cm = np.corrcoef(X.T)
        for i in range(len(FEATURE_NAMES)):
            for j in range(i+1, len(FEATURE_NAMES)):
                rv = round(float(cm[i,j]),3)
                corr_pairs.append({
                    "a":FEATURE_NAMES[i],"b":FEATURE_NAMES[j],"r":rv,
                    "r_class":"corr-high" if abs(rv)>0.7 else
                              "corr-mid"  if abs(rv)>0.4 else "corr-low",
                })

    breakdown = []
    for i in range(n):
        secs = secs_list[i]
        raw  = resumes_raw[i]
        role, conf = predict_role(raw)
        ents       = extract_entities(raw)
        edu        = extract_education(raw)
        breakdown.append({
            "tfidf_svd":  round(float(tf_raw[i])*100,1),
            "semantic":   round(float(sem_raw[i])*100,1),
            "skill":      round(float(sk_raw[i])*100,1),
            "experience": int(exp_raw[i]*20),
            "education":  edu,
            "sections_found":[k for k,v in secs.items() if v.strip() and k!="other"],
            "skill_gap":  skill_gap(jd_skills, raw, must_have),
            "gradient":   grads[i],
            "predicted_role": role,
            "role_confidence": conf,
            "entities":   ents,
        })

    return final, breakdown, corr_pairs

# ── CSV EXPORT ────────────────────────────────────────────────────
@app.route("/export-csv", methods=["POST"])
def export_csv():
    try:    results = json.loads(request.form.get("results_json","[]"))
    except: results = []
    out = io.StringIO()
    w   = csv.writer(out)
    w.writerow(["Rank","Name","Final Score (%)","TF-IDF/SVD (%)","Semantic AI (%)",
                "Skill Match (%)","Experience (yrs)","Education",
                "Predicted Role","Role Confidence (%)",
                "Skill Gaps","AI Summary","Companies","Job Titles",
                "Grad:TF-IDF","Grad:Semantic","Grad:Skill","Grad:Experience"])
    for r in results:
        g   = r.get("gradient",{})
        ent = r.get("entities",{})
        w.writerow([
            r.get("rank",""), r.get("name",""), r.get("final_score",""),
            r.get("tfidf_svd",""), r.get("semantic",""), r.get("skill",""),
            r.get("experience",""), r.get("education",""),
            r.get("predicted_role",""), r.get("role_confidence",""),
            "; ".join(r.get("skill_gap",[])), r.get("ai_summary",""),
            "; ".join(ent.get("orgs",[])), "; ".join(ent.get("titles",[])),
            g.get("TF-IDF/SVD",""), g.get("Semantic",""),
            g.get("Skill Match",""), g.get("Experience",""),
        ])
    resp = make_response(out.getvalue())
    resp.headers["Content-Disposition"] = "attachment; filename=resume_rankings.csv"
    resp.headers["Content-Type"]        = "text/csv"
    return resp

# ── DEBUG ROUTE ───────────────────────────────────────────────────
@app.route("/groq-debug")
def groq_debug():
    if not GROQ_API_KEY:
        return jsonify({"status":"NO KEY","detail":"Set GROQ_API_KEY env variable."})
    ans, err = groq_call([{"role":"user","content":"Say: Groq is working."}], max_tokens=20)
    if err: return jsonify({"status":"ERROR","detail":err})
    return jsonify({"status":"OK","response":ans})

# ── MAIN ROUTE ────────────────────────────────────────────────────
@app.route("/", methods=["GET","POST"])
def index():
    results=[]; corr_pairs=[]; jd_analysis={}; error=None

    if request.method=="POST":
        jd_raw  = request.form.get("job_description","").strip()
        jd_file = request.files.get("jd_file")
        if jd_file and jd_file.filename:
            ft = extract_jd_file(jd_file)
            if ft: jd_raw = (jd_raw+"\n"+ft).strip() if jd_raw else ft

        if not jd_raw:
            error = "Please provide a job description — paste text or upload a file."
        else:
            jd_analysis = analyse_jd(jd_raw)
            jd_proc     = preprocess(jd_raw)
            files       = request.files.getlist("resumes")
            resumes_raw=[]; names=[]

            for f in files:
                if not f.filename: continue
                ext  = os.path.splitext(f.filename)[1].lower()
                path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}{ext}")
                f.save(path)
                raw = extract_text(path)
                try: os.remove(path)
                except: pass
                if raw.strip(): resumes_raw.append(raw); names.append(f.filename)

            if not resumes_raw:
                error = "No readable resume content found."
            else:
                final_scores, breakdown, corr_pairs = rank_resumes(
                    jd_raw, jd_proc, resumes_raw, jd_analysis)
                combined = sorted(zip(names,final_scores,breakdown),
                                  key=lambda x:x[1], reverse=True)
                results = []
                for i,(name,score,bd) in enumerate(combined):
                    results.append({
                        "rank":i+1,"name":name,
                        "final_score":round(float(score)*100,2),
                        "tfidf_svd":bd["tfidf_svd"],"semantic":bd["semantic"],
                        "skill":bd["skill"],"experience":bd["experience"],
                        "education":bd["education"],
                        "sections_found":bd["sections_found"],
                        "skill_gap":bd["skill_gap"],"gradient":bd["gradient"],
                        "predicted_role":bd["predicted_role"],
                        "role_confidence":bd["role_confidence"],
                        "entities":bd["entities"],"ai_summary":"",
                    })
                print("Generating summaries...")
                summaries = generate_summaries(results, jd_analysis)
                for r in results:
                    r["ai_summary"] = summaries.get(r["name"],"")

    return render_template("index.html",
                           results=results, corr_pairs=corr_pairs,
                           jd_analysis=jd_analysis,
                           groq_enabled=bool(GROQ_API_KEY),
                           error=error)

if __name__=="__main__":
    app.run(debug=True, use_reloader=False)