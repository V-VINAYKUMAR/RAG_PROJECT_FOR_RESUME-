import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient
import io
import pandas as pd

# ===================== PAGE CONFIG & STYLE =====================

st.set_page_config(
    page_title="ATS Resume–JD Matcher",
    layout="wide",
    page_icon="🧠",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-title {
        font-size: 32px !important;
        font-weight: 800 !important;
        padding-bottom: 0.3rem;
    }
    .sub-text {
        color: #6c757d;
        font-size: 14px;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.8rem;
        background: #0e1117;
        border: 1px solid #30363d;
    }
    .section-title {
        font-size: 20px !important;
        font-weight: 700 !important;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===================== MODEL & GLOBALS =====================

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

SKILL_VOCAB = [

    # Programming Languages
    "python", "java", "c", "c++", "c#", "javascript", "typescript",
    "go", "golang", "ruby", "rust", "kotlin", "swift", "php", "matlab",
    "scala", "perl", "r",

    # Machine Learning / Data Science Core
    "machine learning", "deep learning", "data science",
    "data analysis", "statistical analysis", "feature engineering",
    "model training", "model evaluation", "model deployment",
    "hyperparameter tuning", "gradient boosting", "time series",

    # ML Libraries
    "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
    "xgboost", "lightgbm", "catboost",

    # Deep Learning Frameworks
    "tensorflow", "keras", "pytorch", "theano", "mxnet",
    "onnx", "cuda", "cuDNN",

    # NLP
    "nlp", "natural language processing", "transformers",
    "bert", "gpt", "llama", "roberta", "distilbert",
    "text classification", "text generation", "tokenization",
    "named entity recognition", "sentiment analysis",

    # Computer Vision
    "computer vision", "opencv", "image processing",
    "object detection", "cnn", "yolo", "resnet", "vgg", "segmentation",

    # Data Engineering
    "etl", "data pipelines", "apache airflow", "apache spark",
    "hadoop", "kafka", "flink", "data warehouse", "data lake",

    # Databases
    "sql", "mysql", "postgresql", "sqlite", "oracle", "mariadb",
    "mongodb", "cassandra", "redis", "dynamodb", "elasticsearch",

    # Cloud Platforms
    "aws", "azure", "gcp", "google cloud", "amazon web services",
    "cloud functions", "lambda", "s3", "ec2", "gke", "eks", "aks",

    # DevOps / CI/CD
    "docker", "kubernetes", "jenkins", "github actions", "gitlab ci",
    "terraform", "ansible", "prometheus", "grafana",

    # Web Development (Frontend)
    "html", "css", "javascript", "react", "angular", "vue",
    "bootstrap", "material-ui", "tailwind",

    # Web Development (Backend)
    "node.js", "express.js", "django", "flask", "fastapi", "spring boot",
    "laravel", "ruby on rails",

    # Mobile Development
    "android", "kotlin", "swift", "flutter", "react native",

    # Testing
    "unit testing", "pytest", "selenium", "cypress",

    # Tools / Misc
    "git", "github", "gitlab", "jira", "linux", "vim", "vs code",
    "postman", "swagger", "rest api", "grpc",

    # AI Engineering / MLOps
    "mlops", "model monitoring", "model optimization",
    "quantization", "model compression", "api deployment",
    "streamlit", "gradio",

    # Big Data Ecosystem
    "hdfs", "yarn", "zookeeper", "hive", "pig", "presto", "trino"
]


# ===================== HELPERS =====================

def extract_text_from_pdf_file(uploaded_file):
    """Extract text from a Streamlit uploaded PDF file."""
    if uploaded_file is None:
        return ""
    data = uploaded_file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def clean_text(t: str) -> str:
    return " ".join(t.split())

def extract_skills_from_text(text: str):
    text_lower = text.lower()
    found = []
    for skill in SKILL_VOCAB:
        if skill.lower() in text_lower:
            found.append(skill)
    return list(set(found))

def build_resume_rag_from_pdf(uploaded_file):
    raw_text = extract_text_from_pdf_file(uploaded_file)
    clean = clean_text(raw_text)
    chunks = chunk_text(clean)
    if not chunks:
        return None
    embeddings = embedder.encode(chunks)
    return {
        "raw_text": raw_text,
        "clean_text": clean,
        "chunks": chunks,
        "embeddings": embeddings,
    }

def build_jd_from_pdf(uploaded_file):
    raw_text = extract_text_from_pdf_file(uploaded_file)
    clean = clean_text(raw_text)
    skills = extract_skills_from_text(clean)
    embedding = embedder.encode([clean])  # single embedding
    return {
        "jd_raw_text": raw_text,
        "jd_clean_text": clean,
        "jd_skills": skills,
        "jd_embedding": embedding,
    }

def compute_skill_match(resume_skills, jd_skills):
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)
    common = list(resume_set.intersection(jd_set))
    if len(jd_set) == 0:
        score = 0.0
    else:
        score = (len(common) / len(jd_set)) * 100
    return round(score, 2), common

def compute_semantic_match(jd_embedding, resume_embeddings):
    sims = cosine_similarity(jd_embedding, resume_embeddings)[0]
    best = max(sims)
    return round(float(best) * 100, 2)

def compute_final_score(skill, semantic):
    # 60% skill + 40% semantic
    return round(0.6 * skill + 0.4 * semantic, 2)

def match_resume_to_jd(resume_data, jd_data):
    resume_text = resume_data["clean_text"].lower()
    resume_skills = extract_skills_from_text(resume_text)

    skill_score, common = compute_skill_match(resume_skills, jd_data["jd_skills"])
    semantic_score = compute_semantic_match(jd_data["jd_embedding"], resume_data["embeddings"])
    final_score = compute_final_score(skill_score, semantic_score)

    return {
        "resume_skills": resume_skills,
        "jd_skills": jd_data["jd_skills"],
        "common_skills": common,
        "skill_match_%": skill_score,
        "semantic_match_%": semantic_score,
        "final_match_%": final_score,
    }

# ===================== LLM (MISTRAL) =====================

@st.cache_resource
def init_mistral_client(hf_token: str):
    return InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        token=hf_token
    )

def build_prompt(result):
    skill = result["skill_match_%"]
    semantic = result["semantic_match_%"]
    final = result["final_match_%"]

    common = ", ".join(result["common_skills"]) if result["common_skills"] else "None"
    missing_list = list(set(result["jd_skills"]) - set(result["resume_skills"]))
    missing = ", ".join(missing_list) if missing_list else "None"

    return f"""
You are an ATS (Applicant Tracking System). Generate a clear, concise and structured evaluation of how well the candidate's resume matches the job description.

Skill Match: {skill}%
Semantic Match: {semantic}%
Final Score: {final}%

Common Skills: {common}
Missing Skills: {missing}

Write the response in this exact structure:

Overall Summary:
- 2–3 sentences

Strengths:
- bullet 1
- bullet 2
- bullet 3

Missing Skills:
- bullet 1
- bullet 2

Suitability:
- 3–4 sentence paragraph

Recommendation:
- Good Fit / Borderline Fit / Not a Good Fit
"""

def generate_ats_report(client: InferenceClient, result):
    prompt = build_prompt(result)
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3,
    )
    choice = response.choices[0]
    if hasattr(choice, "message"):
        return choice.message["content"]
    elif hasattr(choice, "delta"):
        return choice.delta["content"]
    else:
        return str(response)

def ats_report_to_bytes(report_text: str) -> bytes:
    return report_text.encode("utf-8")

# ===================== UI LAYOUT =====================

st.markdown('<div class="main-title">🧠 ATS Resume–JD Matcher</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">RAG + Semantic Search + Mistral 7B Instruct • Resume vs Job Description Matching</div>',
    unsafe_allow_html=True,
)
st.write("")

with st.sidebar:
    st.header("⚙️ Settings & Inputs")

    st.markdown("### 1️⃣ Hugging Face Token")
    hf_token = st.text_input(
        "Hugging Face API Token (`hf_...`)",
        type="password",
        help="Create a Read token at https://huggingface.co/settings/tokens",
    )

    st.markdown("### 2️⃣ Upload Files")
    resume_file = st.file_uploader("Resume PDF", type=["pdf"], key="resume")
    jd_file = st.file_uploader("Job Description PDF", type=["pdf"], key="jd")

    st.markdown("### 3️⃣ Run Matching")
    run_button = st.button("🚀 Analyze Resume vs JD")

    st.markdown("---")
    st.caption("Tip: Your token is never stored. It's only used during this session.")

# State vars
result = None
resume_data = None
jd_data = None
ats_text = None

if run_button:
    if not hf_token:
        st.error("Please enter your Hugging Face API token in the sidebar.")
    elif resume_file is None or jd_file is None:
        st.error("Please upload both Resume and JD PDFs.")
    else:
        with st.spinner("🔍 Processing resume and job description..."):
            try:
                resume_data = build_resume_rag_from_pdf(resume_file)
                jd_data = build_jd_from_pdf(jd_file)

                if resume_data is None or jd_data is None:
                    st.error("Could not extract text from one of the PDFs.")
                else:
                    result = match_resume_to_jd(resume_data, jd_data)
            except Exception as e:
                st.error(f"Error while processing PDFs: {e}")

        if result is not None:
            # Call Mistral
            with st.spinner("🤖 Generating ATS explanation with Mistral 7B..."):
                try:
                    client = init_mistral_client(hf_token)
                    ats_text = generate_ats_report(client, result)
                except Exception as e:
                    ats_text = f"Error while calling Mistral API: {e}"

# ===================== MAIN TABS =====================

if result is not None:
    tab1, tab2, tab3 = st.tabs(["🏠 Overview", "📌 Match Details", "📄 Raw Text"])

    # ---------- TAB 1: OVERVIEW ----------
    with tab1:
        st.markdown('<div class="section-title">📊 Matching Scores Overview</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Skill Match", f"{result['skill_match_%']} %")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Semantic Match", f"{result['semantic_match_%']} %")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Final Match Score", f"{result['final_match_%']} %")
            st.markdown('</div>', unsafe_allow_html=True)

        # Bar chart
        st.markdown("#### 📈 Score Breakdown")
        df_scores = pd.DataFrame(
            {
                "Score Type": ["Skill Match", "Semantic Match", "Final Score"],
                "Value": [
                    result["skill_match_%"],
                    result["semantic_match_%"],
                    result["final_match_%"],
                ],
            }
        )
        st.bar_chart(df_scores.set_index("Score Type"))

        if ats_text:
            st.markdown("#### 🧾 Quick ATS Summary")
            st.write(ats_text.split("Suitability:")[0].strip())

    # ---------- TAB 2: MATCH DETAILS ----------
    with tab2:
        st.markdown('<div class="section-title">📌 Skill Match Details</div>', unsafe_allow_html=True)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Resume Skills (detected):**")
            if result["resume_skills"]:
                st.write(", ".join(sorted(result["resume_skills"])))
            else:
                st.write("_No skills detected from vocab in resume text._")

        with colB:
            st.markdown("**JD Skills (detected):**")
            if result["jd_skills"]:
                st.write(", ".join(sorted(result["jd_skills"])))
            else:
                st.write("_No skills detected from vocab in JD text._")

        st.markdown("**✅ Common Skills:**")
        if result["common_skills"]:
            st.success(", ".join(sorted(result["common_skills"])))
        else:
            st.warning("No overlapping skills between resume and JD.")

        missing_skills = list(set(result["jd_skills"]) - set(result["resume_skills"]))
        st.markdown("**⚠️ Missing (JD but not in Resume):**")
        if missing_skills:
            st.error(", ".join(sorted(missing_skills)))
        else:
            st.info("Resume seems to cover all JD skills in the vocabulary.")

        st.markdown('<div class="section-title">🧾 Full ATS Explanation</div>', unsafe_allow_html=True)
        if ats_text:
            st.markdown(ats_text)
            # Download button
            report_bytes = ats_report_to_bytes(ats_text)
            st.download_button(
                "⬇️ Download ATS Report (TXT)",
                data=report_bytes,
                file_name="ats_report.txt",
                mime="text/plain",
            )
        else:
            st.write("No ATS explanation available.")

    # ---------- TAB 3: RAW TEXT ----------
    with tab3:
        st.markdown('<div class="section-title">📄 Raw Extracted Text</div>', unsafe_allow_html=True)
        colR, colJ = st.columns(2)

        with colR:
            st.markdown("**Resume (first 1500 characters):**")
            st.write(resume_data["clean_text"][:1500] + "..." if len(resume_data["clean_text"]) > 1500 else resume_data["clean_text"])

        with colJ:
            st.markdown("**Job Description (first 1500 characters):**")
            st.write(jd_data["jd_clean_text"][:1500] + "..." if len(jd_data["jd_clean_text"]) > 1500 else jd_data["jd_clean_text"])

else:
    st.info("⬅️ Start by entering your Hugging Face token and uploading a Resume PDF + JD PDF in the sidebar, then click **'🚀 Analyze Resume vs JD'**.")
