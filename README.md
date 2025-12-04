# 🧠 ATS Resume–JD Matching System  
### *RAG + Semantic Search + Skill Matching + Mistral LLM Explanation + Streamlit UI*

This project is an AI-powered **ATS (Applicant Tracking System)** that evaluates how well a candidate's resume matches a job description using:

- Retrieval-Augmented Generation (RAG)
- Sentence Transformer Embeddings (MiniLM-L6-v2)
- Skill extraction engine
- Semantic similarity matching
- Final weighted score
- Mistral 7B LLM ATS-style explanation
- Streamlit-based web application

---

## 🚀 Features

### ✅ Resume Processing (RAG)
- Extract text from PDF (PyMuPDF)
- Clean and normalize text
- Chunk into segments for RAG
- Generate embeddings using SentenceTransformer
- Extract skills using vocabulary match

### ✅ Job Description Processing
- Extract text from PDF
- Clean + normalize
- Extract JD skills automatically
- Generate embedding for the full JD

### ✅ Matching Engine
The system computes:

#### **1️⃣ Skill Match Score**
Percentage of JD skills found in resume.

#### **2️⃣ Semantic Match Score**
Cosine similarity between JD embedding and resume chunk embeddings.

#### **3️⃣ Final Score**
final_score = 0.6 * skill_match + 0.4 * semantic_match

yaml
Copy code

---

## 🤖 LLM ATS Explanation (Mistral 7B)
The system generates:

- Summary of candidate–JD alignment  
- Strengths  
- Missing / weak skills  
- Role suitability paragraph  
- Final recommendation (Good Fit / Borderline Fit / Not Fit)

LLM used: **mistralai/Mistral-7B-Instruct-v0.2** (HuggingFace Inference API)

---

## 🖥️ Streamlit UI
- Upload **Resume PDF** + **JD PDF**
- Shows Match Scores (Skill, Semantic, Final)
- Shows overlapping + missing skills
- Displays full ATS explanation
- Expandable sections showing cleaned text

---

## 🏗️ Architecture Overview

Resume PDF → Extract → Clean → Chunk → Resume Embeddings →
↘
Matching Engine → LLM → Streamlit UI
↗
JD PDF → Extract → Clean → JD Embedding →

yaml
Copy code

Architecture PNG: `ats_architecture.png`  

---

## 📦 Technologies Used

| Component | Technology |
|----------|------------|
| PDF processing | PyMuPDF |
| Embeddings | SentenceTransformer (MiniLM-L6-v2) |
| Similarity | Cosine Similarity (sklearn) |
| LLM | Mistral-7B (HF Inference API) |
| Frontend | Streamlit |
| RAG | Chunking + Embeddings |

---

## 🗂️ Project Structure

📁 ats-matcher
│── app.py # Streamlit application
│── requirements.txt # Dependencies
│── ats_architecture.png # Architecture diagram
│── README.md # Documentation

yaml
Copy code

---

## ⚙️ Installation

### 1️⃣ Clone the repository
git clone https://github.com/yourusername/ats-matcher.git
cd ats-matcher

shell
Copy code

### 2️⃣ Install dependencies
pip install -r requirements.txt

shell
Copy code

### 3️⃣ Run Streamlit
streamlit run app.py

yaml
Copy code

---

## 🔑 HuggingFace API Token
Provide your HF token inside the Streamlit sidebar.

Generate one here:  
https://huggingface.co/settings/tokens  

Token format:
hf_xxxxxxxxxxxxxxxxxxxxxx

yaml
Copy code

---

## 📝 Example Output

Skill Match: 87.5%
Semantic Match: 47.95%
Final Score: 71.68%

Common Skills:
python, java, docker, git, sql, mysql, postgresql, react, node.js, go, data science

ATS Recommendation:
Borderline Fit → Strong skills but missing Redis and EKS.

yaml
Copy code

---

## 🔮 Future Enhancements
- Rank multiple resumes against a single JD  
- Export ATS report as PDF  
- Add chatbot feedback (“Why was I rejected?”)  
- Add ML-based automatic skill extraction (NER)  

---

## 📄 License
This project is for educational and research purposes.

---

## ⭐ Acknowledgements
- HuggingFace Inference API  
- Sentence Transformers  
- Streamlit Framework  
- Mistral AI  

-working link deployed via streamlit:-https://ragbasedprojectforresume.streamlit.app/#score-breakdown
