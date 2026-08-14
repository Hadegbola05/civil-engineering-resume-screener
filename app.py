import gradio as gr
import joblib
import pandas as pd
import numpy as np
import re
from datetime import datetime
from scipy.sparse import hstack, csr_matrix
import fitz  # PyMuPDF
import docx

# ============================================================
# LOAD SAVED MODEL, VECTORIZER, SCALER
# ============================================================
FINAL_MODEL = joblib.load("resume_classifier_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")
scaler = joblib.load("years_experience_scaler.joblib")

label_order = ["Very Poor", "Poor", "Average", "Good", "Excellent"]
label_order_map = {i: label for i, label in enumerate(label_order)}

ENGINEERING_SKILLS = [
    "autocad", "autodesk civil 3d", "revit", "etabs", "staad.pro", "staad pro",
    "safe", "sap2000", "tekla structures", "bentley microstation",
    "bentley openroads designer", "autodesk navisworks", "arcgis",
    "hec-ras", "hec ras", "epanet", "plaxis", "prokon", "primavera p6",
    "primavera", "microsoft project", "microsoft excel", "power bi",
    "python", "matlab", "sql", "quantity surveying", "cost estimation",
    "bill of quantities", "boq", "structural analysis", "structural design",
    "construction project planning", "project scheduling", "site supervision",
    "quality control", "surveying", "setting out", "technical report writing",
    "civil", "mechanical", "electrical", "structural", "design", "construction",
    "project management", "cad", "manufacturing", "hvac", "plc", "blueprint",
    "engineering", "safety", "maintenance", "testing", "specifications",
]

# ============================================================
# HELPER FUNCTIONS (same logic as training notebook)
# ============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_matched_skills(text, skills_list=ENGINEERING_SKILLS):
    text = text.lower()
    return [skill for skill in skills_list if skill in text]

def score_certificate(cert):
    scores = {"ND": 1, "HND": 2, "B.Sc": 2.5, "B.Tech": 3, "M.Sc": 4}
    return scores.get(cert, 2)

def score_certification(cert):
    cert = str(cert)
    if "COREN" in cert:
        return 4
    elif "FNSE" in cert or "FNICE" in cert:
        return 3
    elif "MNSE" in cert or "MNICE" in cert:
        return 2
    else:
        return 0

def score_level(level):
    scores = {"Junior": 1, "Mid-Level": 2, "Senior": 3, "Executive": 4}
    return scores.get(level, 1)

def score_experience(years):
    if years <= 3: return 1
    elif years <= 8: return 2
    elif years <= 15: return 3
    else: return 4

def extract_years_experience(text):
    text_lower = text.lower()

    matches = re.findall(r'(\d{1,2})\s*\+?\s*years?', text_lower)
    if matches:
        return max(int(m) for m in matches)

    if "d.o.b" in text_lower or "date of birth" in text_lower:
        text = re.sub(
            r'(d\.o\.b|date of birth)[:\s]*\d{1,2}\w{0,2}\s*\w+,?\s*(19|20)\d{2}',
            '', text, flags=re.IGNORECASE
        )

    work_section_match = re.search(
        r'(work(ing)? experience|employment history)(.*)', text,
        re.IGNORECASE | re.DOTALL
    )
    search_text = work_section_match.group(3) if work_section_match else text

    full_years = [int(y) for y in re.findall(r'\b((?:19|20)\d{2})\b', search_text)]
    if full_years:
        earliest_year = min(full_years)
        current_year = datetime.now().year
        estimated_years = current_year - earliest_year
        if 0 < estimated_years <= 40:
            return estimated_years

    return None

def extract_text_from_file(filepath):
    if filepath.lower().endswith(".pdf"):
        text = ""
        doc = fitz.open(filepath)
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    elif filepath.lower().endswith(".docx"):
        doc = docx.Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file type. Please upload a .pdf or .docx file.")

# ============================================================
# CORE PREDICTION LOGIC
# ============================================================

def predict_fit_hybrid(resume_text, skill_weight=0.3):
    cleaned = clean_text(resume_text)
    notes = []

    years_experience = extract_years_experience(resume_text)
    if years_experience is None:
        years_experience = 12.0  # dataset median fallback
        notes.append("Years of experience not detected — used dataset median (12) as fallback.")

    if re.search(r'\bhnd\b|higher national diploma', cleaned):
        certificate = "HND"
    elif re.search(r'\bnd\b|national diploma', cleaned) and "higher" not in cleaned:
        certificate = "ND"
    elif re.search(r'\bm\s?sc\b|master', cleaned):
        certificate = "M.Sc"
    elif re.search(r'\bb\s?tech\b', cleaned):
        certificate = "B.Tech"
    elif re.search(r'\bb\s?sc\b|bachelor', cleaned):
        certificate = "B.Sc"
    else:
        certificate = "ND"
        notes.append("Certificate not detected — defaulted to 'ND'.")

    if "coren" in cleaned:
        professional_certification = "COREN"
    elif "fnse" in cleaned or "fnice" in cleaned:
        professional_certification = "FNSE"
    elif "mnse" in cleaned or "mnice" in cleaned:
        professional_certification = "MNSE"
    else:
        professional_certification = "None"

    if "executive" in cleaned or "director" in cleaned:
        current_level = "Executive"
    elif "senior" in cleaned:
        current_level = "Senior"
    elif "mid" in cleaned:
        current_level = "Mid-Level"
    else:
        current_level = "Junior"
        notes.append("Current level not detected — defaulted to 'Junior'.")

    exp_score = score_experience(years_experience)
    cert_score = score_certificate(certificate)
    prof_cert_score = score_certification(professional_certification)
    level_score = score_level(current_level)

    text_features = vectorizer.transform([cleaned])
    numeric_input = pd.DataFrame(
        [[exp_score, cert_score, prof_cert_score, level_score]],
        columns=['exp_score', 'cert_score', 'prof_cert_score', 'level_score']
    )
    numeric_scaled = scaler.transform(numeric_input)
    combined_features = hstack([text_features, csr_matrix(numeric_scaled)])

    predicted_class_index = FINAL_MODEL.predict(combined_features)[0]
    predicted_label = label_order_map[predicted_class_index]
    probabilities = FINAL_MODEL.predict_proba(combined_features)[0]

    matched = extract_matched_skills(cleaned, ENGINEERING_SKILLS)
    skill_score = len(matched) / len(ENGINEERING_SKILLS)

    excellent_idx = label_order.index("Excellent")
    good_idx = label_order.index("Good")
    model_fit_prob = probabilities[excellent_idx] + probabilities[good_idx]

    hybrid_score = (model_fit_prob * (1 - skill_weight)) + (skill_score * skill_weight)

    return {
        "predicted_grade": predicted_label,
        "years_experience_used": years_experience,
        "certificate_used": certificate,
        "professional_certification_used": professional_certification,
        "current_level_used": current_level,
        "model_fit_probability": round(model_fit_prob * 100, 1),
        "skill_match_score": round(skill_score * 100, 1),
        "hybrid_fit_score": round(hybrid_score * 100, 1),
        "matched_skills": matched if matched else ["None detected"],
        "notes": notes,
    }

# ============================================================
# GRADIO INTERFACE FUNCTION
# ============================================================

def screen_resume(file):
    if file is None:
        return "⚠️ Please upload a PDF or DOCX resume file.", "", ""

    try:
        resume_text = extract_text_from_file(file.name)
    except Exception as e:
        return f"❌ Error reading file: {e}", "", ""

    if not resume_text.strip():
        return "⚠️ Could not extract any text from this file. Try a different PDF/DOCX.", "", ""

    result = predict_fit_hybrid(resume_text)

    grade_display = f"## Predicted Grade: **{result['predicted_grade']}**"

    details = f"""
| Metric | Value |
|---|---|
| Years of Experience (used) | {result['years_experience_used']} |
| Certificate (detected) | {result['certificate_used']} |
| Professional Certification (detected) | {result['professional_certification_used']} |
| Current Level (detected) | {result['current_level_used']} |
| Model Fit Probability (Excellent+Good) | {result['model_fit_probability']}% |
| Skill Match Score | {result['skill_match_score']}% |
| **Hybrid Fit Score** | **{result['hybrid_fit_score']}%** |
"""

    skills_display = "### Matched Skills\n" + "\n".join(f"- {s}" for s in result['matched_skills'])

    if result['notes']:
        skills_display += "\n\n### ⚠️ Notes\n" + "\n".join(f"- {n}" for n in result['notes'])

    return grade_display, details, skills_display


# ============================================================
# BUILD THE UI
# ============================================================

with gr.Blocks(title="Civil Engineering Resume Screener") as demo:
    gr.Markdown("# 🏗️ Civil Engineering Resume Screening Classifier")
    gr.Markdown(
        "Upload a civil engineering resume (PDF or DOCX) to get an automated "
        "fit grade based on experience, qualifications, certifications, and skills. "
        "This is a demo ML project — results are for illustrative screening purposes only, "
        "not a substitute for human review."
    )

    with gr.Row():
        file_input = gr.File(label="Upload Resume (PDF or DOCX)", file_types=[".pdf", ".docx"])

    submit_btn = gr.Button("Screen Resume", variant="primary")

    grade_output = gr.Markdown()
    details_output = gr.Markdown()
    skills_output = gr.Markdown()

    submit_btn.click(
        fn=screen_resume,
        inputs=file_input,
        outputs=[grade_output, details_output, skills_output]
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
