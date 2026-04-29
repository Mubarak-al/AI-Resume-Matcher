import os
import re
import sys
import time
from datetime import datetime
from html import escape
from io import BytesIO
from json import dumps
from collections import Counter

import requests
import fitz
import streamlit as st
import streamlit.components.v1 as components
from docx import Document
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

# -------------------------
# Load ENV
# -------------------------
load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HF_TOKEN = st.secrets["HF_TOKEN"]


# -------------------------
# Local Embedding (FREE)
# -------------------------
class LocalHashEmbeddings(Embeddings):
    def __init__(self, dimensions=384):
        self.dimensions = dimensions

    def _embed(self, text):
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-zA-Z0-9+#.]+", text.lower())
        counts = Counter(tokens)

        for token, count in counts.items():
            vector[hash(token) % self.dimensions] += float(count)

        # Normalize
        length = sum(v * v for v in vector) ** 0.5
        if length:
            vector = [v / length for v in vector]

        return vector

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


# -------------------------
# Vector DB
# -------------------------
@st.cache_resource
def load_vector_db():
    loader = TextLoader("data.txt")
    documents = loader.load()

    if not documents:
        raise ValueError("data.txt is empty.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = LocalHashEmbeddings()
    return FAISS.from_documents(docs, embeddings)


def find_matches(db, resume):
    results = db.similarity_search_with_score(resume, k=3)
    matches = []

    for doc, score in results:
        similarity = float(max(0.0, 1.0 - (score / 2.0)))
        matches.append(
            {
                "content": doc.page_content,
                "similarity": round(similarity * 100, 2),
            }
        )

    return matches


# -------------------------
# Resume Text Extraction
# -------------------------
MAX_UPLOAD_SIZE_MB = 10


def extract_pdf_text(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()
    reader = PdfReader(BytesIO(pdf_bytes))

    if reader.is_encrypted:
        raise ValueError("This PDF is password protected. Upload an unlocked PDF.")

    text_parts = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        except Exception:
            text_parts.append(f"[Could not extract text from page {page_number}]")

    extracted_text = "\n".join(text_parts).strip()

    if extracted_text:
        return extracted_text

    fallback_parts = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        if document.is_encrypted:
            raise ValueError("This PDF is password protected. Upload an unlocked PDF.")

        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text").strip()
            if page_text:
                fallback_parts.append(page_text)
            else:
                fallback_parts.append(f"[Could not extract text from page {page_number}]")

    fallback_text = "\n".join(fallback_parts).strip()
    if fallback_text and not fallback_text.startswith("[Could not extract text"):
        return fallback_text

    raise ValueError(
        "No readable text found in this PDF. It may be a scanned/image PDF. "
        "Upload a DOCX/TXT version or run OCR first."
    )


def extract_docx_text(uploaded_file):
    document = Document(BytesIO(uploaded_file.getvalue()))
    text_parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
    return "\n".join(text_parts).strip()


def extract_txt_text(uploaded_file):
    return uploaded_file.getvalue().decode("utf-8", errors="replace").strip()


def extract_resume_text(uploaded_file):
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_UPLOAD_SIZE_MB:
        raise ValueError(f"File is too large. Upload a file under {MAX_UPLOAD_SIZE_MB} MB.")

    file_extension = uploaded_file.name.rsplit(".", 1)[-1].lower()

    try:
        if file_extension == "pdf":
            return extract_pdf_text(uploaded_file)
        if file_extension == "docx":
            return extract_docx_text(uploaded_file)
        if file_extension == "txt":
            return extract_txt_text(uploaded_file)
    except PdfReadError as exc:
        raise ValueError("Could not read this PDF. Try exporting it again or upload a TXT/DOCX version.") from exc

    raise ValueError("Unsupported file type. Upload a PDF, DOCX, or TXT file.")


# -------------------------
# Prepare Prompt
# -------------------------
def build_prompt(resume, matches):
    context = " ".join([match["content"] for match in matches])

    return f"""
You are an AI career assistant.

Analyze the match between the candidate's resume and the given job descriptions.

Resume:
{resume}

Job Descriptions:
{context}

Instructions:
- Give a realistic match score from 0 to 100
- Be concise and clear
- Do NOT repeat information
- Output ONLY once in the exact format below

Format:

Match Score: X/100
Reason: <short explanation>
Missing Skills: <comma-separated list>
Improvements: <2-3 actionable suggestions>
"""


# -------------------------
# Hugging Face LLM
# -------------------------
API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-120b:fastest"


def generate_llm_response(prompt):
    def clean_llm_output(text):
        text = text.strip()
        text = re.sub(r"(?im)^\s*status:\s*\d+\s*$", "", text)
        text = re.sub(r"(?im)^\s*raw:\s*", "", text).strip()

        match_starts = list(re.finditer(r"(?im)^\s*\**match score\**\s*:", text))
        if len(match_starts) > 1:
            text = text[match_starts[0].start():match_starts[1].start()].strip()
        elif match_starts:
            text = text[match_starts[0].start():].strip()

        lines = [line.rstrip() for line in text.splitlines()]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        midpoint = len(lines) // 2
        if len(lines) % 2 == 0 and lines[:midpoint] == lines[midpoint:]:
            lines = lines[:midpoint]

        text = "\n".join(lines).strip()

        required_sections = ["Match Score:", "Reason:", "Missing Skills:", "Improvements:"]
        for section in required_sections:
            pattern = rf"(?im)^\s*\**{re.escape(section[:-1])}\**\s*:"
            text = re.sub(pattern, section, text, count=1)

        return text

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": 900,
    }

    last_error = None

    for attempt in range(3):  # try up to 3 times
        try:
            session = requests.Session()
            session.trust_env = False
            response = session.post(API_URL, headers=headers, json=payload, timeout=60)

            # Handle temporary provider/rate-limit issues
            if response.status_code in (429, 503):
                time.sleep(5)
                continue

            if response.status_code >= 400:
                return f"LLM HTTP error {response.status_code}: {response.text}"

            if not response.text:
                time.sleep(2)
                continue

            result = response.json()

            if isinstance(result, dict):
                choices = result.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content")
                    if content:
                        return clean_llm_output(content)
                    reasoning = message.get("reasoning")
                    if reasoning:
                        return clean_llm_output(reasoning)

                if "error" in result:
                    time.sleep(5)
                    continue

            return f"Unexpected response: {result}"

        except Exception as e:
            last_error = e
            time.sleep(3)

    if last_error:
        return f"LLM failed after retries: {last_error}"

    return "LLM failed after retries (likely cold model or rate limit)."


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="",
    layout="wide",
)

st.title("AI Resume Matcher")
st.caption("Upload or paste a resume, compare it with the job data, and generate a concise AI evaluation.")

st.markdown(
    """
    <style>
    .app-intro {
        border: 1px solid rgba(49, 51, 63, 0.25);
        border-left: 4px solid #2563eb;
        border-radius: 0.5rem;
        background: rgba(37, 99, 235, 0.10);
        margin: 1rem 0 1.5rem 0;
        padding: 0.9rem 1rem;
    }
    .app-intro-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .app-intro-copy {
        color: var(--text-color);
        opacity: 0.82;
        font-size: 0.95rem;
    }
    .section-gap {
        margin-top: 1.5rem;
    }
    .section-label {
        color: var(--text-color);
        opacity: 0.72;
        font-size: 0.92rem;
        margin: -0.35rem 0 1rem 0;
    }
    .action-row {
        display: flex;
        gap: 0.75rem;
        margin: 0.25rem 0 1rem 0;
    }
    .history-card {
        border: 1px solid rgba(49, 51, 63, 0.16);
        border-radius: 0.5rem;
        background: rgba(255, 255, 255, 0.04);
        margin-bottom: 0.75rem;
        padding: 0.85rem;
    }
    .history-title {
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .history-meta {
        color: var(--text-color);
        opacity: 0.72;
        font-size: 0.82rem;
        margin-bottom: 0.35rem;
    }
    .history-preview {
        color: var(--text-color);
        opacity: 0.86;
        font-size: 0.86rem;
        line-height: 1.45;
    }
    .match-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.75rem;
    }
    .match-card-title {
        font-size: 1rem;
        font-weight: 700;
    }
    .score-pill {
        border-radius: 999px;
        color: #ffffff;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 0.2rem 0.65rem;
        white-space: nowrap;
    }
    .score-green {
        background: #16a34a;
    }
    .score-yellow {
        background: #ca8a04;
    }
    .score-red {
        background: #dc2626;
    }
    .job-description {
        color: var(--text-color);
        opacity: 0.88;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 0.9rem;
    }
    .llm-output {
        border: 1px solid rgba(120, 144, 180, 0.35);
        border-radius: 0.5rem;
        padding: 1.05rem 1.15rem;
        background: rgba(255, 255, 255, 0.06);
        color: var(--text-color);
        line-height: 1.6;
    }
    .llm-output strong {
        color: var(--text-color);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_score_theme(score):
    if score > 70:
        return "score-green", "Strong"
    if score >= 50:
        return "score-yellow", "Moderate"
    return "score-red", "Low"


def build_results_report(resume, matches, output):
    match_lines = []

    for index, match in enumerate(matches, start=1):
        match_lines.append(
            "\n".join(
                [
                    f"Match {index}",
                    f"Similarity Score: {match['similarity']}%",
                    "Job Description:",
                    match["content"],
                ]
            )
        )

    return "\n\n".join(
        [
            "AI Resume Matcher Results",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Resume:",
            resume,
            "Top Matches:",
            "\n\n".join(match_lines),
            "LLM Evaluation:",
            output or "No LLM output available.",
        ]
    )


def render_copy_button(text):
    safe_text = dumps(text or "")

    components.html(
        f"""
        <button
            id="copy-llm-output"
            style="
                border: 1px solid rgba(120, 144, 180, 0.45);
                border-radius: 8px;
                background: #111827;
                color: #ffffff;
                cursor: pointer;
                font: 600 14px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                padding: 0.55rem 0.85rem;
                width: 100%;
            "
        >
            Copy LLM Output
        </button>
        <script>
        const button = document.getElementById("copy-llm-output");
        button.addEventListener("click", async () => {{
            await navigator.clipboard.writeText({safe_text});
            button.textContent = "Copied";
            setTimeout(() => button.textContent = "Copy LLM Output", 1400);
        }});
        </script>
        """,
        height=48,
    )


st.markdown(
    """
    <div class="app-intro">
        <div class="app-intro-title">Resume-to-job matching workspace</div>
        <div class="app-intro-copy">
            Review extracted resume text, run semantic matching, and inspect a focused AI evaluation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    db = load_vector_db()
except FileNotFoundError:
    st.error("data.txt not found. Create it first.")
    st.stop()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.subheader("Status")
    st.write("Vector database loaded")
    st.write(f"LLM model: `{MODEL_NAME}`")
    if HF_TOKEN:
        st.success("HF_TOKEN loaded")
    else:
        st.error("HF_TOKEN not set in .env")

st.divider()
st.header("Resume Input")
st.markdown(
    '<div class="section-label">Upload a resume file or paste text directly before analysis.</div>',
    unsafe_allow_html=True,
)

st.subheader("Upload Resume")
uploaded_file = st.file_uploader(
    "Upload a resume file",
    type=["pdf", "docx", "txt"],
)

if "resume_editor" not in st.session_state:
    st.session_state.resume_editor = ""

if "uploaded_resume_id" not in st.session_state:
    st.session_state.uploaded_resume_id = None

if "upload_status" not in st.session_state:
    st.session_state.upload_status = None

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if uploaded_file is None:
    st.session_state.uploaded_resume_id = None
    st.session_state.upload_status = None
else:
    uploaded_resume_id = f"{uploaded_file.name}:{uploaded_file.size}"

    if uploaded_resume_id != st.session_state.uploaded_resume_id:
        try:
            with st.spinner("Extracting resume text..."):
                extracted_text = extract_resume_text(uploaded_file)

            if not extracted_text:
                st.session_state.upload_status = (
                    "error",
                    "Text extraction failed. The uploaded file did not contain readable text. "
                    "If this is a scanned PDF, upload a DOCX/TXT version or use OCR first.",
                )
            else:
                st.session_state.resume_editor = extracted_text
                st.session_state.uploaded_resume_id = uploaded_resume_id
                st.session_state.upload_status = (
                    "success",
                    f"Resume uploaded successfully: {uploaded_file.name}",
                )
        except Exception as exc:
            st.session_state.upload_status = ("error", f"Resume extraction failed: {exc}")

if st.session_state.upload_status:
    status_type, status_message = st.session_state.upload_status
    if status_type == "success":
        st.success(status_message)
    else:
        st.error(status_message)

resume = st.text_area(
    "Extracted or pasted resume text",
    placeholder="Paste the candidate resume here...",
    height=240,
    key="resume_editor",
)

analyze = st.button("Analyze Resume", type="primary", use_container_width=True)

if analyze:
    resume = resume.strip()

    if not resume:
        if uploaded_file is not None:
            try:
                with st.spinner("Extracting resume text before analysis..."):
                    resume = extract_resume_text(uploaded_file).strip()
                st.session_state.resume_editor = resume
            except Exception as exc:
                st.error(f"Resume extraction failed: {exc}")
                st.stop()

        if not resume:
            st.warning(
                "Resume cannot be empty. The uploaded file did not provide readable text. "
                "If this is a scanned PDF, upload a DOCX/TXT version or use OCR first."
            )
            st.stop()

    st.divider()
    st.header("Analysis Results")
    st.markdown(
        '<div class="section-label">Top retrieved job matches and a concise LLM evaluation.</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Analyzing resume and generating evaluation..."):
        matches = find_matches(db, resume)
        prompt = build_prompt(resume, matches)
        output = None
        if HF_TOKEN:
            output = generate_llm_response(prompt)
        report = build_results_report(resume, matches, output)

    st.session_state.analysis_history.insert(
        0,
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "resume_preview": resume[:180],
            "matches": matches,
            "output": output,
            "report": report,
        },
    )
    st.session_state.analysis_history = st.session_state.analysis_history[:5]

    matches_col, evaluation_col = st.columns([1, 1], gap="large")

    with matches_col:
        st.subheader("Top Job Matches")
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

        for index, match in enumerate(matches, start=1):
            score = float(match["similarity"])
            score_class, score_label = get_score_theme(score)

            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="match-card-header">
                        <div class="match-card-title">Match {index}</div>
                        <div class="score-pill {score_class}">{score_label}</div>
                    </div>
                    <div class="job-description">{match["content"]}</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(score / 100)
                st.metric("Match Percentage", f"{score}%")
                st.write("")

    with evaluation_col:
        st.subheader("AI Evaluation")
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

        if not HF_TOKEN:
            st.error("HF_TOKEN not set in .env")
        else:
            st.markdown(
                f'<div class="llm-output">{escape(output).replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )
            action_col_1, action_col_2 = st.columns(2)

            with action_col_1:
                st.download_button(
                    "Download Results",
                    data=report,
                    file_name="resume_match_results.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            with action_col_2:
                render_copy_button(output)

    st.divider()
    st.header("Analysis History")
    st.markdown(
        '<div class="section-label">Recent analyses from this Streamlit session.</div>',
        unsafe_allow_html=True,
    )

    for index, item in enumerate(st.session_state.analysis_history, start=1):
        best_score = max((float(match["similarity"]) for match in item["matches"]), default=0.0)
        score_class, score_label = get_score_theme(best_score)

        with st.expander(f"Analysis {index} - {item['timestamp']} - Best match {best_score}%"):
            st.markdown(
                f"""
                <div class="history-card">
                    <div class="history-title">Resume Preview</div>
                    <div class="history-meta">
                        <span class="score-pill {score_class}">{score_label}</span>
                    </div>
                    <div class="history-preview">{escape(item["resume_preview"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("**LLM Evaluation**")
            st.markdown(item["output"] or "No LLM output available.")
            st.download_button(
                "Download This Analysis",
                data=item["report"],
                file_name=f"resume_match_results_{index}.txt",
                mime="text/plain",
                use_container_width=True,
                key=f"download_history_{index}_{item['timestamp']}",
            )
else:
    st.divider()
    st.info("Upload a resume or paste resume text, then click Analyze Resume.")

    if st.session_state.analysis_history:
        st.header("Analysis History")
        for index, item in enumerate(st.session_state.analysis_history, start=1):
            with st.expander(f"Analysis {index} - {item['timestamp']}"):
                st.markdown(item["output"] or "No LLM output available.")
                st.download_button(
                    "Download This Analysis",
                    data=item["report"],
                    file_name=f"resume_match_results_{index}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key=f"idle_download_history_{index}_{item['timestamp']}",
                )
