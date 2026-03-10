import streamlit as st
import PyPDF2
import io
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Resume Critiquer", page_icon="📄", layout="centered")

st.title("AI Resume Critiquer")
st.markdown("Upload your resume and get AI-powered feedback tailored to your needs!")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if "file_content" not in st.session_state:
    st.session_state.file_content = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None


def extract_text_from_pdf(file_bytes):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


def extract_text_from_file(uploaded_file):
    file_bytes = uploaded_file.getvalue()

    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(file_bytes)

    return file_bytes.decode("utf-8", errors="ignore").strip()


def get_resume_feedback(prompt, api_key, retries=3):
    client = OpenAI(api_key=api_key, timeout=45)
    last_error = None

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert resume reviewer and career coach. "
                            "Be practical, honest, and specific."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.2,
                max_tokens=700,
            )
            return response.choices[0].message.content

        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(2)

    raise last_error


uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file is not None:
    if st.session_state.file_name != uploaded_file.name:
        st.session_state.file_content = extract_text_from_file(uploaded_file)
        st.session_state.file_name = uploaded_file.name

with st.form("resume_form"):
    job_role = st.text_input("Enter the job role you're targeting")
    analyze = st.form_submit_button("Analyze Resume")


if analyze:
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is missing from your .env file.")
        st.stop()

    if not uploaded_file:
        st.warning("Please upload a resume first.")
        st.stop()

    file_content = st.session_state.file_content

    if not file_content or not file_content.strip():
        st.error("File does not have any content.")
        st.stop()

    target_role = job_role.strip() if job_role.strip() else "general job applications"

    prompt = f"""
Review this resume for the target role: {target_role}

Give a structured response with these exact sections:

1. Overall Fit Score (/10)
2. Is This Resume Suitable for This Role?
3. What Matches the Role
4. What Is Missing
5. Top 5 Improvements
6. 3 Rewritten CV Bullet Points
7. Final Verdict

Important instructions:
- Base your answer on the actual resume content.
- If the target role is very different from the candidate's background, say that clearly.
- Explain how well the resume matches the target role.
- Be honest and specific, not generic.
- Keep the answer concise but useful.

Resume:
{file_content[:12000]}
""".strip()

    try:
        with st.spinner("Analyzing resume..."):
            result = get_resume_feedback(prompt, OPENAI_API_KEY)

        st.markdown("### Analysis Results")
        st.markdown(result)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")