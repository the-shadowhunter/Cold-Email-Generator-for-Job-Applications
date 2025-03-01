import uuid
import PyPDF2
import chromadb
import requests
import re
import os
from chromadb.config import Settings  # for legacy use if needed
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
import streamlit as st 

load_dotenv()
GROQ_API_KEY = st.secrets['GROQ_API_KEY']
# Initialize persistent ChromaDB client with new API
client = chromadb.PersistentClient(path="resumes_db")
collection = client.get_or_create_collection("resumes")

# Initialize the SentenceTransformer embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    return embedding_model.encode(text).tolist()

def parse_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# def store_resume(resume_text, metadata):
#     embedding = embed_text(resume_text)
#     doc_id = str(uuid.uuid4())
#     collection.add(ids=[doc_id], documents=[resume_text], metadatas=[metadata], embeddings=[embedding])
# Add to existing imports
from datetime import datetime

# Modified store_resume function with better metadata
def store_resume(resume_text, user_name="User"):
    embedding = embed_text(resume_text)
    doc_id = str(uuid.uuid4())
    metadata = {
        "name": user_name,
        "timestamp": datetime.now().isoformat()
    }
    collection.add(
        ids=[doc_id],
        documents=[resume_text],
        metadatas=[metadata],
        embeddings=[embedding]
    )

# Improved resume retrieval function
def get_stored_resumes():
    """Returns list of resumes sorted by timestamp (newest first)"""
    results = collection.get()
    if not results["ids"]:
        return []
    
    # Combine metadata with documents and handle missing timestamps
    resumes = []
    for idx in range(len(results["ids"])):
        resumes.append({
            "text": results["documents"][idx],
            "metadata": results["metadatas"][idx] or {}  # Handle null metadata
        })
    
    # Sort with fallback for missing timestamps
    return sorted(
        resumes,
        key=lambda x: x.get("metadata", {}).get("timestamp", ""),  # Safe access
        reverse=True
    )

def parse_job_posting(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        job_desc_container = soup.find("div", class_=re.compile("job[-_ ]?description", re.I))
        if job_desc_container:
            job_description = job_desc_container.get_text(" ", strip=True)
        else:
            job_description = soup.get_text(" ", strip=True)
        return job_description
    except Exception as e:
        print(f"Error parsing job posting: {e}")
        return ""

def clean_text(text):
    text = re.sub(r'<[^>]*?>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()
    text = ' '.join(text.split())
    return text

def extract_links(resume_text):
    # Updated regex patterns to allow an optional trailing slash
    linkedin_pattern = r"https?://(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+/?"
    github_pattern = r"https?://(?:www\.)?github\.com/[a-zA-Z0-9_-]+/?"
    linkedin_links = re.findall(linkedin_pattern, resume_text)
    github_links = re.findall(github_pattern, resume_text)
    return {"linkedin": linkedin_links, "github": github_links}

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"  # Updated to a supported model
        )
    
    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills`, `responsibilities` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]
    
    def write_cold_email(self, job_description, resume_details):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION & RESPONSIBILITIES:
            {job_description}

            ### RESUME DETAILS:
            {resume_details}


            ### INSTRUCTIONS:
            You are a student applying for this job opportunity. 
            Using your resume details and the provided contact links, craft a professional and enthusiastic email that clearly maps your skills, experiences, and achievements to the job requirements and responsibilities. 
            Emphasize how your background is a good match for the role and include a call-to-action. Avoid any introductory preamble and provide only the final email text.
            Do NOT include any URLs like github and linkedin.
            
            
            ### FINAL EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({
            "job_description": job_description,
            "resume_details": resume_details,
        })
        return res.content
    
    
import streamlit as st
from io import BytesIO
from main import parse_pdf, clean_text, extract_links, store_resume, parse_job_posting, Chain, get_stored_resumes

chain_instance = Chain()

st.title("Cold Email Generator")
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload resume (optional - updates your profile)
2. Enter job URL
3. Select one or more profiles
4. Generate combined cold email
""")

# Resume upload section
uploaded_file = st.file_uploader("Update Resume Profile (PDF)", type=["pdf"])
if uploaded_file:
    user_name = st.text_input("Enter your name for reference", value="Candidate")
    if st.button("Save Resume Profile"):
        resume_text = parse_pdf(BytesIO(uploaded_file.read()))
        cleaned_resume = clean_text(resume_text)
        store_resume(cleaned_resume, user_name)
        st.success("Resume profile updated successfully!")

# Job URL input
job_url = st.text_input("Enter Job Posting URL")

# Resume selection from stored profiles
stored_resumes = get_stored_resumes()
selected_resume = []

if stored_resumes:
    resume_options = [
        f"{r['metadata']['name']} - {r['metadata']['timestamp'][:10]}"
        for r in stored_resumes
    ]
    selected = st.multiselect(  # Changed to multiselect
        "Choose Profile(s)",
        options=resume_options,
        default=[resume_options[0]] if resume_options else []
    )
    selected_resumes = [stored_resumes[resume_options.index(s)] for s in selected]
else:
    st.warning("No resumes stored yet. Upload a resume to begin.")

# Modified email generation section
if st.button("Generate Cold Email") and job_url:
    if not selected_resumes:
        st.error("Please select at least one resume profile")
        st.stop()
    
    try:
        # Combine text from all selected resumes
        combined_resume_text = "\n\n".join([res["text"] for res in selected_resumes])
        
        # Aggregate links from all resumes (using sets to avoid duplicates)
        linkedin_links = set()
        github_links = set()
        for res in selected_resumes:
            links = extract_links(res["text"])
            linkedin_links.update(links["linkedin"])
            github_links.update(links["github"])
        
        job_description = parse_job_posting(job_url)
        jobs = chain_instance.extract_jobs(job_description)
        job_desc = jobs[0]["description"] if jobs else job_description
        
        # Generate email using combined resume data
        cold_email = chain_instance.write_cold_email(
            job_desc,
            combined_resume_text
        )
        
        # Display results
        st.subheader("Generated Cold Email")
        st.text_area("", value=cold_email, height=300)
        
        # Show selection summary
        st.subheader("Selected Profiles")
        for res in selected_resumes:
            meta = res["metadata"]
            st.write(f"- {meta['name']} ({meta['timestamp'][:10]})")
        
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")