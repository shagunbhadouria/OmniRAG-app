import streamlit as st
import os
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
from bs4 import BeautifulSoup
import ast

# --- Page Configuration ---
st.set_page_config(page_title="OmniRAG", layout="wide")

# --- Session State Initialization ---
if "active_module" not in st.session_state:
    st.session_state.active_module = None
if "doc_vector_store" not in st.session_state:
    st.session_state.doc_vector_store = None
if "code_vector_store" not in st.session_state:
    st.session_state.code_vector_store = None
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None

# --- Helper Functions ---
def get_text_from_pdf(file):
    return "".join(page.extract_text() for page in PdfReader(file).pages)
def get_text_from_txt(file):
    return file.getvalue().decode("utf-8")
def get_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        return ' '.join(p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
    except Exception: return ""

# UPDATED: Added error handling for non-Python content
def parse_python_code_from_text(content):
    try:
        tree = ast.parse(content)
        parsed_items = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node) or "No docstring."
                source_code = ast.get_source_segment(content, node)
                item_text = f"Type: {type(node).__name__}\nName: {node.name}\nDocstring: {docstring}\nCode:\n{source_code}"
                parsed_items.append(item_text)
        return parsed_items
    except SyntaxError:
        st.error("The provided content is not valid Python code. Please provide a URL to a raw code file.")
        return None

# --- Main App Title ---
st.title("🧠 OmniRAG")
st.write("Your all-in-one local intelligence system. Choose a tool below to get started.")
st.divider()

# --- Module Selection Buttons ---
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🧠 Knowledge Explorer", use_container_width=True):
        st.session_state.active_module = "explorer"
with col2:
    if st.button("💻 AI Code Insight", use_container_width=True):
        st.session_state.active_module = "code"
with col3:
    if st.button("📊 Smart Data Analyst", use_container_width=True):
        st.session_state.active_module = "data"

st.divider()

# --- Contextual Controls (Main Page) ---
if st.session_state.active_module == "explorer":
    with st.expander("Load Sources for Knowledge Explorer", expanded=True):
        # ... (Rest of the app logic remains the same)
        uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "txt"], accept_multiple_files=True)
        url_input = st.text_input("Or enter a website URL")
        if st.button("Index Documents"):
            if uploaded_files or url_input:
                with st.spinner("Processing..."):
                    all_text = ""
                    if uploaded_files:
                        for file in uploaded_files:
                            all_text += get_text_from_pdf(file) if file.type == "application/pdf" else get_text_from_txt(file)
                    if url_input:
                        all_text += get_text_from_url(url_input)
                    if all_text:
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        chunks = text_splitter.split_text(all_text)
                        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        st.session_state.doc_vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                        st.success("Documents indexed!")
                    else: st.warning("No text extracted.")

elif st.session_state.active_module == "code":
    with st.expander("Load Sources for AI Code Insight", expanded=True):
        uploaded_code_file = st.file_uploader("Upload a Python file", type=["py"])
        code_url_input = st.text_input("Or enter a raw code URL (e.g., from GitHub)")
        if st.button("Index Code"):
            code_content = ""
            if uploaded_code_file:
                code_content = uploaded_code_file.getvalue().decode("utf-8")
            elif code_url_input:
                try: code_content = requests.get(code_url_input).text
                except Exception: st.error("Failed to fetch code from URL.")

            if code_content:
                with st.spinner("Indexing code..."):
                    parsed_items = parse_python_code_from_text(code_content)
                    if parsed_items: # Will be None if there was a SyntaxError
                        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        st.session_state.code_vector_store = FAISS.from_texts(parsed_items, embedding=embeddings)
                        st.success("Code indexed!")
                    elif parsed_items is not None:
                        st.warning("No functions or classes found to index.")

elif st.session_state.active_module == "data":
    with st.expander("Load Sources for Smart Data Analyst", expanded=True):
        uploaded_csv_file = st.file_uploader("Upload a CSV file", type=["csv"])
        csv_url_input = st.text_input("Or enter a raw CSV URL")
        if st.button("Analyze CSV"):
            if uploaded_csv_file:
                st.session_state.dataframe = pd.read_csv(uploaded_csv_file)
                st.success("CSV file loaded!")
            elif csv_url_input:
                try:
                    st.session_state.dataframe = pd.read_csv(csv_url_input)
                    st.success("CSV from URL loaded!")
                except Exception: st.error("Failed to load CSV from URL.")
            else:
                st.warning("Please provide a CSV source.")

# --- Main Area (Contextual Display / Search) ---
if st.session_state.active_module == "explorer":
    st.subheader("Search Your Documents")
    prompt = st.text_input("Ask a question about your documents...", key="doc_search")
    if prompt and st.session_state.doc_vector_store:
        docs = st.session_state.doc_vector_store.similarity_search(prompt)
        st.write("### Relevant Passages Found:")
        for doc in docs:
            st.markdown(f"> {doc.page_content}")
            st.divider()

elif st.session_state.active_module == "code":
    st.subheader("Search Your Codebase")
    prompt = st.text_input("Ask a question about your code...", key="code_search")
    if prompt and st.session_state.code_vector_store:
        docs = st.session_state.code_vector_store.similarity_search(prompt)
        st.write("### Relevant Code Snippets Found:")
        for doc in docs:
            st.code(doc.page_content, language='python')

elif st.session_state.active_module == "data":
    st.subheader("Data Analysis Results")
    if st.session_state.dataframe is not None:
        st.dataframe(st.session_state.dataframe)
        st.write("### Basic Statistics")
        st.dataframe(st.session_state.dataframe.describe())
    else:
        st.info("Click a main button and load a CSV to see results.")
