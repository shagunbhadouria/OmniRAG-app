import streamlit as st
import os
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
import ast

# --- Tesseract Configuration (Auto-detects Windows vs Cloud) ---
if os.name == 'nt': # If running on Windows
    import pytesseract
    from PIL import Image
    # This is the path for your local computer
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else: # If running on Linux (Streamlit Cloud)
    import pytesseract
    from PIL import Image
    # On Cloud, we rely on the system path (configured via packages.txt)

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
    except SyntaxError: return None

# --- Main App Title ---
st.title("🧠 OmniRAG")
st.write("Your all-in-one local intelligence system. Choose a tool below to get started.")

# --- AI Settings (LLM Controls) ---
# We place this in an expander at the top so it's accessible but not cluttered
with st.expander("⚙️ AI Settings (LLM Mode)", expanded=False):
    col_llm_1, col_llm_2 = st.columns(2)
    with col_llm_1:
        llm_mode = st.toggle("Enable LLM Mode (Generative Answers)", value=False)
    with col_llm_2:
        selected_llm = "phi3"
        if llm_mode:
            # You can add more models here if you download them in Ollama
            selected_llm = st.selectbox("Select Model:", ["phi3", "llama3", "mistral", "gemma:2b"])
            st.caption(f"Using {selected_llm} via Ollama.")

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

st.write("") # Spacer

# --- Contextual Controls (The "Pop-up" Sections) ---
if st.session_state.active_module == "explorer":
    with st.container(border=True):
        st.subheader("📂 Load Sources for Knowledge Explorer")
        uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "txt"], accept_multiple_files=True)
        url_input = st.text_input("Or enter a website URL")
        if st.button("Index Documents"):
            if uploaded_files or url_input:
                with st.spinner("Processing..."):
                    all_text = ""
                    if uploaded_files:
                        for file in uploaded_files:
                            if file.type == "application/pdf":
                                all_text += get_text_from_pdf(file)
                            else:
                                all_text += get_text_from_txt(file)
                    if url_input:
                        all_text += get_text_from_url(url_input)
                    
                    if all_text:
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        chunks = text_splitter.split_text(all_text)
                        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        st.session_state.doc_vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                        st.success("Documents indexed successfully!")
                    else: st.warning("No text could be extracted.")

elif st.session_state.active_module == "code":
    with st.container(border=True):
        st.subheader("📂 Load Sources for AI Code Insight")
        uploaded_code_file = st.file_uploader("Upload a Python file", type=["py"])
        code_url_input = st.text_input("Or enter a raw code URL (e.g., raw.githubusercontent.com...)")
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
                    if parsed_items:
                        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        st.session_state.code_vector_store = FAISS.from_texts(parsed_items, embedding=embeddings)
                        st.success("Codebase indexed successfully!")
                    elif parsed_items is not None:
                        st.warning("No functions or classes found to index.")
                    else:
                        st.error("Invalid Python code syntax.")

elif st.session_state.active_module == "data":
    with st.container(border=True):
        st.subheader("📂 Load Sources for Smart Data Analyst")
        uploaded_csv_file = st.file_uploader("Upload a CSV file", type=["csv"])
        csv_url_input = st.text_input("Or enter a raw CSV URL")
        if st.button("Analyze CSV"):
            df_source = uploaded_csv_file if uploaded_csv_file else csv_url_input
            if df_source:
                try:
                    st.session_state.dataframe = pd.read_csv(df_source)
                    st.success("CSV file loaded successfully!")
                except Exception: st.error("Failed to load CSV.")
            else:
                st.warning("Please provide a CSV source.")

st.divider()

# --- Main Results Area ---
if st.session_state.active_module == "explorer":
    st.subheader("🔍 Search Your Documents")
    prompt = st.text_input("Ask a question about your documents...", key="doc_search")
    if prompt and st.session_state.doc_vector_store:
        docs = st.session_state.doc_vector_store.similarity_search(prompt)
        if not docs:
            st.info("No relevant passages found.")
        else:
            if llm_mode:
                with st.spinner(f"Generating answer with {selected_llm}..."):
                    try:
                        llm = Ollama(model=selected_llm)
                        prompt_template = PromptTemplate.from_template(
                            "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}"
                        )
                        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)
                        response = chain.invoke({"input_documents": docs, "question": prompt})
                        st.write("### Generated Answer")
                        st.write(response["output_text"])
                    except Exception as e:
                        st.error(f"Error connecting to Ollama: {e}. Make sure Ollama is running.")
            else:
                st.write("### Relevant Passages Found (Fast Mode):")
                for doc in docs:
                    st.markdown(f"> {doc.page_content}")
                    st.divider()

elif st.session_state.active_module == "code":
    st.subheader("🔍 Search Your Codebase")
    prompt = st.text_input("Ask a question about your code...", key="code_search")
    if prompt and st.session_state.code_vector_store:
        docs = st.session_state.code_vector_store.similarity_search(prompt)
        if not docs:
            st.info("No relevant code snippets found.")
        else:
            if llm_mode:
                with st.spinner(f"Generating explanation with {selected_llm}..."):
                    try:
                        llm = Ollama(model=selected_llm)
                        prompt_template = PromptTemplate.from_template(
                            "Explain what the following code does, based on the user's question.\n\nContext:\n{context}\n\nQuestion: {question}"
                        )
                        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)
                        response = chain.invoke({"input_documents": docs, "question": prompt})
                        st.write("### Generated Explanation")
                        st.write(response["output_text"])
                    except Exception as e:
                        st.error(f"Error connecting to Ollama: {e}")
            else:
                st.write("### Relevant Code Snippets Found (Fast Mode):")
                for doc in docs:
                    st.code(doc.page_content, language='python')

elif st.session_state.active_module == "data":
    st.subheader("📊 Data Analysis Results")
    if st.session_state.dataframe is not None:
        st.dataframe(st.session_state.dataframe)
        st.write("### Basic Statistics")
        st.dataframe(st.session_state.dataframe.describe())
    else:
        st.info("Upload and analyze a CSV to see results.")
