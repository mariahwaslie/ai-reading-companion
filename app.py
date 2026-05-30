from pathlib import Path
import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
from core.parser import parse_epub, parse_pdf
from core.chunker import chunk_text
from core.vectorstore import VectorStore
from core.agent import Agent

UPLOAD_DIR = Path("sample_books/")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Setup
# -----------------------------
load_dotenv()

st.set_page_config(page_title="AI Reading Companion", layout="wide")
st.title("📚 AI Reading Companion")

# -----------------------------
# Session State Init
# -----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = VectorStore()

if "agent" not in st.session_state:
    st.session_state.agent = Agent(st.session_state.vectorstore)

if "indexed" not in st.session_state:
    st.session_state.indexed = False


# -----------------------------
# Upload + Index Book
# -----------------------------
st.header("1. Upload / Load Book")

uploaded_file = st.file_uploader("Upload EPUB file", type=["epub","pdf"],)

if uploaded_file is not None:

    if st.button("Build Knowledge Base"):
        with st.spinner("Saving file..."):
            file_bytes = uploaded_file.getvalue()
        
        with st.spinner("Parsing book..."):
            if Path(uploaded_file.name).suffix.lower() == ".pdf":
                text= parse_pdf(file_bytes)
            elif Path(uploaded_file.name).suffix.lower() == ".epub":
                text = parse_epub(file_bytes)
            else: 
                st.write("Unsupported file type. Please upload an EPUB or PDF.")
                st.stop()
        with st.spinner("Chunking text..."):
            chunks = chunk_text(text)

        with st.spinner("Building vector database..."):
            st.session_state.vectorstore.build_index(chunks)

        st.session_state.indexed = True

        st.success(f"Indexed book with {len(chunks)} chunks!")


# -----------------------------
# Query Section
# -----------------------------
st.header("2. Ask Questions")

if not st.session_state.indexed:
    st.info("Upload and index a book first.")
else:
    query = st.text_input("Ask something about the book")

    if query:

        with st.spinner("Thinking..."):
            answer = st.session_state.agent.run(query)

        st.subheader("Answer")
        st.write(answer)