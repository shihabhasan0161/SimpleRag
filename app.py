
import os
import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag import (
    load_documents,
    split_documents,
    create_vectorstore,
    generate_answer,
)

load_dotenv()


def get_key_fingerprint(api_key: str) -> str:
    return hashlib.sha256(
        api_key.encode("utf-8")
    ).hexdigest()


st.set_page_config(
    page_title="Document Q&A Chatbot",
    page_icon="📄"
)

st.title("📄 Document Q&A Chatbot")

api_key = os.getenv("OPENAI_API_KEY")

# if no api key in .env then get it from the user
if not api_key:
    api_key = st.sidebar.text_input(
        "Enter your OpenAI API key",
        type="password"
    )

if not api_key:
    st.warning("Please enter your OpenAI API key.")
    st.stop()

current_fingerprint = get_key_fingerprint(api_key)

# 2. Invalidate indexed documents if API key changes
if (
    "vectorstore" in st.session_state
    and st.session_state.get("key_fingerprint")
    != current_fingerprint
):
    del st.session_state["vectorstore"]
    st.session_state.pop("key_fingerprint", None)
    st.warning(
        "API key changed. Please process your documents again."
    )

# 3. Upload documents
uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files and st.button("Process Documents"):

    if len(uploaded_files) > 5:
        st.error("Maximum 5 documents allowed.")
        st.stop()

    try:
        with st.spinner("Processing documents..."):

            # Save uploaded files temporarily
            with tempfile.TemporaryDirectory() as temp_dir:
                file_paths = []

                for uploaded_file in uploaded_files:
                    file_path = (
                        Path(temp_dir)
                        / Path(uploaded_file.name).name
                    )

                    file_path.write_bytes(
                        uploaded_file.getvalue()
                    )

                    file_paths.append(str(file_path))

                documents = load_documents(file_paths)

            # Split into chunks
            chunks = split_documents(documents)

            if not chunks:
                st.error("No readable text found in the documents.")
                st.stop()

            # Create in-memory ChromaDB
            vectorstore = create_vectorstore(
                chunks,
                api_key
            )

            # Save vectorstore to user's Streamlit session
            st.session_state.vectorstore = vectorstore
            st.session_state.key_fingerprint = (
                current_fingerprint
            )

        st.success("Documents processed successfully!")

    except Exception as e:
        st.error(
            f"Document processing failed: {type(e).__name__}"
        )

# 4. Chat with documents
if "vectorstore" in st.session_state:

    question = st.chat_input(
        "Ask a question about your documents..."
    )

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    answer = generate_answer(
                        question,
                        st.session_state.vectorstore,
                        api_key
                    )

                st.write(answer)

            except Exception as e:
                st.error(
                    f"Unable to generate answer: {type(e).__name__}"
                )