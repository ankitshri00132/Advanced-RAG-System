"""
Streamlit frontend for Advanced RAG System.

Provides:
  1. PDF upload → calls POST /ingest
  2. Chat interface → calls POST /retrieve with document_id scoping
"""

import streamlit as st
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL = "http://localhost:8000"  # FastAPI backend

st.set_page_config(
    page_title="RAG Chat",
    page_icon="📄",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "document_id" not in st.session_state:
    st.session_state.document_id = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Sidebar – Document Upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF to ingest into the RAG system.",
    )

    if uploaded_file and st.button("Ingest", type="primary"):
        with st.spinner("Uploading & ingesting..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/ingest",
                    files={
                        "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

                st.session_state.document_id = data["document_id"]
                st.session_state.document_name = uploaded_file.name
                # Reset chat history on new document
                st.session_state.messages = []

                st.success(
                    f"**{uploaded_file.name}** ingested!\n\n"
                    f"- Pages: {data['pages_loaded']}\n"
                    f"- Chunks: {data['chunks_created']}"
                )
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the backend API. Is it running?")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()

    # Show active document info
    if st.session_state.document_id:
        st.caption(f"**Active doc:** {st.session_state.document_name}")
        st.caption(f"`{st.session_state.document_id}`")

        if st.button("Clear document"):
            st.session_state.document_id = None
            st.session_state.document_name = None
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("No document loaded. Upload a PDF to get started, or chat directly.")

# ---------------------------------------------------------------------------
# Main area – Chat
# ---------------------------------------------------------------------------
st.title("💬 RAG Chat")

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {"query": prompt}
                if st.session_state.document_id:
                    payload["document_id"] = st.session_state.document_id

                resp = requests.post(
                    f"{API_BASE_URL}/retrieve",
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

                answer = data.get("answer", "No answer returned.")
                st.markdown(answer)

                # Optionally show sources in an expander
                sources = data.get("sources", [])
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for i, src in enumerate(sources, 1):
                            meta = src.get("metadata", {})
                            file_name = meta.get("file_name", "unknown")
                            page = meta.get("page", "?")
                            score = src.get("score", "")
                            snippet = src.get("document", "")[:300]
                            st.markdown(
                                f"**{i}. {file_name}** — page {page}"
                                + (f" (score: {score:.3f})" if isinstance(score, float) else "")
                            )
                            st.caption(snippet)

            except requests.exceptions.ConnectionError:
                answer = "⚠️ Cannot reach the backend API. Is it running?"
                st.error(answer)
            except Exception as e:
                answer = f"⚠️ Error: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
