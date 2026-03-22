"""
Streamlit UI for the AI Data Science Tutor (RAG + chat memory).
"""

from __future__ import annotations

import traceback

import streamlit as st
from langchain.memory import ConversationBufferMemory

from rag_pipeline import (
    build_and_save_vectorstore,
    build_rag_chain,
    get_embeddings,
    get_or_create_vectorstore,
    invoke_rag_chain,
)
from utils import (
    DATA_DIR,
    VECTORSTORE_DIR,
    ensure_placeholder_pdfs,
    get_openai_api_key,
    load_environment,
    list_pdf_paths,
    sanitize_filename,
    validate_api_key,
    validate_user_question,
)


def _init_session_state() -> None:
    if "memory" not in st.session_state:
        # LangChain memory: stores chat messages for retrieval + display
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
    if "vectorstore_ready" not in st.session_state:
        st.session_state.vectorstore_ready = False
    if "status_message" not in st.session_state:
        st.session_state.status_message = ""
    if "last_interview_mode" not in st.session_state:
        st.session_state.last_interview_mode = None


def _clear_chat() -> None:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)


def _rebuild_knowledge_base() -> tuple[bool, str]:
    """Rebuild FAISS from all PDFs in data/. Returns (success, message)."""
    try:
        pdfs = list_pdf_paths()
        if not pdfs:
            return False, "No PDF files found in the data/ folder. Upload a PDF first."
        embeddings = get_embeddings()
        build_and_save_vectorstore(pdfs, embeddings, VECTORSTORE_DIR)
        return True, f"Knowledge base rebuilt from {len(pdfs)} PDF(s)."
    except Exception as exc:  # noqa: BLE001 — show user-friendly error in UI
        return False, f"Rebuild failed: {exc}\n\n{traceback.format_exc()}"


def _ensure_vectorstore() -> tuple[bool, str]:
    """
    Ensure FAISS exists; create placeholder PDFs if needed.
    Returns (ok, user_message).
    """
    key_ok, key_msg = validate_api_key()
    if not key_ok:
        return False, key_msg

    created = ensure_placeholder_pdfs()
    if created:
        st.sidebar.info("Created a sample PDF in data/ because the folder was empty.")

    try:
        vs, msg = get_or_create_vectorstore()
        st.session_state.vectorstore = vs
        st.session_state.vectorstore_ready = True
        st.session_state.status_message = msg
        return True, msg
    except FileNotFoundError as e:
        return False, str(e)
    except Exception as exc:  # noqa: BLE001
        return False, f"Could not load or build the vector store: {exc}"


def main() -> None:
    load_environment()
    st.set_page_config(
        page_title="AI Data Science Tutor",
        page_icon="📊",
        layout="wide",
    )
    _init_session_state()

    st.title("📊 AI Data Science Tutor")
    st.caption("RAG-powered tutor for Python, Pandas, ML, SQL — grounded in your PDFs.")

    key_ok, key_msg = validate_api_key()
    if not key_ok:
        st.error(key_msg)
        st.stop()

    with st.sidebar:
        st.header("Knowledge base")
        interview_mode = st.toggle("Interview mode", value=False, help="More interview-style answers.")
        if st.button("Rebuild Knowledge Base", type="primary"):
            ok, msg = _rebuild_knowledge_base()
            if ok:
                st.success(msg)
                try:
                    from rag_pipeline import load_vectorstore_from_disk

                    st.session_state.vectorstore = load_vectorstore_from_disk()
                    st.session_state.vectorstore_ready = True
                    st.session_state.last_interview_mode = None  # force chain refresh
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Index saved but reload failed: {exc}")
            else:
                st.error(msg)

        st.divider()
        st.subheader("Upload PDF")
        uploaded = st.file_uploader("Add documents to data/", type=["pdf"], accept_multiple_files=True)
        if uploaded:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            saved = 0
            for f in uploaded:
                name = sanitize_filename(f.name)
                dest = DATA_DIR / name
                dest.write_bytes(f.getvalue())
                saved += 1
            st.success(f"Saved {saved} file(s) to data/. Click **Rebuild Knowledge Base** to index them.")

        st.divider()
        if st.button("Clear conversation"):
            _clear_chat()
            st.rerun()

        pdfs = list_pdf_paths()
        st.caption(f"PDFs in data/: **{len(pdfs)}**")

    # Boot vector store
    if not st.session_state.vectorstore_ready:
        with st.spinner("Loading knowledge base…"):
            ok, msg = _ensure_vectorstore()
        if not ok:
            st.warning(msg)
            st.stop()
        if msg:
            st.sidebar.success(msg)

    # Refresh chain when interview mode changes
    if (
        st.session_state.last_interview_mode != interview_mode
        or "rag_chain" not in st.session_state
    ):
        st.session_state.rag_chain = build_rag_chain(
            st.session_state.vectorstore,
            interview_mode=interview_mode,
        )
        st.session_state.last_interview_mode = interview_mode

    # Chat-style layout
    for msg in st.session_state.memory.chat_memory.messages:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    user_text = st.chat_input("Ask a data science question…")
    if user_text is not None:
        ok_q, err_q = validate_user_question(user_text)
        if not ok_q:
            st.error(err_q)
        else:
            with st.chat_message("user"):
                st.markdown(user_text)

            history = list(st.session_state.memory.chat_memory.messages)
            try:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        result = invoke_rag_chain(
                            st.session_state.rag_chain,
                            user_text,
                            history,
                        )
                    answer = result.get("answer", "Sorry, something went wrong.")
                    st.markdown(answer)
                st.session_state.memory.chat_memory.add_user_message(user_text)
                st.session_state.memory.chat_memory.add_ai_message(answer)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Request failed: {exc}")
                if get_openai_api_key():
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
