"""
RAG pipeline: load PDFs, chunk, embed, FAISS persistence, retrieval + conversational QA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import DATA_DIR, VECTORSTORE_DIR, list_pdf_paths

# FAISS on-disk index name (langchain default)
FAISS_INDEX_NAME = "faiss_index"


def get_embeddings() -> OpenAIEmbeddings:
    """OpenAI embedding model used for FAISS."""
    return OpenAIEmbeddings(model="text-embedding-3-small")


def get_llm() -> ChatOpenAI:
    """Chat model for retrieval rewriting and final answers."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


def load_pdf_documents(pdf_paths: list[Path]) -> list[Document]:
    """Load many PDFs with PyPDFLoader (one loader per file)."""
    all_docs: list[Document] = []
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        all_docs.extend(loader.load())
    return all_docs


def split_documents(documents: list[Document]) -> list[Document]:
    """Chunk documents for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def build_and_save_vectorstore(
    pdf_paths: list[Path],
    embeddings: OpenAIEmbeddings | None = None,
    persist_dir: Path | None = None,
) -> FAISS:
    """
    Load PDFs, split, embed, and persist FAISS to disk.
    """
    if not pdf_paths:
        raise ValueError("No PDF paths provided. Add PDFs to the data/ folder.")

    emb = embeddings or get_embeddings()
    raw_docs = load_pdf_documents(pdf_paths)
    if not raw_docs:
        raise ValueError("No text could be extracted from the provided PDFs.")

    chunks = split_documents(raw_docs)
    if not chunks:
        raise ValueError("Document splitting produced no chunks.")

    store = FAISS.from_documents(chunks, emb)
    out = persist_dir or VECTORSTORE_DIR
    out.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out), index_name=FAISS_INDEX_NAME)
    return store


def load_vectorstore_from_disk(
    embeddings: OpenAIEmbeddings | None = None,
    persist_dir: Path | None = None,
) -> FAISS:
    """Load a saved FAISS index."""
    emb = embeddings or get_embeddings()
    folder = persist_dir or VECTORSTORE_DIR
    return FAISS.load_local(
        str(folder),
        emb,
        index_name=FAISS_INDEX_NAME,
        allow_dangerous_deserialization=True,
    )


def vectorstore_exists(persist_dir: Path | None = None) -> bool:
    folder = persist_dir or VECTORSTORE_DIR
    pkl = folder / f"{FAISS_INDEX_NAME}.pkl"
    faiss = folder / f"{FAISS_INDEX_NAME}.faiss"
    return pkl.is_file() and faiss.is_file()


def get_or_create_vectorstore(
    embeddings: OpenAIEmbeddings | None = None,
    persist_dir: Path | None = None,
) -> tuple[FAISS, str]:
    """
    Load FAISS from disk if present; otherwise build from PDFs in data/.
    Returns (vectorstore, status_message).
    """
    emb = embeddings or get_embeddings()
    pdfs = list_pdf_paths(DATA_DIR)

    if vectorstore_exists(persist_dir):
        vs = load_vectorstore_from_disk(emb, persist_dir)
        return vs, "Loaded existing FAISS index from disk."

    if not pdfs:
        raise FileNotFoundError(
            "No PDFs found in the data/ folder. Upload PDFs or use the sample generator."
        )

    vs = build_and_save_vectorstore(pdfs, emb, persist_dir)
    return vs, f"Built new FAISS index from {len(pdfs)} PDF file(s)."


def build_rag_chain(vectorstore: FAISS, interview_mode: bool = False):
    """
    Conversational RAG chain with history-aware retrieval.
    Invoke with: {"input": str, "chat_history": list[BaseMessage]}
    """
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference "
        "context in the chat history, formulate a standalone question which can be "
        "understood without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    base_rules = (
        "You are a friendly data science tutor. Use simple, beginner-friendly language.\n"
        "Ground your answer in the context below when it is relevant. If the context does "
        "not contain enough information to answer reliably, reply exactly with:\n"
        "\"I don't have enough information from the provided documents.\"\n\n"
        "Otherwise, structure your answer with these headings (use the exact labels):\n"
        "Definition:\n"
        "Example:\n"
        "Real-world application:\n"
    )
    interview_extra = (
        "\nInterview mode is ON: answer as if in a technical interview — concise, "
        "clear structure, mention trade-offs or pitfalls when useful, and keep "
        "the same three headings.\n"
    )
    system_prompt = base_rules + (interview_extra if interview_mode else "") + "\n{context}"

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def invoke_rag_chain(chain: Any, user_input: str, chat_history: list) -> dict[str, Any]:
    """Thin wrapper around chain.invoke for consistent typing."""
    return chain.invoke({"input": user_input, "chat_history": chat_history})
