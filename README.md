# AI Data Science Tutor Chatbot (RAG-based)

An advanced, portfolio-ready chatbot that explains data science topics (Python, Pandas, machine learning, SQL), answers from your uploaded PDFs using **Retrieval-Augmented Generation (RAG)**, and formats answers for learning or **interview-style** responses.

## Overview

The app combines **OpenAI** embeddings and chat models with **LangChain**, **FAISS** vector search, and **Streamlit** for a clean chat UI. Documents live under `data/`; the FAISS index is persisted under `vectorstore/` so you do not re-embed on every restart.

## Features

- **RAG pipeline**: PDF loading (`PyPDFLoader`), chunking, OpenAI embeddings, FAISS storage and retrieval, context-aware answers.
- **Tutor behavior**: Beginner-friendly tone with **Definition**, **Example**, and **Real-world application** sections.
- **Grounding rule**: If the retrieved context is insufficient, the model responds with: *"I don't have enough information from the provided documents."*
- **Chat memory**: **LangChain `ConversationBufferMemory`** keeps multi-turn context for follow-up questions.
- **Streamlit UI**: Chat layout, sidebar PDF upload, **Rebuild Knowledge Base**, **Interview mode** toggle, and **Clear conversation**.
- **Robustness**: Handles missing API key, empty `data/` (optional auto sample PDF via `fpdf2`), and invalid or empty user input.

## Tech Stack

| Layer        | Technology                                      |
|-------------|--------------------------------------------------|
| UI          | Streamlit                                        |
| Orchestration | LangChain 0.3.x (pinned below v1 for classic chain imports) |
| LLM         | OpenAI API (`gpt-4o-mini`)                       |
| Embeddings  | OpenAI (`text-embedding-3-small`)                |
| Vector DB   | FAISS (via `langchain-community`, `faiss-cpu`)   |
| PDF loading | `PyPDFLoader` (`pypdf`)                          |
| Config      | `python-dotenv` + `.env`                         |

## Project Structure

```text
chatbot-project/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # RAG: load/split/embed, FAISS, chains
├── utils.py            # Env, validation, sample PDF helper
├── data/               # Your PDFs (sample may be auto-created)
├── vectorstore/        # Saved FAISS index
├── requirements.txt
├── .env                # API key (do not commit real secrets)
└── README.md
```

## Setup

1. **Python 3.10+** recommended.

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure OpenAI:

   - Copy `.env` and set `OPENAI_API_KEY` to your key from [OpenAI API keys](https://platform.openai.com/api-keys).

5. Add PDFs to `data/` (or rely on the auto-generated sample if the folder is empty and `fpdf2` is installed).

## Run Locally

From the `chatbot-project` directory:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

**Workflow tips**

- After uploading new PDFs in the sidebar, click **Rebuild Knowledge Base** to refresh FAISS.
- Use **Interview mode** for tighter, interview-oriented answers with the same three-part structure.
- Use **Clear conversation** to reset LangChain memory without rebuilding the index.

## Screenshots

_Add your screenshots here after running the app (e.g. main chat view and sidebar with upload / rebuild)._

```text
[Screenshot 1: Main chat — placeholder]
[Screenshot 2: Sidebar — placeholder]
```

## Error Handling

- **Missing / placeholder API key**: The app stops early with a clear message; fix `.env` and reload.
- **Empty `data/`**: A small sample PDF may be created automatically if `fpdf2` is installed; otherwise add PDFs manually and rebuild.
- **Invalid input**: Empty or overly long questions are rejected with a short error in the UI.

## Future Improvements

- Hybrid search (e.g., BM25 + dense) and re-ranking for better retrieval.
- Source citations (page numbers / snippets) in the UI.
- Async ingestion and progress bars for large PDF batches.
- Optional local models (Ollama) for offline demos.
- Auth, usage limits, and deployment (Docker + cloud).

## License

Use freely for learning and portfolio purposes; ensure compliance with OpenAI and third-party terms for production use.
