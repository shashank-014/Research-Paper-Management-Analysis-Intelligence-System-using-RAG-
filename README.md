# Research Paper Management & Analysis Intelligence System

A production-style GenAI research assistant for ingesting academic papers, parsing them into structured sections, indexing them for semantic discovery, answering grounded questions with RAG, and surfacing citation and trend intelligence through a Streamlit interface.

## What This Project Does

This system helps researchers:
- ingest research PDFs
- parse section-level paper structure
- extract metadata and references
- build semantic search over papers and sections with FAISS
- generate grounded summaries and answers
- compare papers and methods
- build citation intelligence
- analyze research trends and emerging topics
- explore everything through a Streamlit app

## Core Features

### 1. Paper ingestion and parsing
- Loads PDFs with PyMuPDF and optional pdfplumber fallback
- Extracts page-wise text
- Detects academic sections such as abstract, introduction, methods, results, discussion, conclusion, references, related work, and limitations
- Builds structured `ResearchPaper` objects
- Supports direct multi-file upload from the Streamlit UI

### 2. Semantic indexing
- Section-aware chunking
- Embeddings with sentence-transformers
- FAISS vector indexing
- Semantic search with metadata filters

### 3. RAG research assistant
- Paper summarization
- Grounded question answering
- Cross-paper comparison
- Source-aware responses
- Uses `GROQ_API_KEY` from Streamlit secrets

### 4. Citation and trend intelligence
- Citation graph construction with NetworkX
- Influential paper metrics
- Keyword extraction
- Topic growth tracking
- Emerging topic detection
- Metadata lookup and related-work tools

### 5. Streamlit research interface
- Paper library dashboard
- Paper viewer
- Research chat assistant
- Cross-paper comparison
- Citation explorer
- Trend dashboard
- Multi-file PDF upload and refresh workflow

## Project Structure

```text
research_ai/
  analytics/
  config/
  indexing/
  ingestion/
  models/
  parsing/
  rag/
  ui/
  utils/
app.py
run_streamlit_app.py
requirements.txt
README.md
```

## Tech Stack
- Python
- Streamlit
- Pydantic
- PyMuPDF
- FAISS
- sentence-transformers
- Groq API
- NetworkX
- pandas
- numpy

## Setup

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Add Streamlit secrets

Create `.streamlit/secrets.toml` and add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

This key is needed for:
- paper summarization
- RAG question answering
- cross-paper comparison

It is not required for:
- parsing PDFs
- local indexing
- FAISS search
- citation analytics
- trend analytics

### 3. Run the app

```powershell
streamlit run app.py
```

## Uploading Papers

You do not need to commit PDFs into the repo just to use the system.

Inside the Streamlit app:
1. Open the sidebar
2. Use `Upload PDFs`
3. Select multiple PDF files
4. Click `Process Uploaded PDFs`

The app will:
- save uploads into `data/raw_pdfs`
- parse them into structured JSON in `data/processed`
- rebuild the FAISS index in `data/indices`

## Streamlit Deployment

Deploy this file:
- `app.py`

Why:
- `app.py` is the root deployment entrypoint
- it imports and runs the real UI from `research_ai/ui/app.py`
- this avoids confusion during Streamlit deployment

## Main Entry Points

### Streamlit UI
- `app.py`: root deployment entrypoint
- `research_ai/ui/app.py`: actual Streamlit app implementation

### Parsing
- `research_ai/parsing/paper_builder.py`

### Indexing
- `research_ai/indexing/index_builder.py`
- `research_ai/indexing/semantic_search.py`

### RAG
- `research_ai/rag/rag_pipeline.py`
- `research_ai/rag/summarizer.py`
- `research_ai/rag/comparison_engine.py`

### Analytics
- `research_ai/analytics/citation_graph.py`
- `research_ai/analytics/trend_analysis.py`
- `research_ai/analytics/mcp_tools.py`

## Important Notes
- LLM features are wired to Groq through `st.secrets["GROQ_API_KEY"]`.
- Crossref/OpenAlex lookup is scaffolded in the analytics tools and depends on network availability.
- Archived replaced files are stored in `DELETED_FILES/` per project rules.

## Current Status

Implemented:
- ingestion and parsing
- semantic indexing
- RAG summarization and QA
- citation analytics
- trend analytics
- Streamlit UI
- direct PDF upload workflow

Still worth hardening further:
- end-to-end runtime testing
- dependency validation in a clean environment
- parser accuracy on diverse academic PDF layouts
- richer external metadata enrichment
- background job support for large paper collections
