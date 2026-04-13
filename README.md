

# Self-RAG App — Lennart Intern Task

A Self-RAG pipeline built with LangChain. Based on the self_rag.ipynb reference notebook.

## How it works

Unlike basic RAG which always searches, Self-RAG is smarter — it asks itself questions at every step:

```
Your question
      │
      ▼
[Step 1] Do I even need to search?  ──No──▶  Answer directly
      │ Yes
      ▼
[Step 2] Retrieve top-3 chunks from documents
      │
      ▼
[Step 3] Is each chunk actually relevant?
      │
      ▼
[Step 4] Generate an answer per relevant chunk
      │
      ▼
[Step 5] Hallucination check — Fully / Partially / No support
      │
      ▼
[Step 6] Utility score — 1 to 5
      │
      ▼
[Step 7] Pick the best answer
```

## Setup

1. Install dependencies:
   pip install -r requirements.txt

2. Create your .env file:
   OPENAI_API_KEY=sk-your-key-here

3. Add your documents to the ./data/ folder (PDF, txt, etc.)

4. Run:
   python rag_app.py

## File structure

```
rag-app/
├── rag_app.py        <- the whole app
├── requirements.txt  <- pip packages
├── .env              <- your API key (never commit this!)
├── .gitignore        <- hides .env and chroma_db from GitHub
├── README.md         <- this file
├── data/             <- put your documents here
└── chroma_db/        <- auto-created vector store
```

