"""
Proper RAG Application 
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────
# Load API Key
# ─────────────────────────────────────────

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

if not os.environ["OPENAI_API_KEY"]:
    raise EnvironmentError("Missing OPENAI_API_KEY in .env")

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────

DATA_DIR = "./data"          # where my PDFs live
CHROMA_DIR = "./chroma_db"   # where embeddings are saved on disk
TOP_K = 5                    # retrieve top 5 most similar chunks
SIMILARITY_THRESHOLD = 0.6   # filter out irrelevant chunks

# ─────────────────────────────────────────
# LLM
# ─────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ─────────────────────────────────────────
# Build Vector Store
# ─────────────────────────────────────────
# PDFs → Load pages → Split into chunks → Embed each chunk → Save to Chroma

def build_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("Loading existing vector store...")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    print("Building vector store...")
# ─────────────────────────────────────────
#  Load PDFs
# ─────────────────────────────────────────
    
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()

    if not docs:
        raise ValueError("No PDFs found in ./data")

    print(f"Loaded {len(docs)} pages")

# ─────────────────────────────────────────
#  Split into chunks
# ─────────────────────────────────────────

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

# ─────────────────────────────────────────
#  Embed and store
# ─────────────────────────────────────────

    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    db.persist()

    return db

# ─────────────────────────────────────────
# RAG Pipeline 

  Question → Expand query → Retrieve chunks → Filter by score → Build context → LLM answers
# ─────────────────────────────────────────

def rag(query, vectorstore):
    print(f"\n{'='*60}")
    print(f"Question: {query}")
    print('='*60)

    # Query expansion (helps retrieval)
    expanded_query = f"{query} context document details explanation"

    #Retrieve top-K chunks
    print("\n[Step 1] Retrieving documents...")
    results = vectorstore.similarity_search_with_score(expanded_query, k=TOP_K)   

    # Debug scores
    for i, (doc, score) in enumerate(results):
        print(f"Chunk {i+1} score: {round(score, 3)}")

    # Filter irrelevant chunks
    filtered = [(doc, score) for doc, score in results if score < SIMILARITY_THRESHOLD]

    if not filtered:
        print("\n❌ No relevant context found.")
        return {
            "answer": "No relevant information found in the documents.",
            "sources": []
        }

    docs = [doc for doc, _ in filtered]

    print(f"\n[Step 2] Using {len(docs)} relevant chunks")

    # Combine context
    context = "\n\n".join([d.page_content for d in docs])

    # Debug preview
    for i, d in enumerate(docs):
        print(f"\n--- Chunk {i+1} ({d.metadata.get('source')}) ---")
        print(d.page_content[:200])

    # Generate answer
    print("\n[Step 3] Generating answer...")

    # Build the prompt
    prompt = f"""
    You MUST answer using ONLY the context below.
    If the answer is not present, say: "I don't know."

    Context:
    {context}

    Question: {query}

    Answer:
    """
  # Get answer
    answer = llm.invoke(prompt).content.strip()

    return {
        "answer": answer,
        "sources": list(set(d.metadata.get("source", "unknown") for d in docs))
    }

# ─────────────────────────────────────────
# Print Result
# ─────────────────────────────────────────

def print_result(result):
    print(f"\n{'-'*60}")
    print("ANSWER:\n")
    print(result["answer"])
    print(f"{'-'*60}")

    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f" - {s}")

    print(f"{'-'*60}\n")

# ─────────────────────────────────────────
# Run
# ─────────────────────────────────────────

if __name__ == "__main__":
    db = build_vectorstore()

    print("\n✅ RAG ready. Ask a question (type 'quit' to exit)\n")

    # The Main Loop
    while True:
        q = input("Question: ").strip()

        if q.lower() in ["quit", "exit"]:
            break

        if not q:
            continue

        result = rag(q, db)
        print_result(result)
