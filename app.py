# file: app.py

import streamlit as st
import os
import rag_app

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 RAG Document Assistant")

# ─────────────────────────────────────────
# Load Vector DB (FAST + SAFE)
# ─────────────────────────────────────────

@st.cache_resource(show_spinner=True)
def load_db():
    return rag_app.build_vectorstore()

# Show loading spinner (important UX)
with st.spinner("🔄 Loading documents (only first time)..."):
    db = load_db()

# Stop if DB failed
if db is None:
    st.error("Failed to load database.")
    st.stop()

# ─────────────────────────────────────────
# Sidebar (Documents)
# ─────────────────────────────────────────

with st.sidebar:
    st.header("📂 Documents")

    if os.path.exists("data"):
        docs = [f for f in os.listdir("data") if f.endswith(".pdf")]
    else:
        docs = []

    if docs:
        for doc in docs:
            st.write(f"📄 {doc}")
    else:
        st.warning("No PDFs found in /data")

    st.markdown("---")
    st.success(f"✅ {len(docs)} documents loaded")

    if st.button("🗑️ Clear chat history"):
        st.session_state.history = []
        st.rerun()

# ─────────────────────────────────────────
# Chat State
# ─────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────
# Input
# ─────────────────────────────────────────

query = st.text_input("Ask a question about your documents:")

col1, col2 = st.columns([1, 1])

with col1:
    ask = st.button("🔍 Search")

with col2:
    clear = st.button("🗑️ Clear Chat")

if clear:
    st.session_state.history = []
    st.rerun()

# ─────────────────────────────────────────
# Run RAG (FAST)
# ─────────────────────────────────────────

if ask and query.strip():

    with st.spinner("🔎 Searching documents..."):
        result = rag_app.rag(query, db)

    st.session_state.history.append({
        "q": query,
        "a": result["answer"],
        "sources": result.get("sources", [])
    })

    st.rerun()

elif ask and not query.strip():
    st.warning("Please enter a question")

# ─────────────────────────────────────────
# Display Chat
# ─────────────────────────────────────────

for chat in st.session_state.history:
    st.markdown(f"### 🧑 You:\n{chat['q']}")
    st.markdown(f"### 🤖 Answer:\n{chat['a']}")

    if chat["sources"]:
        st.markdown("**Sources:**")
        for s in chat["sources"]:
            st.write(f"- {s}")

    st.markdown("---")

# ─────────────────────────────────────────
# Empty State
# ─────────────────────────────────────────

if not st.session_state.history:
    st.info("Try: *What is Artificial Intelligence?*")