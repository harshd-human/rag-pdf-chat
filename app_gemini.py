"""Chat with your PDFs / study notes: RAG app using a FREE Gemini API key.

Run:  streamlit run app_gemini.py
Needs: GEMINI_API_KEY set in your environment (get one at aistudio.google.com).
"""
import os

import chromadb
import streamlit as st
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
try:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass 


CHUNK_SIZE = 800    # characters per chunk
OVERLAP = 150       # characters shared between neighbouring chunks
TOP_K = 4           # chunks retrieved per question
# Free-tier models change often: check AI Studio, or override with GEMINI_MODEL.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

SYSTEM_PROMPT = (
    "You are a study assistant. Answer the question using ONLY the provided "
    "context excerpts. Cite sources like [file.pdf, p.3]. If the context does "
    "not contain the answer, say you couldn't find it in the uploaded notes "
    "instead of guessing."
)


@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_collection():
    client = chromadb.Client()  # in-memory; use PersistentClient(path="db") to save
    return client.get_or_create_collection("notes", metadata={"hnsw:space": "cosine"})


@st.cache_resource
def get_llm():
    return genai.Client()  # reads GEMINI_API_KEY from the environment


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    text = " ".join(text.split())
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def ingest(files):
    """Read each PDF page by page (keeps page numbers), chunk, embed, store."""
    embedder, col = get_embedder(), get_collection()
    total = 0
    for f in files:
        reader = PdfReader(f)
        ids, docs, metas = [], [], []
        for page_no, page in enumerate(reader.pages, start=1):
            for i, chunk in enumerate(chunk_text(page.extract_text() or "")):
                ids.append(f"{f.name}-p{page_no}-{i}")
                docs.append(chunk)
                metas.append({"source": f.name, "page": page_no})
        if docs:
            embeddings = embedder.encode(docs).tolist()
            col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
            total += len(docs)
    return total


def retrieve(question):
    col = get_collection()
    if col.count() == 0:
        return []
    q_emb = get_embedder().encode([question]).tolist()
    res = col.query(query_embeddings=q_emb, n_results=min(TOP_K, col.count()))
    return list(zip(res["documents"][0], res["metadatas"][0]))


def answer(question, history):
    hits = retrieve(question)
    context = "\n\n".join(f"[{m['source']}, p.{m['page']}]\n{d}" for d, m in hits)

    # Gemini uses the role "model" instead of "assistant"
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user",
         "parts": [{"text": m["content"]}]}
        for m in history[-6:]
    ]
    contents.append({
        "role": "user",
        "parts": [{"text": f"Context:\n{context}\n\nQuestion: {question}"}],
    })

    resp = get_llm().models.generate_content(
        model=MODEL,
        contents=contents,
        config={"system_instruction": SYSTEM_PROMPT},
    )
    return resp.text, hits


# ---------------- UI ----------------
st.set_page_config(page_title="Chat with your PDFs", page_icon="📚")
st.title("📚 Chat with your PDFs")
st.caption(f"Model: {MODEL}")

with st.sidebar:
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Process documents") and uploads:
        with st.spinner("Chunking and embedding..."):
            n = ingest(uploads)
        st.success(f"Indexed {n} chunks from {len(uploads)} file(s).")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if q := st.chat_input("Ask something about your notes"):
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                text, hits = answer(q, st.session_state.history)
            st.markdown(text)
            with st.expander("Sources used"):
                for d, m in hits:
                    st.caption(f"{m['source']}, page {m['page']}")
                    st.write(d)
            st.session_state.history += [
                {"role": "user", "content": q},
                {"role": "assistant", "content": text},
            ]
        except Exception as e:
            st.error(f"Gemini request failed: {e}")
