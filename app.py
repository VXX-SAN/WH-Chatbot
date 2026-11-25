import io
import os
import re
import gc
import json
import csv
import tempfile
import streamlit as st
from pypdf import PdfReader
from docx import Document as DocxDocument
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from supabase import create_client
from fastembed import TextEmbedding
from groq import Groq

# --- 1. Configuration & Secrets ---
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", os.environ.get("ADMIN_PASSWORD", "")).strip()
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", "")).strip()
RAW_SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL", "")).strip()
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY", "")).strip()
SUPABASE_URL = RAW_SUPABASE_URL.split("/rest/v1")[0].rstrip("/")

# Customize these for each new website/project:
DEFAULT_LLM = "openai/gpt-oss-120b"
# Use "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" for Arabic + English
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SYSTEM_INSTRUCTIONS = (
    "You are a helpful AI assistant. Answer the user's question using the RETRIEVED CONTEXT below. "
    "Cite the document name when referencing facts. If the context does not contain the answer, "
    "clearly say so before answering from general knowledge. Reply in the same language the user uses."
)

@st.cache_resource
def init_clients(groq_key: str, sb_url: str, sb_key: str, embed_name: str):
    return Groq(api_key=groq_key), create_client(sb_url, sb_key), TextEmbedding(model_name=embed_name)

groq_client, supabase, embedder = init_clients(GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY, EMBED_MODEL)

# --- 2. File Parsing & Memory-Safe Ingestion ---
def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith(".pdf"):
        return "\n".join([p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages])
    elif name.endswith(".epub"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp:
            tmp.write(raw)
            path = tmp.name
        try:
            book = epub.read_epub(path)
            return "\n\n".join(
                BeautifulSoup(item.get_content(), "html.parser").get_text(separator="\n", strip=True)
                for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
            )
        finally:
            if os.path.exists(path):
                os.remove(path)
    elif name.endswith(".docx"):
        return "\n".join([p.text for p in DocxDocument(io.BytesIO(raw)).paragraphs if p.text.strip()])
    elif name.endswith(".csv"):
        return "\n".join([", ".join(r) for r in csv.reader(io.StringIO(raw.decode("utf-8", errors="replace"))) if any(r)])
    elif name.endswith(".json"):
        return json.dumps(json.loads(raw.decode("utf-8", errors="replace")), ensure_ascii=False, indent=2)
    return raw.decode("utf-8", errors="replace")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > 1_000_000:
        chunk_size, overlap = 2200, 200
    chunks, start = [], 0
    while start < len(text):
        c = text[start : start + chunk_size].strip()
        if len(c) > 30:
            chunks.append(c)
        start += chunk_size - overlap
    return chunks

def get_embedding(text: str) -> list[float]:
    return list(embedder.embed([text]))[0].tolist()

def ingest_document(text: str, source_name: str) -> int:
    chunks = chunk_text(text)
    if not chunks:
        return 0
    supabase.table("documents").delete().eq("metadata->>source", source_name).execute()
    prog = st.progress(0, text=f"Ingesting {len(chunks):,} chunks...")
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embs = list(embedder.embed(batch))
        rows = [
            {"content": c, "metadata": {"source": source_name, "chunk": i + j}, "embedding": e.tolist()}
            for j, (c, e) in enumerate(zip(batch, embs))
        ]
        supabase.table("documents").insert(rows).execute()
        del embs, rows
        gc.collect()
        done = min(i + batch_size, len(chunks))
        prog.progress(done / len(chunks), text=f"Stored {done:,} / {len(chunks):,} chunks...")
    prog.empty()
    return len(chunks)

def get_sources() -> list[dict]:
    try:
        return supabase.rpc("get_knowledge_sources").execute().data or []
    except Exception:
        return []

def retrieve_context(query: str, sources: list[str], top_k: int = 4) -> list[dict]:
    if not sources:
        return []
    res = supabase.rpc("match_documents", {
        "query_embedding": get_embedding(query),
        "match_threshold": 0.28,
        "match_count": top_k,
        "filter_sources": sources,
        "query_text": query
    }).execute()
    return res.data or []

# --- 3. Sidebar: Password-Protected Admin Panel ---
stored_sources = get_sources()
all_source_names = [s["source"] for s in stored_sources]

with st.sidebar:
    st.header("⚙️ Admin Panel")
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = (ADMIN_PASSWORD == "")

    if not st.session_state.is_admin:
        pwd = st.text_input("Admin Password:", type="password")
        if st.button("Unlock Admin"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.is_admin = True
                st.rerun()
            else:
                st.error("Invalid password.")
    else:
        st.caption(f"🟢 Connected — {sum(int(s['chunk_count']) for s in stored_sources):,} total chunks")
        selected_sources = st.multiselect("Active Sources:", options=all_source_names, default=all_source_names)

        uploaded_files = st.file_uploader(
            "Upload Sources (PDF, EPUB, DOCX, TXT, MD, CSV, JSON)",
            type=["pdf", "epub", "docx", "txt", "md", "csv", "json"],
            accept_multiple_files=True
        )
        if st.button("Ingest Files", use_container_width=True) and uploaded_files:
            for uf in uploaded_files:
                count = ingest_document(extract_text(uf), uf.name)
                st.success(f"Stored {count:,} chunks from {uf.name}")
            st.rerun()

        if all_source_names:
            to_del = st.selectbox("Delete a source:", options=all_source_names)
            if st.button("Remove Source"):
                supabase.table("documents").delete().eq("metadata->>source", to_del).execute()
                st.rerun()

active_sources = selected_sources if st.session_state.get("is_admin") else all_source_names

# --- 4. Public Chat Interface (Isolated per Visitor) ---
st.title("🤖 AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            matches = retrieve_context(prompt, active_sources, top_k=4)
            context_block = "\n\n---\n\n".join(
                f"Document: {m.get('metadata', {}).get('source', 'unknown')}\n{m.get('content', '')}"
                for m in matches
            )[:6500]

            api_messages = [
                {"role": "system", "content": f"{SYSTEM_INSTRUCTIONS}\n\nRETRIEVED CONTEXT:\n{context_block or 'None'}"}
            ] + st.session_state.messages[-6:]

            kwargs = {
                "model": DEFAULT_LLM,
                "messages": api_messages,
                "temperature": 0.2,
                "max_tokens": 800,
                "stream": True
            }
            if "qwen" in DEFAULT_LLM.lower():
                kwargs["reasoning_effort"] = "none"

            stream = groq_client.chat.completions.create(**kwargs)
            reply = st.write_stream(chunk.choices[0].delta.content or "" for chunk in stream)

            if matches:
                with st.expander("📄 Sources Used"):
                    for m in matches:
                        st.caption(f"**{m.get('metadata', {}).get('source')}** (Score: {m['similarity']:.2f})")
                        st.write(m["content"])

            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"Error: {e}")