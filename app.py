import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()                             
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in .env")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
LLM = genai.GenerativeModel("gemini-2.0-flash")

# ──────────────────────────────────────────────────────────────
# 2) Import TreeHop retriever (local file)
# ──────────────────────────────────────────────────────────────
from corrected_treehop import (
    CorrectedTreeHopRetriever,
    enhanced_multihop_search,
    generate_answer,
)

# ---------- init retriever (cache) ----------
@st.cache_resource(show_spinner="Loading TreeHop retriever…")
def init_retriever():
    return CorrectedTreeHopRetriever(
        model_name="BAAI/bge-m3",
        passages_file="passages.jsonl",    
        embed_dim=1024, g_size=64, n_heads=3, mlp_size=64
    )

retriever = init_retriever()

# ──────────────────────────────────────────────────────────────
# 3) Streamlit UI
# ──────────────────────────────────────────────────────────────
st.title("TreeHop Multi-hop QA Demo")

query = st.text_input("Ask a multi-hop question")

# Toggle helper modules
col1, col2, col3 = st.columns(3)
use_rewrite  = col1.checkbox("Query Rewrite", value=True)
use_planner  = col1.checkbox("Planner Hint",  value=True)
use_rerank   = col2.checkbox("LLM Rerank",    value=True)
use_summary  = col2.checkbox("Passage Summary",value=True)
max_hops     = col3.slider("Max Hops", 1, 5, 3)
top_n        = col3.slider("Top-N / Hop", 1, 10, 5)

if st.button("Run TreeHop") and query.strip():
    with st.spinner("Running TreeHop…"):
        rewritten_q, summaries, passages = enhanced_multihop_search(
            retriever, LLM, query,
            n_hop=max_hops, top_n=top_n,
            use_rewrite = use_rewrite,
            use_planner = use_planner,
            use_rerank  = use_rerank,
            use_summary = use_summary
        )

    # ── Show retrieved passages ──
    st.subheader("Passages")
    if summaries:
        for s, p in zip(summaries, passages):
            st.markdown(f"**{p['title']}** – {s}")
    else:
        for p in passages:
            st.markdown(f"**{p['title']}**  \n{p['text'][:250]}…")

    # ── Get answer ──
    answer, _ = generate_answer(LLM, query, passages)
    st.subheader("Answer")
    st.success(answer)

    # Debug
    with st.expander("Debug info"):
        st.write("Rewritten query:" if use_rewrite else "Original query:", rewritten_q)
        st.json(passages[:3])
