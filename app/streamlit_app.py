import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))
from recommender import Recommender

st.set_page_config(page_title="Semantic Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Semantic Movie Recommender")
st.caption(
    "RAG-style recommendation: embeds movies + your query into vector space, "
    "retrieves nearest neighbors with FAISS, and explains why each was picked."
)


@st.cache_resource
def load_recommender():
    return Recommender()


try:
    rec = load_recommender()
except Exception as e:
    st.error(
        "Could not load the index. Did you run `python build_index.py` first? "
        f"\n\nDetails: {e}"
    )
    st.stop()

mode = st.radio(
    "How do you want to search?",
    ["🔎 Describe what you want", "🎥 Pick a movie you like"],
    horizontal=True,
)

top_k = st.slider("Number of recommendations", min_value=3, max_value=20, value=10)

results = None

if mode == "🔎 Describe what you want":
    query = st.text_input(
        "Describe a movie / mood / vibe",
        placeholder="e.g. mind-bending sci-fi with an emotional twist",
    )
    if st.button("Get Recommendations", type="primary") and query.strip():
        with st.spinner("Searching..."):
            results = rec.search_by_text(query, top_k=top_k)

else:
    all_titles = rec.meta[["movieId", "title"]].sort_values("title")
    selected_title = st.selectbox("Pick a movie", all_titles["title"].tolist())
    if st.button("Find Similar Movies", type="primary"):
        movie_id = int(all_titles[all_titles["title"] == selected_title]["movieId"].iloc[0])
        with st.spinner("Searching..."):
            results = rec.search_by_movie(movie_id, top_k=top_k)

if results is not None:
    if results.empty:
        st.warning("No results found — try a different query.")
    else:
        st.subheader("Recommendations")
        for _, row in results.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{row['title']}**")
                    st.caption(row["genres"])
                    st.caption(f"💡 {row['why']}")
                with col2:
                    st.metric("Similarity", row["similarity"])

st.divider()
with st.expander("ℹ️ About this system"):
    st.markdown(
        """
        - **Approach**: Sentence-embedding based semantic retrieval (RAG-style, no generation LLM by default)
        - **Embedding model**: `all-MiniLM-L6-v2` (sentence-transformers)
        - **Vector index**: FAISS (`IndexFlatIP`, cosine similarity via normalized vectors)
        - **Dataset**: MovieLens `ml-latest-small`
        - **Explanation**: heuristic genre/keyword overlap (cost-free, always available).
          An optional LLM-based explanation mode can be swapped in — see README.
        """
    )
