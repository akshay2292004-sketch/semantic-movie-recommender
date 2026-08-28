"""
recommender.py
----------------
Core retrieval logic:
  1. Load FAISS index + movie metadata built by build_index.py
  2. Embed a user query (free text OR a selected movie's doc_text) with the
     same sentence-transformer model
  3. Retrieve top-K nearest movies by cosine similarity
  4. Generate a short "why recommended" explanation

Explanation generation has two modes:
  - "heuristic" (default, no API key needed): compares shared genres/tags
    between the query and each result -> always works, zero cost.
  - "llm" (optional): if an OPENAI_API_KEY (or similar) is set, sends the
    query + retrieved items to an LLM to write a natural explanation.
    This is the classic RAG pattern: Retrieve -> Augment prompt -> Generate.

Keeping heuristic mode as default means the deployed app never breaks or
costs money even if no API key is configured — important for a public demo.
"""

import os
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INDEX_PATH = os.path.join(DATA_DIR, "movies.index")
META_PATH = os.path.join(DATA_DIR, "movies_meta.parquet")
MODEL_NAME = "all-MiniLM-L6-v2"

# --- Optional LLM explanation mode -----------------------------------------
# If ANTHROPIC_API_KEY (or OPENAI_API_KEY) is set in the environment, the app
# will use it to generate a fluent one-line "why recommended" explanation
# instead of the heuristic genre-overlap text. This is the classic RAG
# "Generate" step: retrieved items + query are fed to an LLM as context.
# If no key is set, the app silently falls back to the heuristic — the
# deployed demo never breaks or costs money by default.
LLM_PROVIDER = None
if os.environ.get("ANTHROPIC_API_KEY"):
    LLM_PROVIDER = "anthropic"
elif os.environ.get("OPENAI_API_KEY"):
    LLM_PROVIDER = "openai"


def _llm_explain_batch(query_text: str, candidates: list[dict]) -> list[str]:
    """
    Given the query and a list of {title, genres} candidates, ask an LLM to
    write one short explanation per candidate. Returns a list of strings in
    the same order as `candidates`. Falls back to None entries on any error
    so the caller can use the heuristic instead.
    """
    prompt = (
        "You are helping explain movie recommendations. The user's query/reference is:\n"
        f'"{query_text}"\n\n'
        "For each candidate movie below, write ONE short sentence (under 15 words) "
        "explaining why it might match the query. Respond ONLY with a JSON array of "
        "strings, in the same order as the candidates, no other text.\n\n"
        "Candidates:\n"
        + "\n".join(f"{i+1}. {c['title']} ({c['genres']})" for i, c in enumerate(candidates))
    )

    try:
        if LLM_PROVIDER == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
        elif LLM_PROVIDER == "openai":
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            text = resp.choices[0].message.content
        else:
            return [None] * len(candidates)

        text = text.strip().strip("```").strip("json").strip()
        explanations = json.loads(text)
        if len(explanations) == len(candidates):
            return explanations
    except Exception:
        pass

    return [None] * len(candidates)


class Recommender:
    def __init__(self, use_llm_explanations: bool = True):
        self.index = faiss.read_index(INDEX_PATH)
        self.meta = pd.read_parquet(META_PATH)
        self.model = SentenceTransformer(MODEL_NAME)
        self.use_llm_explanations = use_llm_explanations and LLM_PROVIDER is not None

    def _embed(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True)
        return np.asarray(vec, dtype="float32")

    def search_by_text(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """Free-text semantic search, e.g. 'funny heist movie with a twist'."""
        query_vec = self._embed(query)
        scores, idxs = self.index.search(query_vec, top_k)
        return self._format_results(idxs[0], scores[0], query)

    def search_by_movie(self, movie_id: int, top_k: int = 10) -> pd.DataFrame:
        """'More like this' search using a movie already in the catalog."""
        row = self.meta[self.meta["movieId"] == movie_id].iloc[0]
        query_vec = self._embed(row["doc_text"])
        scores, idxs = self.index.search(query_vec, top_k + 1)  # +1 to drop self-match
        results = self._format_results(idxs[0], scores[0], row["doc_text"])
        return results[results["movieId"] != movie_id].head(top_k)

    def _format_results(self, idxs, scores, query_text: str) -> pd.DataFrame:
        rows = []
        candidates = []
        for idx, score in zip(idxs, scores):
            if idx == -1:
                continue
            item = self.meta.iloc[idx]
            rows.append({
                "movieId": item["movieId"],
                "title": item["title"],
                "genres": item["genres"],
                "similarity": round(float(score), 3),
                "why": self._explain_heuristic(query_text, item["genres"]),  # default/fallback
            })
            candidates.append({"title": item["title"], "genres": item["genres"]})

        if self.use_llm_explanations and rows:
            llm_explanations = _llm_explain_batch(query_text, candidates)
            for row, llm_why in zip(rows, llm_explanations):
                if llm_why:
                    row["why"] = llm_why

        return pd.DataFrame(rows)

    @staticmethod
    def _explain_heuristic(query_text: str, genres: str) -> str:
        """Simple, cost-free explanation: overlap between query words and genres."""
        query_words = set(query_text.lower().replace(".", " ").replace(",", " ").split())
        genre_words = set(genres.lower().replace("|", " ").split())
        overlap = query_words & genre_words
        if overlap:
            return f"Shares themes/genres: {', '.join(sorted(overlap))}"
        return "Semantically similar based on plot/genre embedding"
