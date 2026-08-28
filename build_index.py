"""
build_index.py
----------------
One-time script: reads movies.csv + tags.csv (MovieLens ml-latest-small),
builds a text "document" per movie, embeds it with a sentence-transformer,
and saves a FAISS index + metadata to disk.

Run this once before starting the Streamlit app:
    python build_index.py

Dataset: download ml-latest-small.zip from
    https://grouplens.org/datasets/movielens/latest/
and unzip movies.csv (and tags.csv, optional) into ./data/
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

DATA_DIR = "data"
MOVIES_CSV = os.path.join(DATA_DIR, "movies.csv")
TAGS_CSV = os.path.join(DATA_DIR, "tags.csv")  # optional, improves text quality

INDEX_PATH = os.path.join(DATA_DIR, "movies.index")
META_PATH = os.path.join(DATA_DIR, "movies_meta.parquet")

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for this use case


def build_document_text(movies: pd.DataFrame, tags: pd.DataFrame | None) -> pd.Series:
    """Combine title + genres (+ tags, if available) into one text blob per movie."""
    genres_text = movies["genres"].str.replace("|", " ", regex=False)
    doc = movies["title"] + ". Genres: " + genres_text

    if tags is not None:
        # aggregate all user tags per movie into a single string
        tags_grouped = (
            tags.groupby("movieId")["tag"]
            .apply(lambda x: " ".join(x.astype(str)))
            .rename("tag_text")
        )
        movies = movies.merge(tags_grouped, on="movieId", how="left")
        doc = doc + ". Tags: " + movies["tag_text"].fillna("")

    return doc


def main():
    print("Loading data...")
    movies = pd.read_csv(MOVIES_CSV)

    tags = None
    if os.path.exists(TAGS_CSV):
        tags = pd.read_csv(TAGS_CSV)

    print(f"{len(movies)} movies loaded. Building document text...")
    movies["doc_text"] = build_document_text(movies, tags)

    print(f"Loading embedding model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding movie documents (this may take a minute)...")
    embeddings = model.encode(
        movies["doc_text"].tolist(),
        show_progress_bar=True,
        normalize_embeddings=True,  # so we can use inner product = cosine similarity
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine sim
    index.add(embeddings)

    os.makedirs(DATA_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    movies[["movieId", "title", "genres", "doc_text"]].to_parquet(META_PATH)

    print(f"Done. Index saved to {INDEX_PATH}, metadata saved to {META_PATH}")


if __name__ == "__main__":
    main()
