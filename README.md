# Semantic Movie Recommender (RAG-style)

## 1. Problem Statement
Users often can't articulate what they want in exact keywords or genre filters —
they describe a *vibe* ("something like Inception but funnier"). Traditional
keyword/genre filters fail here. This project builds a recommender that
understands semantic meaning, not just exact tags.

## 2. Use Case & Motivation
Movie discovery, inspired by Netflix's "More Like This" and semantic search
features on modern streaming platforms. Chosen because MovieLens is a clean,
well-documented, small dataset — ideal for demonstrating the approach without
infrastructure overhead.

## 3. Approach
This is a **retrieval-augmented** recommender:

1. **Index time** (`build_index.py`): every movie's title + genres (+ user tags)
   are combined into a text "document" and embedded with a sentence-transformer
   (`all-MiniLM-L6-v2`). Embeddings are stored in a FAISS vector index.
2. **Query time** (`streamlit_app.py` + `recommender.py`): the user's free-text
   query, or a selected movie's own text, is embedded with the same model.
   FAISS returns the nearest movies by cosine similarity.
3. **Explanation**: each result includes a short "why" — currently a
   cost-free heuristic (shared genre/keyword overlap between query and result).
   This is the natural place to plug in an LLM call (see "Future Improvements")
   to generate a fluent explanation — the classic RAG pattern of
   Retrieve → Augment prompt → Generate — without changing the retrieval logic.

## 4. System Architecture
```
 ┌────────────────┐     one-time      ┌──────────────────┐
 │ movies.csv      │ ───────────────► │ build_index.py    │
 │ tags.csv        │                  │ (embed + FAISS)   │
 └────────────────┘                   └─────────┬─────────┘
                                                  │ movies.index
                                                  │ movies_meta.parquet
                                                  ▼
                              ┌───────────────────────────────┐
 User query/selection ──────►│ Streamlit app (recommender.py) │──► Ranked list + "why"
                              └───────────────────────────────┘
```

## 5. Recommendation Methodology
Content-based, embedding-driven nearest-neighbor retrieval (semantic search),
not collaborative filtering. No user-user or user-item interaction data is
used for ranking — only item content (title, genres, tags).

## 6. Dataset
[MovieLens `ml-latest-small`](https://grouplens.org/datasets/movielens/latest/)
(~9,000 movies, ~100,000 ratings). `movies.csv` is required; `tags.csv` is
optional but improves description quality when present.

**Setup:** download the zip, place `movies.csv` (and `tags.csv`, if desired)
into `./data/`, then run `python build_index.py` once.

## 7. Technologies Used
- Python, pandas, numpy
- `sentence-transformers` (`all-MiniLM-L6-v2`) for embeddings
- FAISS (`faiss-cpu`) for vector similarity search
- Streamlit for the UI, deployed on Streamlit Community Cloud

## 8. Assumptions Made
- Genre/tag/title text is a reasonable proxy for a movie's semantic content
  (no plot summaries were available in the base dataset).
- Users are fine with content-based results that ignore their personal rating
  history (no collaborative signal in this version).

## 9. Key Design Decisions
- Chose a small, fast embedding model (`all-MiniLM-L6-v2`) so the whole
  pipeline runs on CPU with no GPU/API dependency — important for a free,
  always-on public deployment.
- Defaulted the "why" explanation to a heuristic (not an LLM call) so the
  deployed demo never breaks or incurs cost if no API key is configured.
- Used FAISS `IndexFlatIP` with normalized vectors (exact cosine similarity)
  rather than an approximate index, since the dataset is small enough that
  exact search is still fast.

## 10. Evaluation Methodology
Because this is content-based with no held-out user interaction data for
this dataset subset, standard ranking metrics (Precision@K, NDCG) are
approximated using **genre overlap as a relevance proxy**: a recommended
movie is considered "relevant" if it shares at least one genre with the
query movie. This is reported alongside qualitative test cases below.

Metrics tracked:
- **Precision@10** (genre-overlap proxy)
- **Latency** (embedding + FAISS search time per query)
- Qualitative **success/failure** case review (below)

## 11. Test Cases

### Successful Scenarios
- Query: *"mind-bending sci-fi thriller"* → returns Inception, The Matrix,
  Shutter Island-type results — genres and themes align well.
- "More like this" on a well-tagged, well-known movie → close genre/tag
  matches with high similarity scores.

### Failure Scenarios
- Query relying on **plot details not present in the dataset** (e.g. "a movie
  where the twist is that the narrator is unreliable") — the model has only
  genres/tags/title, not plot summaries, so it can't reason about plot twists
  it was never given.
- **Sparse-tag movies**: obscure films with no user tags reduce to just
  title + genre, giving weaker, more generic matches.
- **Ambiguous one-word queries** (e.g. "sad") retrieve a broad, noisy mix
  since embedding similarity alone can't disambiguate intent.

## 12. Known Limitations
- No collaborative filtering — doesn't learn from what similar users liked.
- No plot summaries in the embedding text — semantic matches are shallower
  than they would be with full synopses.
- Explanation text is heuristic, not a fluent LLM-generated rationale.
- Cold-start for brand-new movies with no tags still works (title + genre is
  always available) but with lower-quality matches.

## 13. Future Improvements
- Add a plot-summary field (e.g. from TMDB API) to enrich embeddings.
- Add collaborative filtering (matrix factorization) and blend it with the
  semantic score for hybrid recommendations.
- Personalization: store a user's liked movies and average their embeddings
  as an implicit query vector.
- Richer generation: pass full plot context (once available) to the LLM
  explanation step for deeper reasoning, not just genre/title.

## Optional: LLM-Generated Explanations (full RAG "Generate" step)
By default, the "why recommended" text is a free heuristic (genre/keyword
overlap) so the deployed app always works with zero cost or setup.

To enable fluent, LLM-written explanations instead — the classic RAG
pattern of Retrieve → Augment prompt → Generate — set **one** environment
variable before running the app:

```bash
export ANTHROPIC_API_KEY=your_key_here     # or
export OPENAI_API_KEY=your_key_here
```

`recommender.py` detects whichever key is present and calls that provider
to generate one short explanation per recommended movie, using the query
and each candidate's title/genres as context. If the call fails for any
reason (no key, rate limit, network issue), it silently falls back to the
heuristic explanation for that item — the app never crashes because of it.

## How to Run Locally
```bash
pip install -r requirements.txt
# place movies.csv (+ optional tags.csv) into ./data/
python build_index.py
streamlit run app/streamlit_app.py
```

## Deployment
Deployed via [Streamlit Community Cloud](https://streamlit.io/cloud):
point it at this repo, set the main file to `app/streamlit_app.py`, and
ensure `data/movies.index` + `data/movies_meta.parquet` are committed
(or regenerate them via `build_index.py` in a startup script).

## Bonus: Comparison to Netflix's "More Like This"
- **Similarities**: both surface content-similar items and explain relevance.
- **Differences**: Netflix blends collaborative signals (millions of users'
  watch history) and rich metadata (cast, director, visual features); this
  project uses only title/genre/tags with no user-behavior data.
- **Current limitations**: no personalization, no plot understanding, no
  visual/audio features.
- **What I'd build next**: ingest plot summaries, add a lightweight
  collaborative filtering layer, and personalize via a user's liked-movie
  embedding average.
