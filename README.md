# Deakin College chatbot

Streamlit app that answers Deakin College student questions from a hybrid search index. It expands the query, retrieves passages from Qdrant, then asks Gemini to answer using only that context.

The first screen is a disclaimer. After you agree, you get a chat with thumbs feedback. Each turn is written to Supabase for later review.

## Run it

You need Python 3.11+ and the keys listed in `.env.example`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env`, then:

```bash
streamlit run main.py
```

The app opens on [http://localhost:8501](http://localhost:8501).

In GitHub Codespaces or a VS Code Dev Container, `pip` and Streamlit start from `.devcontainer/devcontainer.json`. Put the same keys in Codespace secrets or a local `.env`.

## Environment

| Variable | Used for |
| --- | --- |
| `GOOGLE_API_KEY` | Gemini 2.0 Flash for query rewrite and the final answer |
| `VOYAGE_API_KEY` | Voyage 3 dense embeddings |
| `QDRANT_HOST` | Qdrant URL |
| `QDRANT_API_KEY` | Qdrant auth |
| `SUPABASE_URL` | Logging and feedback |
| `SUPABASE_KEY` | Supabase service or anon key with access to `DC-analysis` |

Do not commit `.env`. The Qdrant collection name in code is `hybrid-search-splade` (Voyage dense vectors plus Splade sparse vectors, fused with RRF). The Supabase table is `DC-analysis`.

## What happens on a question

1. Gemini rewrites the student question so retrieval has more to match.
2. Voyage and Splade embed that rewrite. Qdrant returns the top passages.
3. Gemini answers from those passages only. `<link>` tags in the source text become markdown links.
4. The turn is stored in `DC-analysis`. Thumbs up or down updates the same row.

If the retrieved context does not cover the question, the prompt tells the model to say so instead of guessing.

## Layout

```
main.py                 Streamlit UI and RAG calls
assets/deakin-college.png
requirements.txt
.env.example
.devcontainer/          Codespaces / Dev Container
```

## Notes

This is a trial chatbot. Answers can be wrong. The disclaimer in the app is part of that. Retrieval depends on the Qdrant collection already being loaded. This repo does not build or refresh that index.

Speaker-style chat labels are not a record of who wrote the original student question. They only show the turn in this UI.
