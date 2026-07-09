"""Post embeddings via a local OpenAI-compatible /v1/embeddings endpoint.

Embeddings are lossy fingerprints of "what is this post about" — they let us
find semantically similar posts with pgvector's cosine distance, so the
curator can see how related earlier posts were judged. The text itself still
lives in the posts table; vectors only help retrieval.

Default server is Ollama with nomic-embed-text (768 dims, matches the
vector(768) column in schema.sql). fetch_posts.py embeds new posts inline;
run this file directly to backfill any posts that are missing a vector
(e.g. fetched while the embedding server was down):

    .venv/bin/python embeddings.py
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

EMBEDDINGS_URL = os.environ.get("EMBEDDINGS_URL",
                                "http://localhost:11434/v1/embeddings")
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "nomic-embed-text")

EMBED_TEXT_CAP = 4000  # chars; nomic truncates around 2K tokens anyway


def embed(text: str, timeout: int = 30) -> list[float]:
    """Embed one text. Raises on any server/network problem — callers that
    can live without a vector catch and store NULL (backfill picks it up)."""
    resp = requests.post(
        EMBEDDINGS_URL,
        json={"model": EMBEDDINGS_MODEL, "input": text[:EMBED_TEXT_CAP]},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def post_text(title: str, selftext: str) -> str:
    """What we embed for a post: title carries most of the signal, body adds
    context. Kept in one place so fetch and backfill stay consistent."""
    return f"{title}\n\n{selftext}"


def vec_literal(vector: list[float]) -> str:
    """pgvector accepts its text form ('[0.1,0.2,...]') cast with ::vector,
    which saves a client-side adapter dependency."""
    return json.dumps(vector)


def backfill() -> None:
    import psycopg

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, title, selftext FROM posts "
                        "WHERE embedding IS NULL ORDER BY created_utc DESC")
            todo = cur.fetchall()
            print(f"{len(todo)} posts to embed")
            for i, (post_id, title, selftext) in enumerate(todo, 1):
                cur.execute("UPDATE posts SET embedding = %s::vector WHERE id = %s",
                            (vec_literal(embed(post_text(title, selftext))), post_id))
                conn.commit()
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}")


if __name__ == "__main__":
    backfill()
