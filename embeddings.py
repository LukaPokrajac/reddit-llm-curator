"""Post embeddings via a local OpenAI-compatible /v1/embeddings endpoint.

Embeddings are lossy fingerprints of "what is this post about" — they let us
find semantically similar posts with pgvector's cosine distance, so the
curator can see how related earlier posts were judged. The text itself still
lives in the posts table; vectors only help retrieval.

Two granularities: every post gets a title+body vector (posts.embedding),
and link posts additionally get chunk vectors of the extracted article or
video transcript (post_chunks) — so similarity works on what the linked page
actually says, not just its headline.

Default server is Ollama with nomic-embed-text (768 dims, matches the
vector(768) column in schema.sql). fetch_posts.py embeds new posts inline
and curate_readings.py chunk-embeds link text as it fetches it; run this
file directly to backfill whatever is missing a vector (e.g. fetched while
the embedding server was down, or link posts from before post_chunks existed):

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

CHUNK_SIZE = 1500   # chars per link-text chunk, comfortably inside nomic's window
CHUNK_MAX = 8       # embed at most this much of one article/transcript

# Chunk vectors of a post's link text (extracted article or video transcript),
# so retrieval matches on what the linked page says, not just the headline.
# Vectors only — the text itself stays in posts.link_text. Like posts.embedding
# these are long-term memory: prune.py clears old link_text but not these.
CHUNKS_DDL = """
    CREATE TABLE IF NOT EXISTS post_chunks (
        post_id   text NOT NULL REFERENCES posts (id),
        idx       smallint NOT NULL,
        embedding vector(768) NOT NULL,
        PRIMARY KEY (post_id, idx)
    )"""


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


def chunk_text(text: str, size: int = CHUNK_SIZE,
               max_chunks: int = CHUNK_MAX) -> list[str]:
    """Split link text into embedding-sized chunks: whole paragraphs are
    packed together up to `size`; an oversized paragraph (transcripts are one
    endless line) is cut at word boundaries. Pure, see test_curate.py."""
    pieces = []
    for para in text.split("\n\n"):
        para = " ".join(para.split())
        while len(para) > size:
            cut = para.rfind(" ", size // 2, size)
            cut = cut if cut != -1 else size
            pieces.append(para[:cut])
            para = para[cut:].lstrip()
        if para:
            pieces.append(para)
    chunks, buf = [], ""
    for piece in pieces:
        if buf and len(buf) + len(piece) + 1 > size:
            chunks.append(buf)
            if len(chunks) == max_chunks:
                return chunks
            buf = piece
        else:
            buf = f"{buf}\n{piece}" if buf else piece
    if buf:
        chunks.append(buf)
    return chunks


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

            # Chunk vectors for link posts fetched before post_chunks existed
            # (or while the embedding server was down).
            cur.execute(CHUNKS_DDL)
            cur.execute("""SELECT id, link_text FROM posts
                           WHERE link_text <> '' AND NOT EXISTS
                               (SELECT 1 FROM post_chunks c WHERE c.post_id = posts.id)
                           ORDER BY created_utc DESC""")
            todo = cur.fetchall()
            print(f"{len(todo)} link texts to chunk-embed")
            for i, (post_id, link_text) in enumerate(todo, 1):
                cur.executemany(
                    "INSERT INTO post_chunks (post_id, idx, embedding) "
                    "VALUES (%s, %s, %s::vector)",
                    [(post_id, n, vec_literal(embed(chunk)))
                     for n, chunk in enumerate(chunk_text(link_text))])
                conn.commit()
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}")


if __name__ == "__main__":
    backfill()
