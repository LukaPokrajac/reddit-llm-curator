"""Overnight reading curator: turn r/singularity posts into personal reading material.

For each post in Postgres (newest first), the local Qwen model first decides
SIGNAL or SKIP for Luka specifically, and for SIGNAL posts writes a short
explainer piece grounded in the post + its comments — explaining unfamiliar
concepts, connecting to his projects and to earlier pieces from the same run.

Results land in a `readings` table (restartable: already-processed posts are
skipped) and SIGNAL articles are appended to readings/digest-<date>.md for
morning reading. Progress goes to curation.log.

Usage:
    .venv/bin/python curate_readings.py              # full run
    .venv/bin/python curate_readings.py --limit 2    # smoke test
"""

import argparse
import os
import sys
import time
from datetime import date, datetime, timezone

import psycopg
import requests
from dotenv import load_dotenv

from embeddings import embed, post_text, vec_literal
from fetch_posts import fetch_comments_from_arctic_shift

load_dotenv()

LOCALAI = os.environ.get("LOCALAI_URL", "http://localhost:8081/v1/chat/completions")
MODEL = os.environ.get("CURATOR_MODEL", "qwen3.6-35b-a3b")
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "curation.log")
DIGEST_DIR = os.path.join(HERE, "readings")

SELFTEXT_CAP = 2500   # chars; keeps prompt inside the model's 8K context
COMMENTS_CAP = 3000
COMMENT_EACH_CAP = 400
LLM_TIMEOUT = 1200    # seconds; thinking pass can be slow
MAX_TOKENS = 4000

# The reader profile is shared with the chat under each reading (app.py),
# so the model talks to the same Luka it curates for.
PROFILE = """You are a reading curator for Luka, an engineer in Belgrade \
building a career in automation, AI/ML, robotics and industrial systems. What he \
knows well: Python, Docker, Kafka, PostgreSQL, MQTT/ESP32 embedded work, Grafana, \
ETL/Airflow. What he is currently learning (explain these when they come up): ML \
fundamentals, linear algebra, power electronics, robotics. He does not follow \
day-to-day AI drama, and does not want hype, memes, doomposting or culture war."""

SYSTEM = PROFILE + """
If RELATED PAST POSTS show the same news or discussion was already covered,
lean SKIP and say it's a rehash; if a past SIGNAL piece connects, reference it.

You will get one Reddit post from r/singularity with top comments. Respond in
EXACTLY this format:

VERDICT: SIGNAL or SKIP
REASON: one line — why this is/isn't worth Luka's reading time
---
(only if SIGNAL) A reading piece of 300-700 words in markdown. Not a dry summary:
rewrite and explain what actually matters in the post and discussion. Define
terms he may not know. Correct factual errors commenters make. Where genuine,
connect to his world (automation, embedded, data engineering, local LLMs on his
RX 6600 XT) or to the earlier pieces listed as context. End with one line:
**Worth going deeper:** yes/no plus what to search for if yes."""


def llm_ready() -> bool:
    """Quick health check so an unattended run aborts cleanly (and gets
    retried by the timer) instead of marking every pending post ERROR."""
    base = LOCALAI.rsplit("/chat/completions", 1)[0]
    try:
        return requests.get(f"{base}/models", timeout=10).ok
    except requests.RequestException:
        return False


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def ensure_table(cur) -> None:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            post_id    text PRIMARY KEY REFERENCES posts (id),
            verdict    text NOT NULL,           -- SIGNAL | SKIP | ERROR
            reason     text,
            article    text,                    -- markdown, NULL for SKIP
            model      text NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            post_id    text NOT NULL REFERENCES posts (id),
            role       text NOT NULL,           -- user | assistant
            content    text NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        )
    """)


CHAT_QUIET = 600  # s without chat activity before curation resumes


def wait_for_chat_idle(cur) -> None:
    """Luka's chat replies beat batch curation: while a conversation under a
    reading is active (a message in the last CHAT_QUIET seconds), hold off
    starting the next post so his follow-ups don't queue behind a ~6-minute
    curation call. Curation loses nothing — the backlog just resumes later."""
    paused = False
    while True:
        cur.execute("SELECT extract(epoch FROM now() - max(created_at)) AS quiet "
                    "FROM chat_messages")
        quiet = cur.fetchone()["quiet"]
        if quiet is None or quiet > CHAT_QUIET:
            if paused:
                log("  chat quiet — resuming curation")
            return
        if not paused:
            log("  chat active — pausing curation between posts")
            paused = True
        time.sleep(30)


def ensure_comments(conn, cur, post_id: str) -> None:
    """Read-through cache, same as the Flask post view."""
    cur.execute("SELECT comments_fetched_at FROM posts WHERE id = %s", (post_id,))
    if cur.fetchone()["comments_fetched_at"] is not None:
        return
    fetched = fetch_comments_from_arctic_shift(post_id)
    cur.executemany(
        """INSERT INTO comments (id, post_id, parent_id, author, body,
                                 created_utc, score)
           VALUES (%s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (id) DO UPDATE SET
               body = EXCLUDED.body, score = EXCLUDED.score, fetched_at = now()""",
        fetched,
    )
    cur.execute(
        "UPDATE posts SET comments_fetched_at = now(), num_comments = %s WHERE id = %s",
        (len(fetched), post_id),
    )
    conn.commit()
    time.sleep(1)  # be polite to the free API


def trim_comments(rows: list[tuple[str, int]],
                  cap: int = COMMENTS_CAP,
                  each_cap: int = COMMENT_EACH_CAP) -> str:
    """Pure trimming logic: (body, score) pairs -> prompt block within budget.

    Each comment is whitespace-collapsed and hard-capped; we stop adding
    comments when the total would exceed the block budget. Kept free of any
    DB/IO so it can be unit-tested (see test_curate.py)."""
    out, used = [], 0
    for body, score in rows:
        body = " ".join(body.split())[:each_cap]
        if used + len(body) > cap:
            break
        out.append(f"[{score} points] {body}")
        used += len(body)
    return "\n".join(out) if out else "(no comments)"


def top_comments(cur, post_id: str) -> str:
    """Top-level comments by score, trimmed to fit the context budget."""
    cur.execute(
        """SELECT body, score FROM comments
           WHERE post_id = %s AND parent_id IS NULL
           ORDER BY score DESC LIMIT 15""",
        (post_id,),
    )
    return trim_comments([(r["body"], r["score"]) for r in cur.fetchall()])


def post_vector(conn, cur, post: dict) -> str:
    """The post's embedding in pgvector text form, computing and storing it
    if fetch_posts ran while the embedding server was down."""
    if post.get("embedding") is not None:
        return post["embedding"]
    vec = vec_literal(embed(post_text(post["title"], post["selftext"])))
    cur.execute("UPDATE posts SET embedding = %s::vector WHERE id = %s",
                (vec, post["id"]))
    conn.commit()
    return vec


def related_posts(cur, post_id: str, vector: str) -> str:
    """Semantically closest already-judged posts, as a prompt block. Lets the
    model spot rehashes and link to earlier coverage beyond tonight's run.
    Cosine similarity below ~0.5 is noise for nomic-embed-text, so we cut there."""
    cur.execute(
        """SELECT p.title, p.created_utc, r.verdict, r.reason
           FROM posts p JOIN readings r ON r.post_id = p.id
           WHERE p.embedding IS NOT NULL AND p.id <> %s
             AND r.verdict IN ('SIGNAL', 'SKIP')
             AND 1 - (p.embedding <=> %s::vector) > 0.5
           ORDER BY p.embedding <=> %s::vector
           LIMIT 5""",
        (post_id, vector, vector),
    )
    rows = cur.fetchall()
    if not rows:
        return "(none)"
    return "\n".join(
        f"- [{r['verdict']}, {r['created_utc']:%Y-%m-%d}] {r['title']}"
        + (f" — {r['reason']}" if r["reason"] else "")
        for r in rows
    )


def ask_model(post: dict, comments: str, prev_titles: list[str],
              related: str = "(none)") -> str:
    prev = "\n".join(f"- {t}" for t in prev_titles[-15:]) or "(none yet)"
    user = (
        f"Earlier SIGNAL pieces tonight (for cross-referencing):\n{prev}\n\n"
        f"RELATED PAST POSTS (semantic matches, how they were judged):\n{related}\n\n"
        f"POST from r/{post['subreddit']} ({post['created_utc']:%Y-%m-%d}, "
        f"{post['score']} points, {post['num_comments']} comments)\n"
        f"TITLE: {post['title']}\n"
        f"BODY:\n{post['selftext'][:SELFTEXT_CAP]}\n\n"
        f"TOP COMMENTS:\n{comments}"
    )
    resp = requests.post(
        LOCALAI,
        json={
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
        },
        timeout=LLM_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def parse_reply(text: str) -> tuple[str, str, str | None]:
    verdict, reason, article = "ERROR", "", None
    head, _, tail = text.partition("---")
    for line in head.splitlines():
        if line.upper().startswith("VERDICT:"):
            v = line.split(":", 1)[1].strip().upper()
            verdict = "SIGNAL" if "SIGNAL" in v else "SKIP" if "SKIP" in v else "ERROR"
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    if verdict == "SIGNAL":
        article = tail.strip() or None
        if not article:  # model said SIGNAL but wrote nothing usable
            verdict, reason = "ERROR", "empty article"
    return verdict, reason, article


def append_digest(post: dict, article: str) -> None:
    os.makedirs(DIGEST_DIR, exist_ok=True)
    path = os.path.join(DIGEST_DIR, f"digest-{date.today():%Y-%m-%d}.md")
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write(f"# Reading digest — {date.today():%Y-%m-%d}\n\n")
        f.write(f"## {post['title']}\n\n")
        f.write(f"*[{post['score']} points, {post['num_comments']} comments]"
                f"({post['permalink']})*\n\n{article}\n\n---\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from psycopg.rows import dict_row
    conn = psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row)
    cur = conn.cursor()
    ensure_table(cur)
    conn.commit()

    # New posts plus earlier ERRORs (LLM hiccups) — those deserve a retry.
    cur.execute("""
        SELECT p.* FROM posts p
        LEFT JOIN readings r ON r.post_id = p.id
        WHERE r.post_id IS NULL OR r.verdict = 'ERROR'
        ORDER BY p.created_utc DESC
    """)
    todo = cur.fetchall()
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        log("run start: nothing to process")
        return
    if not llm_ready():
        log(f"run abort: LLM server not responding at {LOCALAI}")
        sys.exit(1)
    log(f"run start: {len(todo)} posts to process")

    cur.execute("SELECT p.title FROM readings r JOIN posts p ON p.id = r.post_id "
                "WHERE r.verdict = 'SIGNAL' ORDER BY r.created_at")
    prev_titles = [r["title"] for r in cur.fetchall()]

    done = signal = errors = 0
    for post in todo:
        wait_for_chat_idle(cur)
        t0 = time.time()
        try:
            try:
                ensure_comments(conn, cur, post["id"])
            except Exception as e:  # deleted post, API hiccup — curate without comments
                conn.rollback()
                log(f"  comments unavailable for {post['id']}: {e}")
            comments = top_comments(cur, post["id"])
            try:
                related = related_posts(cur, post["id"],
                                        post_vector(conn, cur, post))
            except Exception as e:  # embedding server down — curate without
                conn.rollback()
                related = "(unavailable)"
                log(f"  related posts unavailable for {post['id']}: {e}")
            reply = ask_model(post, comments, prev_titles, related)
            verdict, reason, article = parse_reply(reply)
            if verdict == "ERROR":  # malformed reply — one more try
                reply = ask_model(post, comments, prev_titles, related)
                verdict, reason, article = parse_reply(reply)
        except Exception as e:  # LLM/API/network hiccup: record and move on
            verdict, reason, article = "ERROR", f"{type(e).__name__}: {e}"[:300], None
            errors += 1
            time.sleep(30)  # give LocalAI a breather if it's choking
        cur.execute(
            """INSERT INTO readings (post_id, verdict, reason, article, model)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (post_id) DO UPDATE SET verdict = EXCLUDED.verdict,
                   reason = EXCLUDED.reason, article = EXCLUDED.article,
                   model = EXCLUDED.model, created_at = now()""",
            (post["id"], verdict, reason, article, MODEL),
        )
        conn.commit()
        if verdict == "SIGNAL":
            append_digest(post, article)
            prev_titles.append(post["title"])
            signal += 1
        done += 1
        log(f"[{done}/{len(todo)}] {verdict} ({time.time()-t0:.0f}s) {post['title'][:70]}"
            + (f" | {reason[:80]}" if verdict != "SIGNAL" else ""))

    log(f"run done: {done} processed, {signal} signal, {errors} errors")


if __name__ == "__main__":
    main()
