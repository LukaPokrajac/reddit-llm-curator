"""Tiny web UI for browsing the scraped Reddit posts in Postgres.

How a Flask app works, in one paragraph: Flask starts an HTTP server. Each
function decorated with @app.route("...") is a "view" — when a browser asks
for that URL, Flask calls the function and sends back whatever it returns
(here: HTML rendered from a template in templates/). The view functions are
where we run SQL; the templates are where the HTML lives. That separation
(data logic in Python, presentation in templates) is the core pattern of
almost every web framework.

Usage:
    .venv/bin/python app.py        # then open http://localhost:8010
"""

import os
import re
import threading
from collections import defaultdict

import markdown
import psycopg
import requests
from psycopg.rows import dict_row
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, render_template, request

from fetch_posts import fetch_subreddit, fetch_comments_from_arctic_shift
# Importing curate_readings is safe: its run loop is guarded by __main__.
from curate_readings import LOCALAI, MODEL, PROFILE, LLM_TIMEOUT, top_comments

load_dotenv()

DEFAULT_DB = "postgresql://postgres:postgres@localhost:5432/reddit"
BATCH = 25          # posts per infinite-scroll load

app = Flask(__name__)


def get_conn():
    # A new connection per request is fine at this scale. Real apps use a
    # connection *pool* to avoid the ~10ms connect cost on every request.
    # row_factory=dict_row makes rows come back as dicts (row["title"])
    # instead of tuples (row[2]) — much nicer in templates.
    return psycopg.connect(os.environ.get("DATABASE_URL", DEFAULT_DB),
                           row_factory=dict_row)


def fetch_batch(cur, q: str, before: float | None) -> list[dict]:
    """One batch of posts, using *keyset* (cursor) pagination.

    Instead of OFFSET ("skip N rows"), we say "give me posts older than the
    last one I've already seen". Two advantages: Postgres can jump straight
    to the spot via the created_utc index instead of counting skipped rows,
    and new posts arriving mid-scroll can't shift everything down and make
    you see duplicates — the classic OFFSET bug.
    """
    # NEVER build SQL by string-formatting user input (SQL injection!).
    # The %s placeholders let the driver send query and values separately,
    # so a malicious q can't change the query's meaning.
    where = ["TRUE"]
    params: list = []
    if q:
        where.append("(title ILIKE %s OR selftext ILIKE %s)")
        params += [f"%{q}%", f"%{q}%"]
    if before is not None:
        where.append("created_utc < to_timestamp(%s)")
        params.append(before)

    # The list view doesn't need full essays, so select a 300-char preview.
    cur.execute(
        f"""
        SELECT id, subreddit, title, author, created_utc, score, num_comments,
               left(selftext, 300) AS preview
        FROM posts
        WHERE {" AND ".join(where)}
        ORDER BY created_utc DESC
        LIMIT %s
        """,
        params + [BATCH],
    )
    return cur.fetchall()


def next_cursor(posts: list[dict]) -> float | None:
    """The cursor for the following batch = timestamp of the oldest post
    in this one. None means we've reached the end."""
    if len(posts) < BATCH:
        return None
    return posts[-1]["created_utc"].timestamp()


@app.route("/")
def index():
    q = request.args.get("q", "").strip()
    with get_conn() as conn, conn.cursor() as cur:
        if q:
            cur.execute("SELECT count(*) AS n FROM posts "
                        "WHERE title ILIKE %s OR selftext ILIKE %s",
                        (f"%{q}%", f"%{q}%"))
        else:
            cur.execute("SELECT count(*) AS n FROM posts")
        total = cur.fetchone()["n"]
        posts = fetch_batch(cur, q, before=None)
    return render_template("index.html", posts=posts, q=q, total=total,
                           next=next_cursor(posts))


@app.route("/more")
def more():
    """JSON endpoint the infinite-scroll JS calls for the next batch.

    It returns {"html": "<rendered post cards>", "next": <cursor or null>}.
    The server still renders the HTML (so templates stay in one place);
    the browser just appends it. This pattern is sometimes called
    "HTML over the wire" — simpler than shipping a JS framework.
    """
    q = request.args.get("q", "").strip()
    before = request.args.get("before", type=float)
    with get_conn() as conn, conn.cursor() as cur:
        posts = fetch_batch(cur, q, before)
    return jsonify(html=render_template("_post_list.html", posts=posts),
                   next=next_cursor(posts))


# --- Pulling new subreddits from the web UI -------------------------------
#
# A pull scans hundreds of posts with a 1s pause per page, so it can take
# minutes — far too long to run inside a request (the browser would time out).
# The classic fix at this scale: run it in a background *thread* and let the
# page poll a status endpoint. The job state lives in a plain module-level
# dict, which works because app.run() is a single process; with multiple
# workers (gunicorn etc.) each process would have its own dict and you'd
# reach for Redis or a job queue (Celery/RQ) instead.
fetch_job: dict = {"state": "idle"}
fetch_lock = threading.Lock()  # so two submits can't both pass the "running?" check

# Subreddit names are 2–21 chars of letters/digits/underscore. Validating
# up front gives a clear error instead of a silent empty result.
SUBREDDIT_RE = re.compile(r"[A-Za-z0-9_]{2,21}")


def run_fetch(subreddit: str, limit: int) -> None:
    def progress(scanned, stored, oldest_ts):
        fetch_job.update(scanned=scanned, stored=stored)

    try:
        scanned, stored = fetch_subreddit(subreddit, limit, progress=progress)
        fetch_job.update(state="done", scanned=scanned, stored=stored)
    except Exception as exc:  # surface the failure to the UI, don't die silently
        fetch_job.update(state="error", error=str(exc))


@app.route("/fetch", methods=["POST"])
def fetch():
    subreddit = request.form.get("subreddit", "").strip().removeprefix("r/")
    limit = min(request.form.get("limit", type=int) or 500, 10_000)
    if not SUBREDDIT_RE.fullmatch(subreddit):
        return jsonify(error="not a valid subreddit name"), 400

    with fetch_lock:
        if fetch_job["state"] == "running":
            # 409 Conflict: the request is fine, the current state forbids it.
            return jsonify(error=f"already pulling r/{fetch_job['subreddit']}"), 409
        fetch_job.clear()
        fetch_job.update(state="running", subreddit=subreddit, scanned=0, stored=0)

    # daemon=True: don't let a half-finished pull block server shutdown —
    # the per-page commit in fetch_subreddit means nothing is lost.
    threading.Thread(target=run_fetch, args=(subreddit, limit), daemon=True).start()
    return jsonify(fetch_job)


@app.route("/fetch/status")
def fetch_status():
    return jsonify(fetch_job)


def build_tree(comments: list[dict]) -> list[dict]:
    """Turn the flat comment rows into a nested tree.

    Reddit stores threading as a parent pointer on each comment; to render
    it we need the opposite direction (children lists). One pass groups
    children by parent, then each comment gets a .replies list.
    """
    by_parent = defaultdict(list)
    for c in comments:
        by_parent[c["parent_id"]].append(c)
    for kids in by_parent.values():
        kids.sort(key=lambda c: c["score"], reverse=True)

    def attach(c):
        c["replies"] = [attach(k) for k in by_parent.get(c["id"], [])]
        return c

    return [attach(c) for c in by_parent[None]]


@app.route("/post/<post_id>")
def post(post_id):
    refresh = "refresh" in request.args
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM posts WHERE id = %s", (post_id,))
        row = cur.fetchone()
        if row is None:
            abort(404)

        cur.execute("SELECT * FROM comments WHERE post_id = %s", (post_id,))
        comments = cur.fetchall()

        # Read-through cache: comments live in our DB; the first time a post
        # is opened (or on ?refresh) we fill the cache from Arctic Shift.
        # Every later view is served locally — fast, and kind to the free API.
        #
        # Why not "fetch if num_comments > 0"? Because num_comments is a
        # snapshot from whenever the *post* was scraped — often minutes after
        # posting, before anyone commented. Stale-cache bugs like this are
        # why we track our own marker (comments_fetched_at) instead of
        # trusting a number that ages.
        if row["comments_fetched_at"] is None or refresh:
            fetched = fetch_comments_from_arctic_shift(post_id)
            cur.executemany(
                """
                INSERT INTO comments (id, post_id, parent_id, author, body,
                                      created_utc, score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    body = EXCLUDED.body, score = EXCLUDED.score,
                    fetched_at = now()
                """,
                fetched,
            )
            # While we're at it, correct the stale comment count on the post.
            cur.execute(
                "UPDATE posts SET comments_fetched_at = now(), "
                "num_comments = %s WHERE id = %s",
                (len(fetched), post_id),
            )
            conn.commit()
            cur.execute("SELECT * FROM comments WHERE post_id = %s", (post_id,))
            comments = cur.fetchall()

    return render_template("post.html", p=row, tree=build_tree(comments),
                           n_comments=len(comments))


@app.route("/readings")
def readings():
    """The curated reading list: articles Qwen wrote overnight from posts
    it judged worth Luka's time. Unread first, then newest."""
    show = request.args.get("show", "signal")  # signal | skip | all
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT verdict, count(*) AS n FROM readings GROUP BY verdict")
        counts = {r["verdict"]: r["n"] for r in cur.fetchall()}
        where = "TRUE" if show == "all" else "r.verdict = %s"
        params = [] if show == "all" else [show.upper()]
        cur.execute(
            f"""
            SELECT r.post_id, r.verdict, r.reason, r.read_at, r.created_at,
                   coalesce(array_length(regexp_split_to_array(r.article, '\\s+'), 1), 0)
                       AS words,
                   p.subreddit, p.title, p.created_utc, p.score, p.num_comments
            FROM readings r JOIN posts p ON p.id = r.post_id
            WHERE {where}
            ORDER BY (r.read_at IS NULL) DESC, p.created_utc DESC
            """,
            params,
        )
        rows = cur.fetchall()
    return render_template("readings.html", rows=rows, counts=counts, show=show)


@app.route("/readings/status")
def readings_status():
    """Cheap poll target so the readings list refreshes itself when the
    24/7 curator lands a new verdict. (Static routes beat the <post_id>
    converter below, so 'status' is never taken for a post id.)"""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n, max(created_at) AS latest FROM readings")
        row = cur.fetchone()
    return jsonify(n=row["n"],
                   latest=row["latest"].isoformat() if row["latest"] else None)


@app.route("/readings/<post_id>")
def reading(post_id):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT r.*, p.subreddit, p.title, p.permalink, p.created_utc,
                      p.score, p.num_comments
               FROM readings r JOIN posts p ON p.id = r.post_id
               WHERE r.post_id = %s""",
            (post_id,),
        )
        row = cur.fetchone()
        if row is None:
            abort(404)
        cur.execute("SELECT role, content FROM chat_messages "
                    "WHERE post_id = %s ORDER BY id", (post_id,))
        chat = cur.fetchall()
    for m in chat:  # user text renders escaped; model replies are markdown
        if m["role"] == "assistant":
            m["html"] = markdown.markdown(m["content"], extensions=["extra"])
    body = markdown.markdown(row["article"] or "", extensions=["extra"])
    return render_template("reading.html", r=row, body=body, chat=chat)


# --- Chat with the curator under a reading --------------------------------
#
# Same background-thread + polling pattern as /fetch: a Qwen reply takes
# minutes on this hardware (more if the 24/7 curator has it busy), far past
# what a request — or the cloudflared tunnel's ~100s limit — will wait.
# POST stores the user message and starts a thread; the page polls
# /chat/status until the reply is in the DB, then reloads.
chat_jobs: dict = {}  # post_id -> {"state": running|done|error, ...}
chat_lock = threading.Lock()

# The model can also revise its piece from chat. The contract mirrors the
# curator's VERDICT/--- format: prose first, then a marker line, then the
# full replacement article. parse, don't guess.
ARTICLE_MARKER = "---ARTICLE---"

CHAT_SYSTEM = PROFILE + f"""

You wrote (or declined to write) the reading piece below for Luka, from a
Reddit post and its top comments. Continue as a conversation: answer his
follow-ups in concise markdown, grounded in the post and comments — say so
when they don't answer something, don't invent thread content. Answering
what he actually asked comes first — never launch into revising the piece
unless he explicitly asks for a change to it. If he asks
you to change, extend or rewrite the piece itself, reply with one short line
on what you changed, then a line containing exactly
{ARTICLE_MARKER}
followed by the complete revised piece in markdown (it replaces the old one,
so include all of it, not just the changed part)."""

# Context budget, in the spirit of the curator's caps: everything below has
# to fit the model's 8K context alongside a 3K-token reply.
CHAT_BODY_CAP = 1500      # the piece matters more than the raw post here
CHAT_ARTICLE_CAP = 4000
CHAT_HISTORY_MAX = 6      # prior turns sent to the model
CHAT_MSG_CAP = 800        # chars per prior turn
CHAT_MAX_TOKENS = 3000


def chat_context(cur, post_id: str) -> str:
    """The pinned first message: post, comments, and the current piece."""
    cur.execute(
        """SELECT p.subreddit, p.title, p.selftext, p.created_utc, p.score,
                  r.verdict, r.reason, r.article
           FROM posts p JOIN readings r ON r.post_id = p.id
           WHERE p.id = %s""",
        (post_id,),
    )
    row = cur.fetchone()
    piece = (row["article"] or "")[:CHAT_ARTICLE_CAP] or (
        f"(you skipped this post — your reason: {row['reason']})")
    return (
        f"POST from r/{row['subreddit']} ({row['created_utc']:%Y-%m-%d}, "
        f"{row['score']} points)\n"
        f"TITLE: {row['title']}\n"
        f"BODY:\n{row['selftext'][:CHAT_BODY_CAP]}\n\n"
        f"TOP COMMENTS:\n{top_comments(cur, post_id)}\n\n"
        f"YOUR PIECE:\n{piece}"
    )


def split_chat_reply(text: str) -> tuple[str, str | None]:
    """(chat prose, replacement article or None)."""
    head, sep, tail = text.partition(ARTICLE_MARKER)
    if sep and tail.strip():
        return head.strip(), tail.strip()
    return text.strip(), None


def run_chat(post_id: str) -> None:
    try:
        with get_conn() as conn, conn.cursor() as cur:
            context = chat_context(cur, post_id)
            cur.execute("SELECT role, content FROM chat_messages "
                        "WHERE post_id = %s ORDER BY id", (post_id,))
            history = cur.fetchall()

        messages = [{"role": "system", "content": f"{CHAT_SYSTEM}\n\n{context}"}]
        for m in history[-CHAT_HISTORY_MAX:]:
            messages.append({"role": m["role"],
                             "content": m["content"][:CHAT_MSG_CAP]})

        resp = requests.post(
            LOCALAI,
            json={"model": MODEL, "max_tokens": CHAT_MAX_TOKENS,
                  "messages": messages},
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"].strip()
        prose, new_article = split_chat_reply(reply)

        with get_conn() as conn, conn.cursor() as cur:
            if new_article:
                cur.execute("UPDATE readings SET article = %s WHERE post_id = %s",
                            (new_article, post_id))
            cur.execute("INSERT INTO chat_messages (post_id, role, content) "
                        "VALUES (%s, 'assistant', %s)",
                        (post_id, prose or "(revised the piece)"))
            conn.commit()
        chat_jobs[post_id] = {"state": "done"}
    except Exception as exc:  # surface the failure to the UI, don't die silently
        chat_jobs[post_id] = {"state": "error", "error": str(exc)[:300]}


@app.route("/readings/<post_id>/chat", methods=["POST"])
def chat_send(post_id):
    message = request.form.get("message", "").strip()
    if not message:
        return jsonify(error="empty message"), 400

    with chat_lock:
        if chat_jobs.get(post_id, {}).get("state") == "running":
            return jsonify(error="still thinking about the last message"), 409
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM readings WHERE post_id = %s", (post_id,))
            if cur.fetchone() is None:
                abort(404)
            cur.execute("INSERT INTO chat_messages (post_id, role, content) "
                        "VALUES (%s, 'user', %s)", (post_id, message))
            conn.commit()
        chat_jobs[post_id] = {"state": "running"}

    threading.Thread(target=run_chat, args=(post_id,), daemon=True).start()
    return jsonify(chat_jobs[post_id])


@app.route("/readings/<post_id>/chat/status")
def chat_status(post_id):
    if post_id not in chat_jobs:
        # The job dict is memory, but the question is in the DB — if Flask
        # restarted while a reply was cooking, the last stored message is
        # still the user's. Respawn the reply instead of reporting idle, so
        # a pending chat survives any restart (the page polls on load).
        with chat_lock:
            if post_id not in chat_jobs:  # re-check: another poll may have won
                with get_conn() as conn, conn.cursor() as cur:
                    cur.execute("SELECT role FROM chat_messages WHERE post_id = %s "
                                "ORDER BY id DESC LIMIT 1", (post_id,))
                    row = cur.fetchone()
                if row and row["role"] == "user":
                    chat_jobs[post_id] = {"state": "running"}
                    threading.Thread(target=run_chat, args=(post_id,),
                                     daemon=True).start()
    job = chat_jobs.get(post_id, {"state": "idle"})
    if job["state"] in ("done", "error"):  # one-shot result: hand over, reset
        chat_jobs.pop(post_id, None)
    return jsonify(job)


@app.route("/readings/<post_id>/read", methods=["POST"])
def mark_read(post_id):
    """Toggle read/unread; the reading list uses it to sink finished pieces."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE readings
               SET read_at = CASE WHEN read_at IS NULL THEN now() END
               WHERE post_id = %s
               RETURNING read_at""",
            (post_id,),
        )
        row = cur.fetchone()
        if row is None:
            abort(404)
        conn.commit()
    return jsonify(read=row["read_at"] is not None)


if __name__ == "__main__":
    # Bound to the LAN so the cloudflared tunnel on the laptop can reach it
    # (readings.lukapokrajac.com). debug is opt-in via FLASK_DEBUG=1: the
    # Werkzeug debugger executes arbitrary code, so it must never run behind
    # the public hostname. 8000 was taken by the etl-api container, hence 8010.
    app.run(host="0.0.0.0", port=8010,
            debug=os.environ.get("FLASK_DEBUG") == "1")
