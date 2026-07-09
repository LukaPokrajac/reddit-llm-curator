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
from collections import defaultdict
from datetime import datetime, timezone

import markdown
import psycopg
import requests
from psycopg.rows import dict_row
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, render_template, request

load_dotenv()

DEFAULT_DB = "postgresql://postgres:postgres@localhost:5432/reddit"
COMMENTS_API = "https://arctic-shift.photon-reddit.com/api/comments/search"
BATCH = 25          # posts per infinite-scroll load
MAX_COMMENTS = 500  # safety cap per post when fetching from Arctic Shift

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
        SELECT id, title, author, created_utc, score, num_comments,
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


def fetch_comments_from_arctic_shift(post_id: str) -> list[tuple]:
    """Pull all comments for one post from Arctic Shift, oldest first."""
    session = requests.Session()
    session.headers["User-Agent"] = "singularity-scraper/0.1 (personal project)"
    rows, after = [], None
    while len(rows) < MAX_COMMENTS:
        params = {"link_id": post_id, "limit": 100, "sort": "asc"}
        if after is not None:
            params["after"] = after
        resp = session.get(COMMENTS_API, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()["data"]
        if not batch:
            break
        for c in batch:
            after = int(c["created_utc"])
            # parent_id is "t3_<postid>" for top-level comments and
            # "t1_<commentid>" for replies — Reddit's type-prefix scheme.
            parent = c["parent_id"]
            parent = None if parent.startswith("t3_") else parent[3:]
            rows.append((
                c["id"], post_id, parent, c.get("author"),
                c.get("body", ""),
                datetime.fromtimestamp(c["created_utc"], tz=timezone.utc),
                c.get("score", 0),
            ))
        if len(batch) < 100:
            break
    return rows


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
                   p.title, p.created_utc, p.score, p.num_comments
            FROM readings r JOIN posts p ON p.id = r.post_id
            WHERE {where}
            ORDER BY (r.read_at IS NULL) DESC, p.created_utc DESC
            """,
            params,
        )
        rows = cur.fetchall()
    return render_template("readings.html", rows=rows, counts=counts, show=show)


@app.route("/readings/<post_id>")
def reading(post_id):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT r.*, p.title, p.permalink, p.created_utc, p.score,
                      p.num_comments
               FROM readings r JOIN posts p ON p.id = r.post_id
               WHERE r.post_id = %s""",
            (post_id,),
        )
        row = cur.fetchone()
        if row is None:
            abort(404)
    body = markdown.markdown(row["article"] or "", extensions=["extra"])
    return render_template("reading.html", r=row, body=body)


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
