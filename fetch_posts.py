"""Pull posts from a subreddit via Arctic Shift and store them in Postgres.

Text posts are stored with their body; link posts are stored with their URL
(the curator later fetches and extracts the linked article's text before
judging). Pure media posts — images, videos, galleries — are skipped: with
no text to read there is nothing for a text-only curator to judge.

Arctic Shift (https://arctic-shift.photon-reddit.com) is a free community-run
archive of Reddit with a public API — no credentials needed. It has near-live
data AND full history, unlike Reddit's own listings which stop at ~1000 posts.

Usage:
    .venv/bin/python fetch_posts.py                # 500 newest posts
    .venv/bin/python fetch_posts.py --limit 2000
    .venv/bin/python fetch_posts.py --before 2025-01-01   # go back in time
"""

import argparse
import os
import time
from datetime import datetime, timezone

import psycopg
import requests
from dotenv import load_dotenv

from embeddings import embed, post_text, vec_literal

load_dotenv()

API = "https://arctic-shift.photon-reddit.com/api/posts/search"
COMMENTS_API = "https://arctic-shift.photon-reddit.com/api/comments/search"
DEFAULT_DB = "postgresql://postgres:postgres@localhost:5432/reddit"
MAX_COMMENTS = 500  # safety cap per post when fetching comments

UPSERT = """
    INSERT INTO posts (id, subreddit, title, selftext, author,
                       created_utc, score, num_comments, permalink, url,
                       embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
    ON CONFLICT (id) DO UPDATE SET
        score        = EXCLUDED.score,
        num_comments = EXCLUDED.num_comments,
        selftext     = EXCLUDED.selftext,
        embedding    = COALESCE(EXCLUDED.embedding, posts.embedding),
        fetched_at   = now()
"""

# Hosts/extensions that mean "the content is a picture or video" — nothing a
# text model can read, so these posts are not stored at all. YouTube is NOT
# here on purpose: video announcements still get a title and a discussion
# thread worth judging; the curator just won't have the video's content.
MEDIA_HOSTS = ("i.redd.it", "v.redd.it", "imgur.com", "i.imgur.com",
               "reddit.com/gallery", "www.reddit.com/gallery")
MEDIA_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".gifv", ".webp",
                    ".mp4", ".webm")


def is_media_url(url: str) -> bool:
    bare = url.split("://", 1)[-1]
    path = bare.split("?", 1)[0].lower()
    return (bare.startswith(MEDIA_HOSTS)
            or path.endswith(MEDIA_EXTENSIONS))


def fetch_page(session: requests.Session, subreddit: str, before: int | None) -> list[dict]:
    params = {"subreddit": subreddit, "limit": 100, "sort": "desc"}
    if before is not None:
        params["before"] = before
    resp = session.get(API, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["data"]


def fetch_subreddit(subreddit: str, limit: int = 500, before: int | None = None,
                    progress=None) -> tuple[int, int]:
    """Scan up to `limit` posts from a subreddit and upsert the text posts.

    Importable, so both the CLI below and the web UI's /fetch endpoint can
    run the same code. `progress`, if given, is called after each page with
    (scanned, stored, oldest_timestamp) — the CLI prints it, the web UI
    stashes it for the status endpoint to report. Returns (scanned, stored).
    """
    session = requests.Session()
    session.headers["User-Agent"] = "singularity-scraper/0.1 (personal project)"

    scanned = stored = 0
    embeddings_down = False  # warn once, then store NULLs for the backfill
    with psycopg.connect(os.environ.get("DATABASE_URL", DEFAULT_DB)) as conn:
        with conn.cursor() as cur:
            while scanned < limit:
                posts = fetch_page(session, subreddit, before)
                if not posts:
                    break
                for p in posts:
                    scanned += 1
                    # next page = everything older than the oldest post seen
                    before = int(p["created_utc"])
                    if p["is_self"]:
                        if p["selftext"] in ("[removed]", "[deleted]", ""):
                            continue  # empty or moderated-away text post
                        url = None
                    else:
                        url = p.get("url") or ""
                        if not url or is_media_url(url):
                            continue  # image/video/gallery: nothing to read
                    vector = None
                    if not embeddings_down:
                        try:
                            vector = vec_literal(embed(post_text(p["title"], p["selftext"])))
                        except Exception as e:
                            embeddings_down = True
                            print(f"embedding server unavailable ({e}); "
                                  "storing posts without vectors — run embeddings.py later")
                    cur.execute(UPSERT, (
                        p["id"],
                        subreddit,
                        p["title"],
                        p["selftext"],
                        p.get("author"),
                        datetime.fromtimestamp(p["created_utc"], tz=timezone.utc),
                        p.get("score", 0),
                        p.get("num_comments", 0),
                        f"https://reddit.com{p['permalink']}",
                        url,
                        vector,
                    ))
                    stored += 1
                conn.commit()  # keep progress even if a later page fails
                if progress:
                    progress(scanned, stored, before)
                time.sleep(1)  # be polite to the free API
    return scanned, stored


def fetch_comments(post_id: str) -> list[tuple]:
    """Comments for any post id, whatever its source: hn_-prefixed ids go to
    the Algolia item API, everything else to Arctic Shift. The one place
    callers (curator, web UI) need to know about."""
    if post_id.startswith("hn_"):
        from fetch_hn import fetch_comments_hn  # avoid an import cycle
        return fetch_comments_hn(post_id)
    return fetch_comments_from_arctic_shift(post_id)


def community(row: dict) -> str:
    """Human label for where a post is from: 'HN' or 'r/<subreddit>'."""
    return "HN" if row.get("source") == "hn" else f"r/{row['subreddit']}"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subreddit", default="singularity")
    parser.add_argument("--limit", type=int, default=500,
                        help="how many posts to scan (text posts are a subset)")
    parser.add_argument("--before", default=None,
                        help="only posts before this date (YYYY-MM-DD), for backfilling")
    args = parser.parse_args()

    before = None
    if args.before:
        dt = datetime.strptime(args.before, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        before = int(dt.timestamp())

    def report(scanned, stored, oldest_ts):
        oldest = datetime.fromtimestamp(oldest_ts, tz=timezone.utc)
        print(f"  scanned {scanned}, stored {stored}, back to {oldest:%Y-%m-%d %H:%M}")

    scanned, stored = fetch_subreddit(args.subreddit, args.limit, before, report)
    print(f"Done. Scanned {scanned} posts, stored/updated {stored} text posts.")


if __name__ == "__main__":
    main()
