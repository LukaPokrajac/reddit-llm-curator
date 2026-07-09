"""Pull text posts from a subreddit via Arctic Shift and store them in Postgres.

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

load_dotenv()

API = "https://arctic-shift.photon-reddit.com/api/posts/search"
DEFAULT_DB = "postgresql://postgres:postgres@localhost:5432/reddit"

UPSERT = """
    INSERT INTO posts (id, subreddit, title, selftext, author,
                       created_utc, score, num_comments, permalink)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        score        = EXCLUDED.score,
        num_comments = EXCLUDED.num_comments,
        selftext     = EXCLUDED.selftext,
        fetched_at   = now()
"""


def fetch_page(session: requests.Session, subreddit: str, before: int | None) -> list[dict]:
    params = {"subreddit": subreddit, "limit": 100, "sort": "desc"}
    if before is not None:
        params["before"] = before
    resp = session.get(API, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["data"]


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

    session = requests.Session()
    session.headers["User-Agent"] = "singularity-scraper/0.1 (personal project)"

    scanned = stored = 0
    with psycopg.connect(os.environ.get("DATABASE_URL", DEFAULT_DB)) as conn:
        with conn.cursor() as cur:
            while scanned < args.limit:
                posts = fetch_page(session, args.subreddit, before)
                if not posts:
                    break
                for p in posts:
                    scanned += 1
                    # next page = everything older than the oldest post seen
                    before = int(p["created_utc"])
                    if not p["is_self"]:
                        continue
                    if p["selftext"] in ("[removed]", "[deleted]", ""):
                        continue
                    cur.execute(UPSERT, (
                        p["id"],
                        args.subreddit,
                        p["title"],
                        p["selftext"],
                        p.get("author"),
                        datetime.fromtimestamp(p["created_utc"], tz=timezone.utc),
                        p.get("score", 0),
                        p.get("num_comments", 0),
                        f"https://reddit.com{p['permalink']}",
                    ))
                    stored += 1
                conn.commit()  # keep progress even if a later page fails
                oldest = datetime.fromtimestamp(before, tz=timezone.utc)
                print(f"  scanned {scanned}, stored {stored}, back to {oldest:%Y-%m-%d %H:%M}")
                time.sleep(1)  # be polite to the free API

    print(f"Done. Scanned {scanned} posts, stored/updated {stored} text posts.")


if __name__ == "__main__":
    main()
