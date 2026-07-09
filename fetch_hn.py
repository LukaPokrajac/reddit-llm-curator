"""Pull Hacker News front-page stories into the same posts table via Algolia.

HN's official Algolia search API (https://hn.algolia.com/api) is free and
needs no credentials. Stories land next to the Reddit posts with
source='hn' and hn_-prefixed ids (Algolia's numeric ids could in principle
collide with Reddit's base36 ones), so everything downstream — embeddings,
pgvector retrieval, curation, the web UI — works on them unchanged. Cross-
source rehash detection comes for free: an HN thread and a Reddit thread
about the same news embed close together.

Usage:
    .venv/bin/python fetch_hn.py               # current front page (~30)
    .venv/bin/python fetch_hn.py --pages 2     # front page, two pages deep
"""

import argparse
import html
import os
import re
import time
from datetime import datetime, timezone

import psycopg
import requests
from dotenv import load_dotenv

from embeddings import embed, post_text, vec_literal
from fetch_posts import DEFAULT_DB, MAX_COMMENTS, UPSERT, is_media_url

load_dotenv()

SEARCH_API = "https://hn.algolia.com/api/v1/search"
ITEM_API = "https://hn.algolia.com/api/v1/items/{}"
UA = {"User-Agent": "reading-curator/0.1 (personal project)"}

# Same UPSERT as the Reddit ingest, with source='hn' stamped in the same
# statement so an HN row can never sit there marked 'reddit'.
HN_UPSERT = (UPSERT
             .replace("permalink, url,", "permalink, url, source,")
             .replace("%s, %s::vector)", "%s, 'hn', %s::vector)"))
assert "'hn'" in HN_UPSERT  # the replace must not silently miss


def html_to_text(fragment: str | None) -> str:
    """Algolia returns story/comment text as an HTML fragment; flatten it to
    plain text (<p> becomes a blank line, entities decoded, tags dropped)."""
    if not fragment:
        return ""
    text = re.sub(r"<p[^>]*>", "\n\n", fragment)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def story_row(hit: dict) -> tuple | None:
    """Map one Algolia story hit onto the posts-table UPSERT, or None if it's
    a media link we skip (same policy as the Reddit ingest)."""
    url = hit.get("url") or None
    if url and is_media_url(url):
        return None
    text = html_to_text(hit.get("story_text"))
    if not url and not text:
        return None  # dead/flagged item: no link, no text
    return (
        f"hn_{hit['objectID']}",
        "hackernews",                       # community label for the UI
        hit["title"],
        text,
        hit.get("author"),
        datetime.fromtimestamp(hit["created_at_i"], tz=timezone.utc),
        hit.get("points") or 0,
        hit.get("num_comments") or 0,
        f"https://news.ycombinator.com/item?id={hit['objectID']}",
        url,
    )


def fetch_front_page(pages: int = 1, progress=None) -> tuple[int, int]:
    """Upsert the current HN front page (optionally more pages). Returns
    (scanned, stored)."""
    session = requests.Session()
    session.headers.update(UA)
    scanned = stored = 0
    embeddings_down = False
    with psycopg.connect(os.environ.get("DATABASE_URL", DEFAULT_DB)) as conn:
        with conn.cursor() as cur:
            for page in range(pages):
                resp = session.get(SEARCH_API, timeout=30,
                                   params={"tags": "front_page", "page": page})
                resp.raise_for_status()
                for hit in resp.json()["hits"]:
                    scanned += 1
                    row = story_row(hit)
                    if row is None:
                        continue
                    vector = None
                    if not embeddings_down:
                        try:
                            vector = vec_literal(embed(post_text(row[2], row[3])))
                        except Exception as e:
                            embeddings_down = True
                            print(f"embedding server unavailable ({e}); "
                                  "storing without vectors — run embeddings.py later")
                    cur.execute(HN_UPSERT, row + (vector,))
                    stored += 1
                conn.commit()
                if progress:
                    progress(scanned, stored, None)
                time.sleep(1)
    return scanned, stored


def flatten_comments(item: dict) -> list[tuple]:
    """The item endpoint returns the full comment tree; walk it into the
    comments-table rows. HN comments carry no public score — stored as 0,
    display/trim order falls back to age."""
    post_id = f"hn_{item['id']}"
    rows: list[tuple] = []

    def walk(node: dict, parent: str | None) -> None:
        for child in node.get("children", []):
            if len(rows) >= MAX_COMMENTS:
                return
            body = html_to_text(child.get("text"))
            cid = f"hn_{child['id']}"
            if body:  # deleted comments come back textless
                rows.append((
                    cid, post_id, parent, child.get("author"), body,
                    datetime.fromtimestamp(child["created_at_i"],
                                           tz=timezone.utc),
                    0,
                ))
            walk(child, cid)

    walk(item, None)
    return rows


def fetch_comments_hn(post_id: str) -> list[tuple]:
    """All comments for one hn_-prefixed post id, as comments-table rows."""
    item_id = post_id.removeprefix("hn_")
    resp = requests.get(ITEM_API.format(item_id), headers=UA, timeout=30)
    resp.raise_for_status()
    return flatten_comments(resp.json())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=1,
                        help="front-page pages to pull (~30 stories each)")
    args = parser.parse_args()
    scanned, stored = fetch_front_page(args.pages)
    print(f"Done. Scanned {scanned} stories, stored/updated {stored}.")


if __name__ == "__main__":
    main()
