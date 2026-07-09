"""Data lifecycle: shed the bulk of old skipped posts, keep the memory.

The pgvector retrieval needs every judged post's title, embedding and
verdict forever — that IS the curator's long-term memory. What nothing
needs months later is the bulk attached to posts that were skipped:
their comment threads and extracted link articles. This deletes those
for SKIP posts judged more than --days ago (default 90).

SIGNAL posts are never touched: their comments back the reading page.

Usage:
    .venv/bin/python prune.py            # dry run: report what would go
    .venv/bin/python prune.py --apply
"""

import argparse
import os

import psycopg
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--apply", action="store_true",
                        help="actually delete (default is a dry run)")
    args = parser.parse_args()

    where = """post_id IN (
        SELECT post_id FROM readings
        WHERE verdict = 'SKIP' AND created_at < now() - %s * interval '1 day')"""

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM comments WHERE {where}",
                        (args.days,))
            n_comments = cur.fetchone()[0]
            cur.execute(
                f"""SELECT count(*) FROM posts
                    WHERE link_text <> '' AND id IN (
                        SELECT post_id FROM readings
                        WHERE verdict = 'SKIP'
                          AND created_at < now() - %s * interval '1 day')""",
                (args.days,))
            n_links = cur.fetchone()[0]

            if not args.apply:
                print(f"dry run: would delete {n_comments} comments and "
                      f"{n_links} stored link articles for SKIP posts older "
                      f"than {args.days} days (use --apply)")
                return

            cur.execute(f"DELETE FROM comments WHERE {where}", (args.days,))
            # comments_fetched_at stays set: a deliberately pruned thread
            # shouldn't be re-downloaded on the next page view.
            cur.execute(
                f"""UPDATE posts SET link_text = ''
                    WHERE link_text <> '' AND id IN (
                        SELECT post_id FROM readings
                        WHERE verdict = 'SKIP'
                          AND created_at < now() - %s * interval '1 day')""",
                (args.days,))
            conn.commit()
            print(f"deleted {n_comments} comments, "
                  f"cleared {n_links} link articles")


if __name__ == "__main__":
    main()
