"""Verdict-quality eval: re-judge a sample of curated posts and measure agreement.

The curator's SIGNAL/SKIP calls are only trustworthy if they're stable — and
the interesting question for this hardware is whether the 8K-context 35B-MoE
setup is costing accuracy. This harness re-judges a random sample of already-
judged posts (same prompt assembly: body/article, comments, related-past-posts)
and reports where the second opinion disagrees with the stored verdict.

Two ways to use it:
  - same model (default): measures self-consistency — verdicts that flip on
    a re-roll were never confident calls.
  - EVAL_MODEL=<other model> (or --model): measures agreement with a stronger
    referee, e.g. a bigger model served by the same OpenAI-compatible server.

Verdict-only prompts (no article writing), so a 20-post sample is ~20 quick
calls, not 20 ten-minute generations. Results go to stdout and
readings/eval-<date>.md. Each disagreement lists both reasons — read those,
not just the percentage; the sample is small.

Usage:
    .venv/bin/python eval_verdicts.py                 # 20 posts, same model
    .venv/bin/python eval_verdicts.py --sample 40
    EVAL_MODEL=qwen3-72b .venv/bin/python eval_verdicts.py
"""

import argparse
import os
import time
from datetime import date, datetime, timezone

import psycopg
from psycopg.rows import dict_row

from curate_readings import (DIGEST_DIR, MODEL, SYSTEM, ask_model, llm_ready,
                             log, post_vector, related_posts, top_comments)

EVAL_SYSTEM = SYSTEM + """

For this evaluation run, respond with ONLY the VERDICT and REASON lines —
do not write the reading piece."""
EVAL_MAX_TOKENS = 1500  # room for the thinking pass, not for an article


def parse_verdict(text: str) -> tuple[str, str]:
    verdict, reason = "ERROR", ""
    for line in text.splitlines():
        if line.upper().startswith("VERDICT:"):
            v = line.split(":", 1)[1].upper()
            verdict = "SIGNAL" if "SIGNAL" in v else "SKIP" if "SKIP" in v else "ERROR"
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return verdict, reason


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--model",
                        default=os.environ.get("EVAL_MODEL", MODEL))
    args = parser.parse_args()

    conn = psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row)
    cur = conn.cursor()
    cur.execute(
        """SELECT p.*, r.verdict AS stored_verdict, r.reason AS stored_reason
           FROM posts p JOIN readings r ON r.post_id = p.id
           WHERE r.verdict IN ('SIGNAL', 'SKIP')
           ORDER BY random() LIMIT %s""",
        (args.sample,),
    )
    todo = cur.fetchall()
    if not todo:
        log("eval: nothing judged yet")
        return
    if not llm_ready():
        log("eval abort: LLM server not responding")
        return

    log(f"eval start: {len(todo)} posts, referee model {args.model}")
    results, errors = [], 0
    for i, post in enumerate(todo, 1):
        t0 = time.time()
        try:
            related = related_posts(cur, post["id"],
                                    post_vector(conn, cur, post))
            reply = ask_model(post, top_comments(cur, post["id"]), [],
                              related, system=EVAL_SYSTEM, model=args.model,
                              max_tokens=EVAL_MAX_TOKENS)
            verdict, reason = parse_verdict(reply)
        except Exception as e:
            verdict, reason = "ERROR", f"{type(e).__name__}: {e}"[:200]
        if verdict == "ERROR":
            errors += 1
        results.append({**post, "eval_verdict": verdict, "eval_reason": reason})
        mark = "=" if verdict == post["stored_verdict"] else "≠"
        log(f"  [{i}/{len(todo)}] {post['stored_verdict']} {mark} {verdict} "
            f"({time.time()-t0:.0f}s) {post['title'][:60]}")

    ok = [r for r in results if r["eval_verdict"] != "ERROR"]
    agree = [r for r in ok if r["eval_verdict"] == r["stored_verdict"]]
    flips = [r for r in ok if r["eval_verdict"] != r["stored_verdict"]]

    lines = [
        f"# Verdict eval — {date.today():%Y-%m-%d}",
        "",
        f"Referee: `{args.model}` · sample {len(results)} "
        f"({errors} errors excluded)",
        "",
        f"**Agreement: {len(agree)}/{len(ok)}"
        + (f" ({100 * len(agree) / len(ok):.0f}%)**" if ok else "**"),
        "",
    ]
    for kind, frm, to in (("wrongly skipped?", "SKIP", "SIGNAL"),
                          ("over-eager?", "SIGNAL", "SKIP")):
        rows = [r for r in flips if r["stored_verdict"] == frm]
        lines.append(f"## {frm} → {to} ({kind}) — {len(rows)}")
        for r in rows:
            lines += [f"- **{r['title']}** (`{r['id']}`)",
                      f"  - stored: {r['stored_reason']}",
                      f"  - referee: {r['eval_reason']}"]
        lines.append("")

    report = "\n".join(lines)
    print("\n" + report)
    os.makedirs(DIGEST_DIR, exist_ok=True)
    path = os.path.join(DIGEST_DIR, f"eval-{date.today():%Y-%m-%d}.md")
    with open(path, "w") as f:
        f.write(report + "\n")
    log(f"eval done: report in {path}")


if __name__ == "__main__":
    main()
