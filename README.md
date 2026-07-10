# Local LLM Reading Curator

A 24/7 reading curator: a **local LLM** (Qwen3.6-35B-A3B on consumer
hardware) triages Reddit and Hacker News posts into signal vs. noise and
**writes a personalized explainer article** for each post worth reading —
grounded in the post, the article it links to, its comment thread, and a
profile of what the reader already knows. Results land in Postgres and are
served as a reading queue by a small Flask app.

This is not a summarizer pipeline: the model **decides** (keep/skip, with a
stated reason), **explains** (rewrites the discussion into a 300–700-word
piece, defining unfamiliar terms and correcting factual errors made by
commenters), **cross-references** earlier pieces — both tonight's and,
via pgvector similarity search, semantically related posts it judged weeks
ago, so rehashed news gets skipped as a rehash — and **learns**: free-text
notes the reader leaves on posts ride into future prompts for similar posts,
and once a week it steps back and writes a synthesis of what actually moved.

```
Arctic Shift API ──► fetch_posts.py ─┐
HN Algolia API  ──► fetch_hn.py     ─┴─► Postgres+pgvector (posts, comments)
                            │                        ▲
                            ▼                        │ top-5 similar judged
                     Ollama (nomic-embed-text)       │ posts + reader notes
                     title+body → vector(768)        │
                                                     │
     linked-article text (trafilatura) ──► curate_readings.py  batch, resumable
                                    ▼
                     LocalAI / llama.cpp (Qwen3.6-35B-A3B, Vulkan)
                     8 GB VRAM + mmap'd RAM offload, 8K context
                                    │
                                    ▼
                      Postgres (readings, syntheses) + daily digest .md
                                    │
              app.py (Flask/gunicorn) ──► /readings queue · /weekly
```

## What's interesting in here

**Batch LLM pipeline built for cheap hardware** — the model is a 35B MoE
(3B active params) split across an 8 GB RX 6600 XT and memory-mapped system
RAM via llama.cpp's Vulkan backend. Measured during a full run: ~5 tokens/s,
peak 7.9 GB system RAM on a 16 GB machine, ~9.5 min per post including the
model's thinking pass. Speed is traded for quality deliberately: the pipeline
runs continuously and unattended (systemd timer, `Nice=10`), so latency per
post doesn't matter.

**Long-term memory via pgvector** — every post is embedded (nomic-embed-text,
768 dims) on ingest, and link posts additionally get chunk embeddings of the
extracted article or transcript, so similarity works on what the linked page
says, not just the headline; before judging a post, the curator retrieves the
five most similar already-judged posts (best match across all their vectors)
by cosine distance and puts their verdicts in the prompt. An 8K-context model can't remember hundreds of past posts, but
it doesn't have to — retrieval selects what's relevant. Observed effect: the
model starts calling out rehashes ("rehash of the 'LLM coding agent can ship
playable software' narrative") instead of judging every post in isolation.
No separate vector store: it's one `vector(768)` column and an `ORDER BY
embedding <=> ...` in the Postgres that was already there. Because Reddit and
HN share the table, rehash detection works across sources.

**A feedback loop, not a fixed profile** — under every post there's a box for
free-text notes ("wrong call, I wanted this one"). The retrieval above carries
those notes into the prompt whenever a *similar* post is judged, under a
marker the system prompt elevates above ordinary quoted content — so a note
left today changes verdicts on similar posts tomorrow, without retraining or
prompt surgery. The marker is neutralized in every Reddit/web-controlled
field, so hostile content can't imitate the reader's voice. And because the
judged sample never stops growing, `eval_verdicts.py` re-judges a random
sample (verdict-only, cheap) and reports where a second opinion — same model
or a stronger referee via `EVAL_MODEL` — disagrees with the stored verdicts.

**Reads the article, not just the thread** — link posts (most of what these
communities are) get the linked page downloaded once and reduced to readable
text with trafilatura, trimmed into the same context budget, so the model can
correct the commenters from the source instead of judging news by its
headline. YouTube links get their caption transcript instead (yt-dlp probes
the tracks, manual subs preferred over auto-generated), so talks and demos
are judged by what was said, not by the title. Media-only links (images,
raw video files) are skipped at ingest; caption-less videos and paywalled
pages degrade gracefully to title + comments.

**Zoom out on schedule** — `synthesize.py` (weekly systemd timer) hands the
model excerpts of the week's SIGNAL pieces and has it write "what actually
moved this week", cross-source, served at `/weekly` with links back to the
readings it drew from.

**Everything assumes failure** — the curator is fully resumable (processed
posts are recorded with `UPSERT` and skipped on restart), malformed model
output gets one retry, a failed comment fetch degrades to curating without
comments instead of losing the post, and every verdict is committed
immediately so a crash costs at most one post.

**Context budget engineering** — post body, top comments (by score), reader
profile, and a rolling list of tonight's earlier articles are trimmed to fit
input + a 4K-token generation inside the model's 8K context, so quality
stays stable regardless of thread size.

**Reading queue UI** — Flask + Jinja with keyset (cursor) pagination for
infinite scroll, a read-through comment cache (Arctic Shift for Reddit,
Algolia for HN) filled on first view, markdown-rendered articles, unread-first
ordering, and mark-as-read. Skipped posts stay inspectable with the model's
reason, so the curator's judgment can be audited. Chat replies stream into
the page as the model writes them — at ~5 tokens/s, watching the text grow
is the difference between "working" and "broken".

## Stack

Python · Flask + gunicorn · PostgreSQL + pgvector (psycopg3) · LocalAI
(llama.cpp, Vulkan) · Qwen3.6-35B-A3B GGUF · Ollama (nomic-embed-text) ·
Arctic Shift API · HN Algolia API · trafilatura · yt-dlp · Docker · systemd

## Run it

```bash
createdb reddit && psql reddit < schema.sql   # needs pgvector, e.g. pgvector/pgvector:pg16
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                        # point at your Postgres + LLM + embeddings servers
ollama pull nomic-embed-text

.venv/bin/python fetch_posts.py --subreddit singularity --limit 500
.venv/bin/python fetch_hn.py                # HN front page into the same table
.venv/bin/python embeddings.py              # backfill post + link-text chunk vectors
.venv/bin/python curate_readings.py         # one curation run (resumable)
.venv/bin/python app.py                     # http://localhost:8010/readings (dev)
```

To run it 24/7 — pipeline re-armed 30 min after each run finishes, weekly
synthesis Sunday evenings, web UI under gunicorn:

```bash
cp systemd/reddit-*.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now reddit-curator.timer reddit-synthesis.timer reddit-web.service
loginctl enable-linger $USER                # keep it running when logged out
```

Occasional maintenance: `prune.py` (dry run by default) drops the comment
threads and cached link articles of long-skipped posts while keeping every
title/embedding/verdict — the curator's memory — intact.

Any OpenAI-compatible server works for the LLM (`LOCALAI_URL`) and for
embeddings (`EMBEDDINGS_URL`); the reader profile and editorial rules live in
the `SYSTEM` prompt in `curate_readings.py` — edit them to curate for someone
who isn't me.

## Exposing it publicly

The site is safe to put behind a public tunnel as a read-only demo: anyone
can browse posts, readings and past chats, but every mutating action —
chatting with the model, pulling subreddits, toggling read state — requires
an admin cookie. Set `ADMIN_TOKEN` in `.env` (e.g.
`python -c "import secrets; print(secrets.token_urlsafe(24))"`), then open
`/unlock/<that token>` once in your browser; share that URL only with people
you trust to spend your GPU time. Without `ADMIN_TOKEN` set, the app fails
closed (nobody can mutate anything).

Untrusted content is also fenced defensively end to end: Reddit/HN text and
extracted web articles are quoted into LLM prompts as explicitly untrusted
data (with structural marker strings stripped — including the elevated
reader-feedback marker), and everything the model writes is sanitized with
`nh3` before it's rendered as HTML — so a hostile post can neither hijack
the curator's instructions nor get scripts into the page.

## Numbers from a real run

106 posts from r/singularity, one night on a Ryzen 3 3100 + RX 6600 XT (8 GB):
**~20 SIGNAL articles written, ~65 posts skipped with reasons, 0 lost to
errors** after the resilience fixes. Total cost: one idle desktop and some
electricity.
