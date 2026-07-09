# Reddit LLM Curator

An overnight reading curator: a **local LLM** (Qwen3.6-35B-A3B on consumer
hardware) triages a subreddit's posts into signal vs. noise and **writes a
personalized explainer article** for each post worth reading — grounded in the
post, its comment thread, and a profile of what the reader already knows.
Results land in Postgres and are served as a reading queue by a small Flask app.

This is not a summarizer pipeline: the model **decides** (keep/skip, with a
stated reason), **explains** (rewrites the discussion into a 300–700-word
piece, defining unfamiliar terms and correcting factual errors made by
commenters), and **cross-references** earlier pieces from the same run.

```
Arctic Shift API ──► fetch_posts.py ──► Postgres (posts, comments)
                                            │
                          curate_readings.py│  batch, resumable
                                            ▼
                     LocalAI / llama.cpp (Qwen3.6-35B-A3B, Vulkan)
                     8 GB VRAM + mmap'd RAM offload, 8K context
                                            │
                                            ▼
                              Postgres (readings) + daily digest .md
                                            │
                                app.py (Flask) ──► /readings queue UI
```

## What's interesting in here

**Batch LLM pipeline built for cheap hardware** — the model is a 35B MoE
(3B active params) split across an 8 GB RX 6600 XT and memory-mapped system
RAM via llama.cpp's Vulkan backend. Measured during a full run: ~5 tokens/s,
peak 7.9 GB system RAM on a 16 GB machine, ~9.5 min per post including the
model's thinking pass. Speed is traded for quality deliberately: the run is
designed to happen overnight, unattended.

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
infinite scroll, a read-through comment cache filled from Arctic Shift on
first view, markdown-rendered articles, unread-first ordering, and
mark-as-read. Skipped posts stay inspectable with the model's reason, so the
curator's judgment can be audited.

## Stack

Python · Flask · PostgreSQL (psycopg3) · LocalAI (llama.cpp, Vulkan) ·
Qwen3.6-35B-A3B GGUF · Arctic Shift API · Docker

## Run it

```bash
createdb reddit && psql reddit < schema.sql
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                        # point at your Postgres + LLM server

.venv/bin/python fetch_posts.py --subreddit singularity --limit 500
.venv/bin/python curate_readings.py         # the overnight run (resumable)
.venv/bin/python app.py                     # http://localhost:8010/readings
```

Any OpenAI-compatible server works for the LLM (`LOCALAI_URL`); the reader
profile and editorial rules live in the `SYSTEM` prompt in
`curate_readings.py` — edit them to curate for someone who isn't me.

## Numbers from a real run

106 posts from r/singularity, one night on a Ryzen 3 3100 + RX 6600 XT (8 GB):
**~20 SIGNAL articles written, ~65 posts skipped with reasons, 0 lost to
errors** after the resilience fixes. Total cost: one idle desktop and some
electricity.
