-- Semantic similarity between posts (see embeddings.py). Needs the pgvector
-- extension compiled into the server, e.g. the pgvector/pgvector:pg16 image.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS posts (
    id           text PRIMARY KEY,          -- Reddit's base36 post id, e.g. "1dxyz9"
    subreddit    text NOT NULL,
    title        text NOT NULL,
    selftext     text NOT NULL,             -- the body of a text post
    author       text,                      -- NULL if the account was deleted
    created_utc  timestamptz NOT NULL,      -- when the post was made
    score        integer NOT NULL,
    num_comments integer NOT NULL,
    permalink    text NOT NULL,
    fetched_at   timestamptz NOT NULL DEFAULT now(),
    comments_fetched_at timestamptz,        -- NULL = comments never fetched
    embedding    vector(768),               -- nomic-embed-text of title+body;
                                            -- NULL until embeddings.py backfills
    url          text,                      -- link posts: where the post points
    link_text    text,                      -- extracted article text ('' if none)
    link_fetched_at timestamptz             -- NULL = article never fetched
);

CREATE INDEX IF NOT EXISTS posts_created_utc_idx ON posts (created_utc DESC);
-- No ANN index on embedding: exact scan is fast up to ~100K rows; add
-- `USING hnsw (embedding vector_cosine_ops)` if the table ever gets there.

CREATE TABLE IF NOT EXISTS comments (
    id          text PRIMARY KEY,            -- Reddit's base36 comment id
    post_id     text NOT NULL REFERENCES posts (id),
    parent_id   text,                        -- NULL = top-level, else the parent comment's id
    author      text,
    body        text NOT NULL,
    created_utc timestamptz NOT NULL,
    score       integer NOT NULL,
    fetched_at  timestamptz NOT NULL DEFAULT now()
);

-- We always load comments per post, so index the foreign key.
CREATE INDEX IF NOT EXISTS comments_post_id_idx ON comments (post_id);

-- LLM curation output: one verdict per post; SIGNAL rows carry a generated
-- reading piece in markdown. (curate_readings.py also creates this on start.)
CREATE TABLE IF NOT EXISTS readings (
    post_id    text PRIMARY KEY REFERENCES posts (id),
    verdict    text NOT NULL,           -- SIGNAL | SKIP | ERROR
    reason     text,
    article    text,                    -- markdown, NULL for SKIP
    model      text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    read_at    timestamptz              -- set when marked read in the UI
);

-- Per-post conversation with the curator model, shown under the reading
-- (app.py /readings/<id>/chat). The model can revise readings.article from
-- here, so the chat is the piece's edit history in prose form.
CREATE TABLE IF NOT EXISTS chat_messages (
    id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    post_id    text NOT NULL REFERENCES posts (id),
    role       text NOT NULL,           -- user | assistant
    content    text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chat_messages_post_id_idx ON chat_messages (post_id, id);

-- Free-text reader notes on a post ("wrong call — I wanted this one",
-- "loved the depth here"). The curator retrieves notes attached to
-- semantically similar judged posts and uses them to calibrate future
-- verdicts — this is what makes the profile learn instead of staying static.
CREATE TABLE IF NOT EXISTS feedback (
    id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    post_id    text NOT NULL REFERENCES posts (id),
    content    text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS feedback_post_id_idx ON feedback (post_id, id);
