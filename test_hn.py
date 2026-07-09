"""Unit tests for the Hacker News ingest (fetch_hn.py).

The Algolia payloads are mapped to the same posts/comments rows the Reddit
ingest produces — these tests pin the mapping: id prefixing, HTML
flattening, media policy reuse, and the comment-tree walk.

Run: .venv/bin/pytest
"""

from fetch_hn import HN_UPSERT, flatten_comments, html_to_text, story_row


# --- html_to_text: Algolia HTML fragments -> plain text ---------------------

def test_tags_are_dropped_and_entities_decoded():
    assert html_to_text("I &amp; my <i>opinions</i>") == "I & my opinions"

def test_paragraphs_become_blank_lines():
    out = html_to_text("first<p>second")
    assert out == "first\n\nsecond"

def test_none_is_empty():
    assert html_to_text(None) == ""


# --- story_row: Algolia hit -> posts UPSERT row ------------------------------

def _hit(**kw):
    hit = {"objectID": "41000000", "title": "Show HN: my thing",
           "url": "https://example.com/post", "story_text": None,
           "author": "pg", "created_at_i": 1751000000,
           "points": 120, "num_comments": 45}
    hit.update(kw)
    return hit

def test_link_story_maps_to_prefixed_row():
    row = story_row(_hit())
    assert row[0] == "hn_41000000"
    assert row[1] == "hackernews"
    assert row[8] == "https://news.ycombinator.com/item?id=41000000"
    assert row[9] == "https://example.com/post"

def test_text_story_keeps_flattened_text():
    row = story_row(_hit(url=None, story_text="ask <b>HN</b>: how?"))
    assert row[3] == "ask HN: how?"
    assert row[9] is None

def test_media_story_is_skipped():
    assert story_row(_hit(url="https://i.imgur.com/x.png")) is None

def test_dead_story_is_skipped():
    assert story_row(_hit(url=None, story_text=None)) is None

def test_hn_upsert_stamps_source():
    assert "source" in HN_UPSERT and "'hn'" in HN_UPSERT


# --- flatten_comments: item tree -> comments rows ---------------------------

def _comment(cid, text, children=()):
    return {"id": cid, "author": "u", "text": text,
            "created_at_i": 1751000100, "children": list(children)}

def test_tree_is_flattened_with_parent_pointers():
    item = {"id": 41000000, "children": [
        _comment(1, "top", [_comment(2, "reply")]),
        _comment(3, "other top"),
    ]}
    rows = flatten_comments(item)
    by_id = {r[0]: r for r in rows}
    assert set(by_id) == {"hn_1", "hn_2", "hn_3"}
    assert by_id["hn_1"][2] is None          # top-level: no parent
    assert by_id["hn_2"][2] == "hn_1"        # reply points at its parent
    assert all(r[1] == "hn_41000000" for r in rows)

def test_deleted_comments_are_dropped_but_children_kept():
    item = {"id": 9, "children": [
        _comment(1, None, [_comment(2, "orphan survives")]),
    ]}
    rows = flatten_comments(item)
    assert [r[0] for r in rows] == ["hn_2"]
