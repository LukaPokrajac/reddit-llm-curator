"""Unit tests for the ingest policy in fetch_posts.py.

is_media_url decides which link posts are stored at all — a wrong "media"
call silently drops real articles, a wrong "not media" call queues unjudgeable
image posts for ~10 minutes of GPU each.

Run: .venv/bin/pytest
"""

from fetch_posts import is_media_url


def test_reddit_image_host_is_media():
    assert is_media_url("https://i.redd.it/abc123.jpeg")

def test_reddit_video_host_is_media():
    assert is_media_url("https://v.redd.it/abc123")

def test_imgur_is_media():
    assert is_media_url("https://imgur.com/gallery/abc")
    assert is_media_url("https://i.imgur.com/abc.gifv")

def test_reddit_gallery_is_media():
    assert is_media_url("https://www.reddit.com/gallery/1abcde")

def test_image_extension_anywhere_is_media():
    assert is_media_url("https://example.com/photos/robot.png")
    assert is_media_url("https://example.com/clip.mp4?src=share")

def test_news_article_is_not_media():
    assert not is_media_url("https://arstechnica.com/ai/2026/07/some-story/")

def test_youtube_is_not_media():
    # deliberate: video posts keep their title + discussion thread
    assert not is_media_url("https://www.youtube.com/watch?v=abc123")

def test_query_string_does_not_hide_extension():
    assert is_media_url("https://cdn.example.com/x.webp?w=1200&fmt=auto")

def test_extension_in_query_only_is_not_media():
    assert not is_media_url("https://example.com/article?img=x.png")
