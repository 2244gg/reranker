"""Track-level LastFM-1K adapter: smoke + parser-interaction tests.

These tests do NOT require the 2.5 GB raw events file. They exercise the
public-facing knobs:

  * ``LastFm1KAdapter(granularity=...)`` accepts both modes and reflects
    the choice in ``domain()`` so the prompt builder routes correctly.
  * The ``music_track_en.j2`` template renders without errors.
  * ``parse_recommendations(strip_trailing_metadata=False)`` keeps
    ``"Track - Artist"`` intact (whereas the default behaviour, used for
    artist mode, would strip the artist suffix).
"""
from __future__ import annotations

import pytest

from src.datasets.base import InteractionItem, UserRecord
from src.datasets.factory import build_dataset
from src.datasets.lastfm1k import LastFm1KAdapter, TRACK_KEY_SEP, TRACK_TITLE_SEP
from src.experiment.prompt_builder import PromptBuilder
from src.experiment.runner import parse_recommendations


def test_lastfm1k_artist_mode_default_domain():
    a = build_dataset("lastfm1k")
    assert a.granularity == "artist"
    assert a.domain() == "music"


def test_lastfm1k_track_mode_domain_and_schema():
    a = build_dataset("lastfm1k", granularity="track", include_country=True)
    assert a.granularity == "track"
    assert a.domain() == "music_track"
    assert a.demographic_schema() == ["gender", "age_group", "country"]


def test_lastfm1k_invalid_granularity_raises():
    with pytest.raises(ValueError):
        LastFm1KAdapter(granularity="album")


def test_lastfm1k_track_positive_feedback_predicate():
    a = LastFm1KAdapter(granularity="track")
    # Median = 1: only tracks with playcount >= 2 AND > 1 count as positive.
    item_one = InteractionItem(item_id="x", title="x", rating=1.0,
                               metadata={"user_median_playcount": 1.0})
    item_two = InteractionItem(item_id="x", title="x", rating=2.0,
                               metadata={"user_median_playcount": 1.0})
    item_three = InteractionItem(item_id="x", title="x", rating=10.0,
                                 metadata={"user_median_playcount": 5.0})
    assert a.positive_feedback_predicate(item_one) is False
    assert a.positive_feedback_predicate(item_two) is True
    assert a.positive_feedback_predicate(item_three) is True


def test_music_track_template_renders():
    user = UserRecord(
        user_id="u1",
        demographics={"gender": "female", "age_group": "young"},
        history=[
            InteractionItem(
                item_id=f"The Beatles{TRACK_KEY_SEP}Yesterday",
                title=f"Yesterday{TRACK_TITLE_SEP}The Beatles",
                rating=42.0,
            ),
            InteractionItem(
                item_id=f"Radiohead{TRACK_KEY_SEP}Creep",
                title=f"Creep{TRACK_TITLE_SEP}Radiohead",
                rating=10.0,
            ),
        ],
    )
    pb = PromptBuilder(domain="music_track")
    rendered = pb.render(user, scenario=("gender", "age_group"), movie_count=5)
    assert "Yesterday - The Beatles" in rendered.text
    assert "Creep - Radiohead" in rendered.text
    assert "tracks" in rendered.text.lower()
    assert rendered.template_name == "music_track_en.j2"


def test_parse_recommendations_keeps_track_artist_when_disabled():
    text = (
        "1. Yesterday - The Beatles\n"
        "2. Creep - Radiohead\n"
        "3. Imagine - John Lennon\n"
    )
    full = parse_recommendations(text, strip_trailing_metadata=False)
    assert full == [
        "Yesterday - The Beatles",
        "Creep - Radiohead",
        "Imagine - John Lennon",
    ]


def test_parse_recommendations_default_strips_trailing_dash():
    """Default behaviour (used for movie/book/artist) MUST keep stripping."""
    text = "1. Toy Story (1995) - Animation\n"
    out = parse_recommendations(text, strip_trailing_metadata=True)
    assert out == ["Toy Story (1995)"]
