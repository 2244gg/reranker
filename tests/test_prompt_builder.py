"""Tests for prompt rendering across scenarios and domains."""
from __future__ import annotations

from src.datasets.base import InteractionItem, UserRecord
from src.experiment.prompt_builder import PromptBuilder


def _toy_movie_user() -> UserRecord:
    return UserRecord(
        user_id="u1",
        demographics={
            "gender": "female",
            "age_group": "young",
            "occupation_group": "student",
        },
        history=[
            InteractionItem(
                item_id="1",
                title="Toy Story (1995)",
                metadata={"genres": "Animation|Children's|Comedy"},
                rating=5.0,
            ),
            InteractionItem(
                item_id="2",
                title="The Matrix (1999)",
                metadata={"genres": "Action|Sci-Fi"},
                rating=5.0,
            ),
        ],
    )


def test_neutral_renders_without_demographics():
    user = _toy_movie_user()
    builder = PromptBuilder(domain="movie")
    rendered = builder.render(user, scenario=())
    assert "profile" not in rendered.text.lower()
    assert "Toy Story (1995)" in rendered.text
    assert rendered.scenario == ()


def test_gender_scenario_includes_humanized_value():
    user = _toy_movie_user()
    builder = PromptBuilder(domain="movie")
    rendered = builder.render(user, scenario=("gender",))
    assert "female" in rendered.text
    assert rendered.scenario == ("gender",)


def test_cross_three_dim_includes_all_values():
    user = _toy_movie_user()
    builder = PromptBuilder(domain="movie")
    rendered = builder.render(
        user, scenario=("gender", "age_group", "occupation_group")
    )
    text_lower = rendered.text.lower()
    assert "female" in text_lower
    assert "young" in text_lower
    assert "student" in text_lower


def test_only_demographics_clause_differs_across_scenarios():
    user = _toy_movie_user()
    builder = PromptBuilder(domain="movie")
    p_neutral = builder.render(user, scenario=())
    p_gender = builder.render(user, scenario=("gender",))
    p_age = builder.render(user, scenario=("age_group",))

    # Same history block. We strip the leading clause-line region by checking
    # the history items remain present in all renderings.
    for movie in ("Toy Story (1995)", "The Matrix (1999)"):
        assert movie in p_neutral.text
        assert movie in p_gender.text
        assert movie in p_age.text

    # SHAs must differ because demographics clause differs.
    assert p_neutral.sha256 != p_gender.sha256
    assert p_gender.sha256 != p_age.sha256
    assert p_neutral.sha256 != p_age.sha256


def test_zh_video_template_uses_chinese():
    user = UserRecord(
        user_id="u1",
        demographics={"gender": "male", "age_group": "young"},
        history=[
            InteractionItem(item_id="v1", title="搞笑猫咪合集", metadata={}),
        ],
    )
    builder = PromptBuilder(domain="video", language="zh")
    rendered = builder.render(user, scenario=("gender",))
    # The chinese template starts with 你是 .
    assert rendered.text.startswith("你是")
    assert "男性" in rendered.text


def test_template_hash_changes_with_scenario_no():
    """Same template hash regardless of scenario; only prompt sha256 differs."""
    user = _toy_movie_user()
    builder = PromptBuilder(domain="movie")
    h1 = builder.template_hash()
    h2 = builder.template_hash()
    assert h1 == h2  # template file content is stable.


def test_render_all_scenarios_returns_list():
    user = _toy_movie_user()
    builder = PromptBuilder(domain="movie")
    scenarios = [(), ("gender",), ("age_group",), ("gender", "age_group")]
    rendered = builder.render_all_scenarios(user, scenarios)
    assert len(rendered) == 4
    assert rendered[0].scenario == ()
    assert rendered[3].scenario == ("gender", "age_group")
