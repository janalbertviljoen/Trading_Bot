"""LaTeX escaping of the report text (skatkis/report.py)."""

import pytest

from Toolbox.skatkis.report import escape


def test_plain_text_special_characters_are_escaped() -> None:
    assert escape("a & b") == "a \\& b"
    assert escape("50% of #1") == "50\\% of \\#1"


def test_bold_becomes_textbf_with_escaping_inside() -> None:
    assert escape("**R&D 5%**") == "\\textbf{R\\&D 5\\%}"


def test_math_is_left_alone() -> None:
    assert escape("the ratio $a_1 & b$ here") == "the ratio $a_1 & b$ here"


def test_label_becomes_ref() -> None:
    assert escape("see [[tab:picks]]") == "see \\ref{tab:picks}"


def test_mixed_text() -> None:
    out = escape("**Bold & co** has $x^2$ in [[sec-a]], 5%.")
    assert out == "\\textbf{Bold \\& co} has $x^2$ in \\ref{sec-a}, 5\\%."


def test_unmatched_bold_marker_is_plain_text() -> None:
    out = escape("a ** b")
    assert "\\textbf" not in out
    assert "**" in out


def test_empty_bold_marker_is_plain_text() -> None:
    out = escape("x **** y")
    assert "\\textbf" not in out


@pytest.mark.parametrize("text", ["", "plain"])
def test_text_without_markup_is_unchanged(text: str) -> None:
    assert escape(text) == text
