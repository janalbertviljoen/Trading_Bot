"""The model fingerprint and the static-figures notice (value_investing/__main__.py)."""

import copy
import re
from pathlib import Path
from typing import Any

import pytest

from Toolbox.skatkis.io.config import load_config
from Toolbox.value_investing.__main__ import _static_note, model_fingerprint
from Toolbox.value_investing.rules import RULES

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RULES = ["G6", "G2a", "G7", "G2b", "E2", "NN", "E1", "G5", "B1", "E4"]


@pytest.fixture
def rules() -> dict[str, Any]:
    return {
        "rules": {"G6": {"max_pe": 15.0}, "B1": {"years": 3}},
        "score": {"min_available": 5, "buckets": 6},
        "valuation": {"average_years": 3},
        "schedule": {"months": [3, 6, 9, 12], "history_years": 13},
        "wallet": {"names": 10, "book_usd": 100.0},
        "report": {"sell_threshold": 0.4},
    }


def test_fingerprint_is_twelve_hex_characters_and_stable(rules: dict[str, Any]) -> None:
    first = model_fingerprint(rules)
    assert re.fullmatch(r"[0-9a-f]{12}", first)
    assert model_fingerprint(copy.deepcopy(rules)) == first


def test_fingerprint_changes_with_names_threshold_score_and_valuation(
    rules: dict[str, Any],
) -> None:
    base = model_fingerprint(rules)
    changes = [
        ("wallet", "names", 5),
        ("score", "buckets", 5),
        ("valuation", "average_years", 5),
    ]
    seen = {base}
    for section, key, value in changes:
        changed = copy.deepcopy(rules)
        changed[section][key] = value
        seen.add(model_fingerprint(changed))
    changed = copy.deepcopy(rules)
    changed["rules"]["G6"]["max_pe"] = 12.0
    seen.add(model_fingerprint(changed))
    assert len(seen) == 5


def test_fingerprint_ignores_report_and_other_wallet_settings(
    rules: dict[str, Any],
) -> None:
    base = model_fingerprint(rules)
    changed = copy.deepcopy(rules)
    changed["report"]["sell_threshold"] = 0.9
    changed["wallet"]["book_usd"] = 5.0
    assert model_fingerprint(changed) == base


def _text(rules: dict[str, Any], data_as_of: str = "2026-10-31") -> dict[str, Any]:
    return {
        "static_figures": {
            "produced": "2026-11-01",
            "model_fingerprint": model_fingerprint(rules),
            "data_as_of": data_as_of,
        }
    }


def test_static_note_is_empty_when_stamp_matches(
    rules: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert _static_note(rules, _text(rules), "2026-10-31") == ""
    assert capsys.readouterr().out == ""


def test_static_note_warns_when_the_model_changed(
    rules: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    text = _text(rules)
    rules["wallet"]["names"] = 7
    note = _static_note(rules, text, "2026-10-31")
    assert note != ""
    assert "WARNING" in capsys.readouterr().out


def test_static_note_warns_when_the_data_date_differs(
    rules: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    note = _static_note(rules, _text(rules), "2027-01-30")
    assert note != ""
    assert "WARNING" in capsys.readouterr().out


def test_default_config_has_exactly_the_ten_default_rules() -> None:
    config = load_config(REPO / "config" / "rules.toml")
    assert set(config["rules"]) == set(DEFAULT_RULES)
    assert len(config["rules"]) == 10
    for rule_id in config["rules"]:
        assert rule_id in RULES
