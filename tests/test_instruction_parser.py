import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdp.perception.instruction_parser import parse_instruction


def test_parse_downloaded_template():
    text = "unlocked_waist: pick the squash from the cutting board and place it in the basket"
    factors = parse_instruction(text)
    assert [factor.factor_text for factor in factors] == [
        "the squash",
        "the cutting board",
        "the basket",
    ]
    assert [factor.entity_text for factor in factors] == [factor.factor_text for factor in factors]
    assert [text[factor.char_start : factor.char_end] for factor in factors] == [
        factor.factor_text for factor in factors
    ]


def test_parse_instruction_variation():
    text = "Move the red cup from the table onto the shelf."
    factors = parse_instruction(text)
    assert [factor.factor_text for factor in factors] == [
        "the red cup", "the table", "the shelf"
    ]
    assert [factor.entity_text for factor in factors] == [factor.factor_text for factor in factors]
    assert [factor.factor_id for factor in factors] == [0, 1, 2]
    assert [text[factor.char_start : factor.char_end] for factor in factors] == [
        factor.factor_text for factor in factors
    ]


def test_parse_filters_pronouns_and_preserves_repeated_entities():
    factors = parse_instruction("Take the cup from the plate and put it on the plate.")
    assert [factor.factor_text for factor in factors] == ["the cup", "the plate", "the plate"]
    assert [factor.entity_text for factor in factors] == [factor.factor_text for factor in factors]


def test_parse_filters_abstract_nouns():
    factors = parse_instruction("Pick the cup with care and place it in the bowl with caution.")
    assert [factor.factor_text for factor in factors] == ["the cup", "the bowl"]
    assert [factor.entity_text for factor in factors] == [factor.factor_text for factor in factors]


def test_parse_preserves_attributes_and_location():
    text = "Pick the red wooden cup on the left from the table."
    factors = parse_instruction(text)
    assert [factor.factor_text for factor in factors] == ["the red wooden cup on the left", "the table"]
    assert [text[factor.char_start : factor.char_end] for factor in factors] == [
        factor.factor_text for factor in factors
    ]


def test_parse_extracts_all_factors():
    factors = parse_instruction("Look at the lamp, cup, plate, basket and shelf.")
    assert [factor.factor_text for factor in factors] == ["the lamp", "cup", "plate", "basket", "shelf"]
