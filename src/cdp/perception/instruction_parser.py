"""Extract grounded instruction phrases with spaCy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import spacy
from nltk.corpus import wordnet


_LOCATION_NOUNS = {"left", "right", "front", "back", "middle", "center", "centre", "top", "bottom"}
_LOCATION_PREPOSITIONS = {"on", "in", "at"}


@dataclass(frozen=True)
class FactorCandidate:
    factor_id: int
    factor_text: str
    entity_text: str
    char_start: int
    char_end: int
    dependency_metadata: str

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def _load_parser():
    return spacy.load("en_core_web_sm", disable=["ner"])


@lru_cache(maxsize=4096)
def _is_concrete_noun(lemma: str) -> bool:
    senses = wordnet.synsets(lemma, pos=wordnet.NOUN)
    physical = wordnet.synset("physical_entity.n.01")
    return not senses or any(
        sense == physical or physical in sense.closure(lambda item: item.hypernyms())
        for sense in senses
    )


def _phrase_root(phrase):
    if phrase.root.pos_ in {"NOUN", "PROPN"}:
        return phrase.root
    return next((token for token in reversed(phrase) if token.pos_ in {"NOUN", "PROPN"}), phrase.root)


def _is_location_phrase(phrase):
    root = _phrase_root(phrase)
    return root.lemma_ in _LOCATION_NOUNS and root.head.pos_ == "ADP"


def _noun_phrases(doc):
    phrases = []
    for chunk in doc.noun_chunks:
        start = chunk.start
        for token in chunk:
            if token.text == "," or token.pos_ == "CCONJ":
                if start < token.i:
                    phrases.append(doc[start:token.i])
                start = token.i + 1
        if start < chunk.end:
            phrases.append(doc[start:chunk.end])
    return phrases


def _located_phrase(doc, phrase, noun_phrases):
    end = phrase.end
    for modifier in noun_phrases:
        if not _is_location_phrase(modifier):
            continue
        modifier_root = _phrase_root(modifier)
        preposition = modifier_root.head
        if preposition.i == end and preposition.lemma_ in _LOCATION_PREPOSITIONS:
            end = modifier.end
            for reference in noun_phrases:
                reference_root = _phrase_root(reference)
                if reference_root.head.lemma_ == "of" and reference_root.head.head == modifier_root:
                    end = max(end, reference.end)
    for child in _phrase_root(phrase).children:
        if child.dep_ == "prep" and child.lemma_ not in {"from", "to"}:
            end = max(end, max(token.i for token in child.subtree) + 1)
    return doc[phrase.start:end]


def parse_instruction(instruction: str) -> list[FactorCandidate]:
    prefix = "unlocked_waist:"
    offset = len(prefix) if instruction.startswith(prefix) else 0
    doc = _load_parser()(instruction[offset:])
    noun_phrases = _noun_phrases(doc)

    factors = []
    for phrase in noun_phrases:
        root = _phrase_root(phrase)
        if root.pos_ not in {"NOUN", "PROPN"}:
            continue
        if _is_location_phrase(phrase) or not _is_concrete_noun(root.lemma_.lower()):
            continue
        span = _located_phrase(doc, phrase, noun_phrases)
        factors.append(
            FactorCandidate(
                factor_id=len(factors),
                factor_text=span.text,
                entity_text=span.text,
                char_start=offset + span.start_char,
                char_end=offset + span.end_char,
                dependency_metadata=root.dep_,
            )
        )
    return factors
