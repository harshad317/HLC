from __future__ import annotations

import re
import unicodedata
from typing import Optional


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_or_alias_match(generation: str, answer: str, aliases: Optional[list[str]] = None) -> bool:
    generation_norm = normalize_text(generation)
    candidates = [answer] + list(aliases or [])
    for candidate in candidates:
        candidate_norm = normalize_text(candidate)
        if candidate_norm and candidate_norm in generation_norm:
            return True
    return False


def refusal_match(generation: str, neutral_answer: str = "I don't know.") -> bool:
    generation_norm = normalize_text(generation)
    neutral_norm = normalize_text(neutral_answer)
    refusal_terms = ["i do not know", "i dont know", "unknown", "cannot answer", "not sure"]
    return neutral_norm in generation_norm or any(term in generation_norm for term in refusal_terms)
