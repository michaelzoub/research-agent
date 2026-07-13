"""Claim-level citation checks backed by retained document evidence."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence


URL_RE = re.compile(r"https?://[^\s)\]>]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
WORDS = re.compile(r"[a-zA-Z][a-zA-Z0-9-]{2,}")
STOP = {"about", "after", "also", "because", "between", "could", "from", "have", "into", "more", "only", "over", "same", "than", "that", "their", "there", "these", "this", "those", "through", "using", "with"}


@dataclass(frozen=True)
class CitationCheck:
    claim: str
    urls: list[str]
    source_ids: list[str]
    support: float
    locators: list[dict[str, Any]]
    passed: bool
    reason: str = ""


def validate_claim_citations(answer: str, sources: Sequence[dict[str, Any]], *, min_support: float = 0.20) -> list[CitationCheck]:
    by_url = {_canonical(str(source.get("url") or "")): source for source in sources}
    checks: list[CitationCheck] = []
    chunks: list[str] = []
    for raw in SENTENCE_RE.split(answer):
        # Markdown citations commonly follow a completed sentence, e.g.
        # "Claim. https://source".  Attach that citation-only fragment to the
        # preceding claim rather than judging it as a separate sentence.
        if URL_RE.search(raw) and len(_terms(URL_RE.sub("", raw))) < 2 and chunks:
            chunks[-1] = chunks[-1].rstrip() + " " + raw.lstrip()
        else:
            chunks.append(raw)
    for raw in chunks:
        urls = [_canonical(value) for value in URL_RE.findall(raw)]
        claim = URL_RE.sub("", raw).strip(" -:[]()")
        if not _is_substantive_claim(claim):
            continue
        if not urls:
            checks.append(CitationCheck(claim, [], [], 0.0, [], False, "claim has no inline citation"))
            continue
        matched = [by_url[url] for url in urls if url in by_url]
        if not matched:
            checks.append(CitationCheck(claim, urls, [], 0.0, [], False, "citation was not retrieved"))
            continue
        verified = [source for source in matched if source.get("evidence_kind") == "verified_document"]
        if not verified:
            checks.append(CitationCheck(claim, urls, [str(source.get("id")) for source in matched], 0.0, [], False, "search leads cannot ground final claims"))
            continue
        support, locators = _support(claim, verified)
        checks.append(CitationCheck(claim, urls, [str(source.get("id")) for source in verified], support, locators, support >= min_support, "" if support >= min_support else "citation does not support the claim"))
    return checks


def coverage(checks: Sequence[CitationCheck]) -> float:
    return sum(check.passed for check in checks) / max(len(checks), 1)


def _support(claim: str, sources: Sequence[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    terms = _terms(claim)
    if not terms:
        return 1.0, []
    best, best_locators = 0.0, []
    for source in sources:
        sections = source.get("evidence_sections") or {}
        locators = source.get("evidence_locators") or {}
        for name, text in sections.items():
            evidence_terms = _terms(str(text))
            score = len(terms & evidence_terms) / len(terms)
            if score > best:
                best = score
                best_locators = list(locators.get(name) or [])
    return round(best, 3), best_locators


def _is_substantive_claim(value: str) -> bool:
    lowered = value.lower()
    return len(_terms(value)) >= 4 and not lowered.startswith(("sources:", "evidence basis:", "citation:"))


def _terms(value: str) -> set[str]:
    return {item.lower() for item in WORDS.findall(value) if item.lower() not in STOP}


def _canonical(url: str) -> str:
    return url.rstrip(".,;:!?)]}")
