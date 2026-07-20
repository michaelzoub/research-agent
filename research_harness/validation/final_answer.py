"""Final-answer and citation validation for the research trajectory."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Sequence

from ..agent_state import canonical_citation_url
from ..citation_validation import validate_claim_citations


@dataclass(frozen=True)
class ValidationResult:
    status: Literal["pass", "revise"]
    feedback: str


class FinalAnswerValidator:
    """Validate a proposed answer without controlling the surrounding loop."""

    def validate(self, answer: str, objective: str, sources: Sequence[dict[str, object]]) -> ValidationResult:
        if not answer.strip():
            return ValidationResult("revise", "The final answer was empty. Address the objective directly.")
        if _generic_user_handoff(answer):
            return ValidationResult("revise", "Do not turn an incomplete discovery pass into a generic request for URLs, source permission, or a multiple-choice handoff. Recover with the registered primary-source tools and provide the best grounded answer possible; use needs_input only when a specific missing user fact is necessary.")
        if _objective_requires_external_evidence(objective) and not sources:
            return ValidationResult("revise", "The objective explicitly requested external evidence, but no source was retrieved. Use a working registered search or document tool, or state that external retrieval is unavailable.")
        if sources:
            cited = set(re.findall(r"https?://[^\s)\]>]+", answer))
            known = {canonical_citation_url(str(source.get("url") or "")) for source in sources}
            unsupported = sorted(url for url in cited if canonical_citation_url(url) not in known)
            if unsupported:
                return ValidationResult("revise", "These citations were not retrieved in this run: " + ", ".join(unsupported))
            if not cited:
                return ValidationResult("revise", "Evidence was retrieved. Cite the retrieved source URLs or explicitly state that the evidence was not used.")
            source_by_url = {canonical_citation_url(str(source.get("url") or "")): source for source in sources}
            lead_urls = [url for url in cited if source_by_url.get(canonical_citation_url(url), {}).get("evidence_kind") == "lead"]
            if lead_urls:
                return ValidationResult("revise", "Search snippets are discovery leads, not final evidence. Fetch and cite the underlying document instead: " + ", ".join(sorted(lead_urls)))
            failed = [check for check in validate_claim_citations(answer, sources) if not check.passed]
            if failed:
                return ValidationResult("revise", "Claim-level citation validation failed: " + "; ".join(f"{check.reason} ({check.claim[:90]})" for check in failed[:3]))
        return ValidationResult("pass", "Final answer passed evidence validation.")


def _objective_requires_external_evidence(objective: str) -> bool:
    lowered = objective.lower()
    return any(marker in lowered for marker in ("external source", "external evidence", "use sources", "cite sources", "with sources"))


def _generic_user_handoff(answer: str) -> bool:
    lowered = answer.lower()
    markers = (
        "send 5", "candidate urls", "provide 5", "permission to use",
        "pick one:", "option a", "option b", "tell me an allowed discovery source",
    )
    return sum(marker in lowered for marker in markers) >= 2
