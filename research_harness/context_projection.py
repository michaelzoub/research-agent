"""Lossless working-state projection for long grader trajectories.

The append-only transcript remains the audit authority.  This module derives a
smaller model-facing context from that transcript without asking a model to
paraphrase exact candidate code or official measurements.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .agent_state import AgentState, canonical_citation_url


GRADER_TOOL = "evaluate_prediction_market_candidate"


@dataclass(frozen=True)
class ContextProjection:
    messages: list[dict[str, Any]]
    audit_message_count: int
    projected_message_count: int
    grader_trial_count: int
    fetched_document_count: int


@dataclass(frozen=True)
class GraderTrialState:
    candidate_id: str
    code: str
    rationale: str
    status: str
    official_measured: bool
    score_eligible: bool
    mean_edge: Any
    mean_arb_edge: Any
    mean_retail_edge: Any
    promoted: bool
    trial_path: str
    error: str


class WorkingStateProjector:
    """Build the next model request from durable state, not transcript replay."""

    def __init__(self, *, max_literature_characters: int = 8_000, max_documents: int = 12):
        self.max_literature_characters = max_literature_characters
        self.max_documents = max_documents

    def project(self, state: AgentState, *, max_grader_calls: Optional[int]) -> ContextProjection:
        audit = state.messages
        if state.current_iteration <= 1 or not max_grader_calls:
            messages = list(audit)
            return ContextProjection(messages, len(audit), len(messages), 0, 0)

        prefix = _initial_messages(audit)
        recent = _latest_unresolved_exchange(audit)
        trials = _grader_trials(state)
        literature = _fetched_literature(
            state.sources,
            max_characters=self.max_literature_characters,
            max_documents=self.max_documents,
        )
        recent_codes = {
            str(call.get("arguments", {}).get("code") or "")
            for message in recent
            for call in message.get("tool_calls") or []
            if call.get("name") == GRADER_TOOL
        }
        checkpoint = _checkpoint_message(
            trials,
            literature,
            recent_codes=recent_codes,
            max_grader_calls=max_grader_calls,
        )
        messages = [*prefix, checkpoint, *recent]
        return ContextProjection(
            messages=messages,
            audit_message_count=len(audit),
            projected_message_count=len(messages),
            grader_trial_count=len(trials),
            fetched_document_count=len(literature),
        )


def _initial_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    prefix: list[dict[str, Any]] = []
    user_seen = False
    for message in messages:
        role = str(message.get("role") or "")
        if role == "system" and not user_seen:
            prefix.append(message)
            continue
        if role == "user" and not user_seen:
            prefix.append(message)
            user_seen = True
            break
        break
    return prefix


def _latest_unresolved_exchange(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return [_compact_recent_message(message) for message in messages[index:]]
    return []


def _compact_recent_message(message: dict[str, Any]) -> dict[str, Any]:
    compact = copy.deepcopy(message)
    if compact.get("role") != "tool" or not isinstance(compact.get("content"), dict):
        return compact
    content = compact["content"]
    data = content.get("data") if isinstance(content.get("data"), dict) else None
    if data is None:
        return compact
    name = str(compact.get("name") or "")
    if name == "fetch_document":
        sections = data.get("evidence_sections") if isinstance(data.get("evidence_sections"), dict) else {}
        excerpt = "\n".join(f"[{key}] {value}" for key, value in sections.items())[:2_000]
        content["data"] = {
            key: data.get(key)
            for key in ("url", "document_type", "content_type", "cached", "truncated", "renderer")
            if data.get(key) is not None
        }
        content["data"]["excerpt"] = excerpt or str(data.get("content") or "")[:2_000]
        content["data"]["evidence_locators"] = data.get("evidence_locators") or {}
    elif "search" in name and isinstance(data.get("results"), list):
        data["results"] = [
            {
                **row,
                "summary": str(row.get("summary") or "")[:600],
            }
            for row in data["results"]
            if isinstance(row, dict)
        ]
    return compact


def _grader_trials(state: AgentState) -> list[GraderTrialState]:
    results: dict[tuple[str, str], list[Any]] = {}
    for message in state.messages:
        if message.get("role") != "tool":
            continue
        key = (str(message.get("tool_call_id") or ""), str(message.get("name") or ""))
        results.setdefault(key, []).append(message.get("content"))
    records: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in state.tool_calls:
        key = (str(record.get("id") or ""), str(record.get("tool") or ""))
        records.setdefault(key, []).append(record)
    trials: list[GraderTrialState] = []
    for message in state.messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if call.get("name") != GRADER_TOOL:
                continue
            call_id = str(call.get("id") or "")
            key = (call_id, GRADER_TOOL)
            arguments = call.get("arguments") or {}
            raw_observation = results.get(key, []).pop(0) if results.get(key) else {}
            observation = raw_observation if isinstance(raw_observation, dict) else {}
            data = observation.get("data") if isinstance(observation.get("data"), dict) else {}
            record = records.get(key, []).pop(0) if records.get(key) else {}
            promotion = data.get("promotion") if isinstance(data.get("promotion"), dict) else {}
            trials.append(GraderTrialState(
                candidate_id=str(data.get("candidate_id") or call_id),
                code=str(arguments.get("code") or ""),
                rationale=str(arguments.get("rationale") or ""),
                status=str(observation.get("status") or record.get("status") or "unknown"),
                official_measured=bool(record.get("official_measured", data.get("official_measured", False))),
                score_eligible=bool(data.get("score_eligible", record.get("official_measured", False))),
                mean_edge=data.get("mean_edge"),
                mean_arb_edge=data.get("mean_arb_edge"),
                mean_retail_edge=data.get("mean_retail_edge"),
                promoted=bool(promotion.get("promoted_this_round", False)),
                trial_path=str(data.get("trial_path") or ""),
                error=str(observation.get("error") or data.get("failure") or record.get("error") or ""),
            ))
    return trials


def _fetched_literature(
    sources: Sequence[dict[str, Any]], *, max_characters: int, max_documents: int
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    remaining = max_characters
    for source in sources:
        if source.get("evidence_kind") != "verified_document" or remaining <= 0:
            continue
        url = str(source.get("url") or "")
        canonical = canonical_citation_url(url)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        sections = source.get("evidence_sections") if isinstance(source.get("evidence_sections"), dict) else {}
        excerpt = "\n".join(
            f"[{name}] {value}" for name, value in sections.items() if str(value).strip()
        ) or str(source.get("summary") or "")
        excerpt = excerpt[: min(1_500, remaining)]
        remaining -= len(excerpt)
        documents.append({
            "title": str(source.get("title") or url),
            "url": url,
            "excerpt": excerpt,
            "locators": source.get("evidence_locators") or {},
        })
        if len(documents) >= max_documents:
            break
    return documents


def _checkpoint_message(
    trials: Sequence[GraderTrialState],
    literature: Sequence[dict[str, Any]],
    *,
    recent_codes: set[str],
    max_grader_calls: int,
) -> dict[str, Any]:
    eligible = [trial for trial in trials if trial.official_measured and trial.score_eligible]
    champion = max(eligible, key=lambda trial: float(trial.mean_edge or 0.0), default=None)
    latest = trials[-1] if trials else None
    lines = [
        "## Harness working-state checkpoint",
        "This deterministic checkpoint replaces older conversational history. Exact artifacts remain in the audit log.",
        f"Official evaluations: {len(eligible)}/{max_grader_calls} completed.",
        f"{max(0, max_grader_calls - len(eligible))} requested official evaluation(s) remain.",
    ]
    if trials:
        lines.extend(["", "### Strategy ledger", "candidate | status | mean_edge | mean_arb_edge | mean_retail_edge | promoted | rationale | artifact"])
        for trial in trials:
            rationale = " ".join(trial.rationale.split())[:240]
            error = " ".join(trial.error.split())[:160]
            status = "official" if trial.official_measured and trial.score_eligible else f"{trial.status} (not eligible)"
            note = rationale or error
            lines.append(
                f"{trial.candidate_id} | {status} | {_metric(trial.mean_edge)} | {_metric(trial.mean_arb_edge)} | "
                f"{_metric(trial.mean_retail_edge)} | {'yes' if trial.promoted else 'no'} | {note} | {trial.trial_path}"
            )
    if champion is not None and champion.code and champion.code not in recent_codes:
        lines.extend(["", "### Current champion strategy (exact code)", f"Metrics: edge={_metric(champion.mean_edge)}, arb={_metric(champion.mean_arb_edge)}, retail={_metric(champion.mean_retail_edge)}", "```python", champion.code, "```"])
    if latest is not None and latest is not champion and latest.code and latest.code not in recent_codes:
        lines.extend(["", "### Latest evaluated strategy (exact code)", f"Metrics: edge={_metric(latest.mean_edge)}, arb={_metric(latest.mean_arb_edge)}, retail={_metric(latest.mean_retail_edge)}", "```python", latest.code, "```"])
    if literature:
        lines.extend(["", "### Already fetched literature", "Do not refetch these URLs; use the retained extracts or fetch a different primary document."])
        for document in literature:
            lines.extend([
                f"- {document['title']} — {document['url']}",
                f"  Extract: {' '.join(str(document['excerpt']).split())}",
                f"  Locators: {document['locators']}",
            ])
    else:
        lines.extend(["", "### Already fetched literature", "None."])
    return {"role": "user", "content": "\n".join(lines)}


def _metric(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "—"
