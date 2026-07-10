from __future__ import annotations

import html
import json
import re

from ...store import ArtifactStore
from ..types import EvalTask, GraderResult
from .common import _result
from .dag import dag_grader_result, dag_node, right_wrong, verdict_from_score


TOOL_SOURCE_FAMILIES = {
    "alchemy_blockchain_search": "alchemy",
    "arxiv_api_search": "arxiv",
    "docs_blogs_search": "docs_blogs",
    "github_repo_search": "github",
    "local_corpus_search": "local",
    "openalex_api_search": "openalex",
    "prior_artifact_memory_search": "memory",
    "semantic_scholar_api_search": "semantic_scholar",
    "social_web_search": "social",
    "web_search": "web",
    "wikipedia_search": "wikipedia",
}


def _grade_research_source_diversity(task: EvalTask, store: ArtifactStore) -> GraderResult:
    traces = store.list("agent_traces")
    sources = store.list("sources")
    min_families = int(task.metadata.get("min_distinct_source_families", 4))
    successful_tool_names = _successful_tool_names(traces)
    tool_families = sorted({_tool_source_family(tool_name) for tool_name in successful_tool_names})
    source_families = sorted({_source_family(source) for source in sources})
    passed = len(tool_families) >= min_families and len(source_families) >= min_families
    score = min(1.0, min(len(tool_families), len(source_families)) / max(min_families, 1))
    return _result(
        "research_source_diversity",
        "code",
        "successful tool/API and retained-source family check",
        score,
        passed,
        1.0,
        (
            f"Research successfully called {len(tool_families)} distinct source family/families "
            f"and retained {len(source_families)} source artifact family/families."
        ),
        [
            {
                "check": "min_distinct_successful_tool_sources",
                "actual": len(tool_families),
                "expected_at_least": min_families,
                "families": tool_families,
                "tools": sorted(successful_tool_names),
                "passed": len(tool_families) >= min_families,
            },
            {
                "check": "min_distinct_retained_source_families",
                "actual": len(source_families),
                "expected_at_least": min_families,
                "families": source_families,
                "passed": len(source_families) >= min_families,
            },
        ],
    )


def _successful_tool_names(traces: list[dict[str, object]]) -> set[str]:
    tool_names: set[str] = set()
    for trace in traces:
        calls = trace.get("tool_calls", [])
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            if int(call.get("results", 0) or 0) <= 0:
                continue
            clean = str(call.get("tool", "")).strip()
            if clean:
                tool_names.add(clean)
    return tool_names


def _tool_source_family(tool_name: str) -> str:
    return TOOL_SOURCE_FAMILIES.get(tool_name, tool_name.removesuffix("_search").removesuffix("_api"))


def _source_family(source: dict[str, object]) -> str:
    source_type = str(source.get("source_type", "")).lower()
    url = str(source.get("url", "")).lower()
    if "arxiv" in source_type or "arxiv.org" in url:
        return "arxiv"
    if "openalex" in source_type or "openalex.org" in url:
        return "openalex"
    if "semantic_scholar" in source_type or "semanticscholar.org" in url:
        return "semantic_scholar"
    if "github" in source_type or "github.com" in url:
        return "github"
    if source_type in {"docs_blog", "docs_blogs"}:
        return "docs_blogs"
    if source_type.startswith("wikipedia") or "wikipedia.org" in url:
        return "wikipedia"
    if source_type == "web_result":
        return "web"
    if source_type == "social_web":
        return "social"
    if source_type.startswith("alchemy"):
        return "alchemy"
    if source_type == "prior_artifact_memory":
        return "memory"
    if source_type in {"local_corpus", "paper", "benchmark_report", "systems_note", "challenge_spec"}:
        return "local"
    return source_type or "unknown"


def _grade_report_no_fabricated_sources(task: EvalTask, store: ArtifactStore) -> GraderResult:
    report = store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else ""
    tex_path = getattr(store, "report_tex_path", store.root / "final_report.tex")
    tex = tex_path.read_text(encoding="utf-8") if tex_path.exists() else ""
    sources = store.list("sources")
    known_urls = {str(s.get("url", "")) for s in sources}
    report_urls = _cited_report_urls(report, tex)
    fabricated = []
    is_prediction_market_task = _is_prediction_market_eval_task(task)
    for url in report_urls:
        if url not in known_urls:
            fabricated.append({"url": url, "reason": "not in sources.json"})
        elif not is_prediction_market_task and _is_prediction_market_report_url(url):
            fabricated.append({"url": url, "reason": "prediction-market challenge source in non-challenge report"})
    if not is_prediction_market_task and _references_prediction_market_challenge(report + "\n" + tex):
        fabricated.append({"url": "report text", "reason": "prediction-market challenge reference in non-challenge report"})
    passed = not fabricated
    score = max(0.0, 1.0 - len(fabricated) * 0.25)
    return _result(
        "report_no_fabricated_sources",
        "code",
        "source URL verification",
        score,
        passed,
        1.0,
        f"Found {len(fabricated)} fabricated source URL(s) in report out of {len(report_urls)} cited.",
        [{"check": "no_fabricated_sources", "passed": passed, "fabricated_urls": fabricated}],
    )


def _cited_report_urls(report: str, tex: str) -> list[str]:
    urls = re.findall(r"\]\(([^)]+)\)", report)
    urls.extend(re.findall(r"\\url\{([^}]+)\}", tex))
    deduped = []
    seen = set()
    for url in urls:
        clean = html.unescape(url).strip()
        if clean and clean not in seen:
            seen.add(clean)
            deduped.append(clean)
    return deduped


def _is_placeholder_report_url(url: str) -> bool:
    lowered = url.lower()
    return any(domain in lowered for domain in ["example.org", "example.com", "example.net", "example.invalid"])


def _is_prediction_market_report_url(url: str) -> bool:
    lowered = url.lower().replace("-", "_")
    return (
        "challenges/prediction_market" in lowered
        or "prediction_market/evaluator.py" in lowered
        or "prediction_market/spec.md" in lowered
        or "danrobinson/prediction_market_challenge" in lowered
    )


def _references_prediction_market_challenge(text: str) -> bool:
    lowered = text.lower().replace("-", "_")
    return any(
        phrase in lowered
        for phrase in [
            "prediction market strategy design notes",
            "prediction market local evaluator rubric",
            "orderbook prediction market challenge",
            "challenges/prediction_market",
            "danrobinson/prediction_market_challenge",
        ]
    )


def _is_prediction_market_eval_task(task: EvalTask) -> bool:
    normalized = task.prompt.lower().replace("-", " ")
    return task.evaluator_name == "prediction_market" or ("prediction" in normalized and "market" in normalized)


def _grade_research_groundedness(task: EvalTask, store: ArtifactStore) -> GraderResult:
    sources = store.list("sources")
    claims = store.list("claims")
    grounded_claims = [claim for claim in claims if claim.get("source_ids")]
    source_score = min(1.0, len(sources) / 4)
    claim_score = min(1.0, len(claims) / 8)
    grounded_score = len(grounded_claims) / max(len(claims), 1)
    score = (source_score * 0.3) + (claim_score * 0.3) + (grounded_score * 0.4)
    passed = score >= 0.8
    return _result(
        "research_groundedness",
        "code",
        "groundedness assertions",
        score,
        passed,
        1.25,
        f"{len(sources)} sources, {len(claims)} claims, {len(grounded_claims)} grounded claims.",
        [
            {"check": "min_sources", "actual": len(sources), "expected_at_least": 4, "passed": len(sources) >= 4},
            {"check": "min_claims", "actual": len(claims), "expected_at_least": 8, "passed": len(claims) >= 8},
            {"check": "all_claims_have_sources", "actual": len(grounded_claims), "total": len(claims), "passed": grounded_score == 1.0},
        ],
    )


def _grade_research_task_specific_acceptance(task: EvalTask, store: ArtifactStore) -> GraderResult:
    prompt = task.prompt.lower()
    metadata_kind = str(task.metadata.get("research_acceptance", "")).lower() if isinstance(task.metadata, dict) else ""
    sources = store.list("sources")
    report = store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else ""
    combined = "\n".join(
        [
            report,
            *[str(source.get("url", "")) for source in sources],
            *[str(source.get("title", "")) for source in sources],
            *[str(source.get("source_type", "")) for source in sources],
        ]
    )
    lower = combined.lower()

    needs_paper_id = metadata_kind == "paper_id" or any(term in prompt for term in ["paper", "papers", "doi", "arxiv"])
    needs_dataset = metadata_kind == "dataset_url" or any(term in prompt for term in ["dataset", "datasets", "data source", "data url"])
    needs_benchmark_table = metadata_kind == "benchmark_table" or "benchmark table" in prompt or "leaderboard" in prompt

    checks: list[tuple[str, bool]] = []
    if needs_paper_id:
        checks.append(("found_doi_or_arxiv_id", bool(re.search(r"\b10\.\d{4,9}/[-._;()/:a-z0-9]+\b", lower)) or bool(re.search(r"\barxiv(?:\.org/(?:abs|pdf)/)?[:/\s]?\d{4}\.\d{4,5}", lower))))
    if needs_dataset:
        dataset_domains = ["huggingface.co/datasets", "kaggle.com", "zenodo.org", "figshare.com", "data.gov", "github.com"]
        checks.append(("found_dataset_url", any(domain in lower for domain in dataset_domains) or any("dataset" in str(source.get("source_type", "")).lower() for source in sources)))
    if needs_benchmark_table:
        markdown_table = bool(re.search(r"^\s*\|.+\|\s*$", report, flags=re.M)) and "---" in report
        checks.append(("extracted_benchmark_table", markdown_table or "benchmark" in lower and "score" in lower and "model" in lower))

    if not checks:
        grounded_sources = len(sources) >= 1
        checks.append(("no_specific_research_artifact_requested", grounded_sources))

    score = sum(1 for _, passed in checks if passed) / max(len(checks), 1)
    passed = score == 1.0
    return _result(
        "research_task_specific_acceptance",
        "code",
        "task-specific research artifact verification",
        score,
        passed,
        1.0,
        f"Checked {len(checks)} task-specific research acceptance condition(s).",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_literature_section_evidence(task: EvalTask, store: ArtifactStore) -> GraderResult:
    sources = store.list("sources")
    paper_sources = [source for source in sources if "paper" in str(source.get("source_type", "")).lower() or "work" in str(source.get("source_type", "")).lower()]
    sectioned = []
    for source in paper_sources:
        sections = source.get("evidence_sections") if isinstance(source.get("evidence_sections"), dict) else {}
        present = [name for name in ["abstract", "introduction", "conclusion"] if str(sections.get(name, "")).strip()]
        if present:
            sectioned.append({"source_id": source.get("id"), "title": source.get("title"), "sections": present})
    ratio = len(sectioned) / max(len(paper_sources), 1)
    report = store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else ""
    report_mentions_basis = "evidence basis" in report.lower() or "paper context" in report.lower()
    passed = bool(paper_sources) and ratio >= 0.8 and report_mentions_basis
    score = (ratio * 0.75) + (0.25 if report_mentions_basis else 0.0)
    return _result(
        "literature_section_evidence",
        "code",
        "paper-section evidence verification",
        score,
        passed,
        1.0,
        f"{len(sectioned)}/{len(paper_sources)} paper-like sources included abstract/introduction/conclusion evidence sections.",
        [
            {"check": "paper_sources_present", "actual": len(paper_sources), "passed": bool(paper_sources)},
            {"check": "sectioned_paper_ratio", "ratio": round(ratio, 3), "passed": ratio >= 0.8, "examples": sectioned[:6]},
            {"check": "report_mentions_evidence_basis", "passed": report_mentions_basis},
        ],
    )


def _grade_hypothesis_evidence_matrix(task: EvalTask, store: ArtifactStore) -> GraderResult:
    report = store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else ""
    lower_report = report.lower()
    hypotheses = store.list("hypotheses")
    claims_by_id = {str(claim.get("id")): claim for claim in store.list("claims")}
    supported = [
        hypothesis
        for hypothesis in hypotheses
        if any(str(claim_id) in claims_by_id for claim_id in hypothesis.get("supporting_claim_ids", []))
    ]
    challenged = [
        hypothesis
        for hypothesis in hypotheses
        if hypothesis.get("contradicting_claim_ids") or "counterpoint" in lower_report or "limitation" in lower_report or "contradiction" in lower_report
    ]
    matrix_in_report = "hypothesis evidence matrix" in lower_report and "proof:" in lower_report and "counterpoint:" in lower_report
    support_ratio = len(supported) / max(len(hypotheses), 1)
    challenge_ratio = len(challenged) / max(len(hypotheses), 1)
    score = (support_ratio * 0.4) + (challenge_ratio * 0.3) + (0.3 if matrix_in_report else 0.0)
    passed = bool(hypotheses) and score >= 0.8
    return _result(
        "hypothesis_evidence_matrix",
        "code",
        "hypothesis proof/counterpoint verification",
        score,
        passed,
        1.0,
        f"{len(supported)}/{len(hypotheses)} hypotheses have retained proof claims; {len(challenged)}/{len(hypotheses)} have counterpoint or limitation handling.",
        [
            {"check": "hypotheses_present", "actual": len(hypotheses), "passed": bool(hypotheses)},
            {"check": "supporting_claim_ratio", "ratio": round(support_ratio, 3), "passed": support_ratio >= 0.8},
            {"check": "counterpoint_or_limitation_ratio", "ratio": round(challenge_ratio, 3), "passed": challenge_ratio >= 0.8},
            {"check": "report_contains_evidence_matrix", "passed": matrix_in_report},
        ],
    )


def _grade_transcript_progress(task: EvalTask, store: ArtifactStore) -> GraderResult:
    progress = store.progress_path.read_text(encoding="utf-8") if store.progress_path.exists() else ""
    traces = store.list("agent_traces")
    has_complete = "<promise>COMPLETE</promise>" in progress
    has_incomplete_stop = "Stopped with" in progress and "incomplete loop tasks" in progress
    has_steps = "Task 1:" in progress and len(progress.splitlines()) >= 5
    passed = (has_complete or has_incomplete_stop) and has_steps
    return _result(
        "transcript_progress",
        "code",
        "transcript analysis",
        1.0 if passed else 0.0,
        passed,
        0.75,
        f"Progress lines={len(progress.splitlines())}; traces={len(traces)}.",
        [
            {"check": "complete_or_incomplete_stop_marker", "passed": has_complete or has_incomplete_stop},
            {"check": "step_visibility", "passed": has_steps},
        ],
    )


def _grade_report_rubric(task: EvalTask, store: ArtifactStore) -> GraderResult:
    report = store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else ""
    evaluations = store.list("variant_evaluations")
    research_metrics = [
        row.get("metrics", {})
        for row in evaluations
        if row.get("inner_loop") == "research" and isinstance(row.get("metrics"), dict)
    ]
    rubric_dimensions = {"factual_accuracy", "citation_accuracy", "completeness", "source_quality", "tool_efficiency"}
    has_research_rubric_metrics = bool(research_metrics) and all(
        dimension in research_metrics[0] for dimension in rubric_dimensions
    )
    lower = report.lower()
    has_summary = "summary" in lower or "findings" in lower
    mentions_sources = "source" in lower
    mentions_uncertainty = any(term in lower for term in ["uncertain", "caveat", "contradiction", "limitation"])
    has_synthesis = any(term in lower for term in ["synthesis", "recommendation", "takeaway", "findings"])
    length_score = min(1.0, len(report.split()) / 80.0)
    nodes = [
        _binary_node("report_summary", "Does the report expose summary/findings?", has_summary, "Report contains summary/findings.", "Report lacks a clear summary/findings section."),
        _binary_node("evidence_mentions", "Does the report discuss sources/evidence?", mentions_sources, "Report names source/evidence basis.", "Report does not discuss source/evidence basis."),
        _binary_node("uncertainty_handling", "Does the report handle uncertainty or limitations?", mentions_uncertainty, "Report includes uncertainty, caveats, contradictions, or limitations.", "Report omits uncertainty/limitation handling."),
        _binary_node("research_metrics", "Did the research loop emit rubric dimensions?", has_research_rubric_metrics, "Research metrics include expected rubric dimensions.", "Research metrics are absent or incomplete."),
        dag_node(
            "substance",
            "Is the report substantial enough to judge?",
            verdict_from_score(length_score),
            length_score,
            right=["Report has enough substance for review."] if length_score >= 0.7 else [],
            wrong=["Report is too short for reliable model-style judgment."] if length_score < 0.7 else [],
            evidence={"word_count": len(report.split()), "has_synthesis": has_synthesis},
        ),
    ]
    return dag_grader_result(
        "model_report_rubric",
        "model",
        "DAG model-style report rubric",
        nodes=nodes,
        weight=0.8,
        threshold=0.7,
        summary="DAG judge scored report structure, evidence, uncertainty, research metrics, and substance.",
    )


def _grade_llm_research_quality_challenger(task: EvalTask, store: ArtifactStore) -> GraderResult:
    report = store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else ""
    claims = store.list("claims")
    sources = store.list("sources")
    evaluations = store.list("variant_evaluations")
    research_metrics = [
        row.get("metrics", {})
        for row in evaluations
        if row.get("inner_loop") == "research" and isinstance(row.get("metrics"), dict)
    ]
    first_metrics = research_metrics[0] if research_metrics else {}
    grounded_claims = [claim for claim in claims if claim.get("source_ids")]
    dimensions = {
        "factual_accuracy": max(float(first_metrics.get("factual_accuracy", 0.0)), len(grounded_claims) / max(len(claims), 1)),
        "citation_accuracy": max(float(first_metrics.get("citation_accuracy", 0.0)), 1.0 if grounded_claims and len(grounded_claims) == len(claims) else 0.0),
        "completeness": max(float(first_metrics.get("completeness", 0.0)), min(1.0, len(report.split()) / 180.0)),
        "source_quality": max(float(first_metrics.get("source_quality", 0.0)), min(1.0, len(sources) / 4.0)),
    }
    nodes = []
    labels = {
        "factual_accuracy": ("Are claims grounded in retained artifacts?", "Claims are grounded in source artifacts.", "Claims are weakly grounded or unsupported."),
        "citation_accuracy": ("Are citations/source links attached to claims?", "Claims retain citation/source IDs.", "Claims lack citation/source IDs."),
        "completeness": ("Does the report cover the task with enough depth?", "Report has enough depth for the prompt.", "Report is shallow or incomplete."),
        "source_quality": ("Did the run retain enough credible source material?", "Run retained multiple source artifacts.", "Run retained too little source material."),
    }
    for name, value in dimensions.items():
        criteria, right, wrong = labels[name]
        nodes.append(
            dag_node(
                name,
                criteria,
                verdict_from_score(value),
                value,
                right=[right] if value >= 0.7 else [],
                wrong=[wrong] if value < 0.7 else [],
                evidence={"metric": round(value, 3)},
            )
        )
    return dag_grader_result(
        "llm_research_quality_challenger",
        "model",
        "DAG LLM challenger research-quality rubric",
        nodes=nodes,
        weight=0.8,
        threshold=0.7,
        summary="DAG judge rated research quality across factual grounding, citation accuracy, completeness, and source quality.",
    )


def _grade_llm_hypothesis_novelty_challenger(task: EvalTask, store: ArtifactStore) -> GraderResult:
    hypotheses = store.list("hypotheses")
    claims = {str(claim.get("text", "")).strip().lower() for claim in store.list("claims")}
    nodes: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        text = str(hypothesis.get("text", "")).strip()
        novelty_score = float(hypothesis.get("novelty_score", 0.0) or 0.0)
        not_copy = text.lower() not in claims
        has_test = bool(hypothesis.get("next_experiment"))
        length_ok = len(text.split()) >= 6
        score = (novelty_score * 0.5) + (0.2 if not_copy else 0.0) + (0.2 if has_test else 0.0) + (0.1 if length_ok else 0.0)
        right = []
        wrong = []
        if not_copy:
            right.append("Hypothesis is not a direct copy of a claim.")
        else:
            wrong.append("Hypothesis copies an existing claim.")
        if has_test:
            right.append("Hypothesis includes a next experiment.")
        else:
            wrong.append("Hypothesis lacks a next experiment.")
        if length_ok:
            right.append("Hypothesis is specific enough to inspect.")
        else:
            wrong.append("Hypothesis is too short/generic.")
        nodes.append(
            dag_node(
                str(hypothesis.get("id") or f"hypothesis_{len(nodes) + 1}"),
                "Is this hypothesis novel, specific, and testable?",
                verdict_from_score(score),
                min(1.0, score),
                right=right,
                wrong=wrong,
                evidence={"novelty_score": novelty_score, "text": text[:160]},
            )
        )
    if not nodes:
        nodes.append(dag_node("has_hypotheses", "Did the run create hypotheses?", "missing", 0.0, wrong=["No hypotheses were created."]))
    return dag_grader_result(
        "llm_hypothesis_novelty_challenger",
        "model",
        "DAG LLM challenger hypothesis novelty rubric",
        nodes=nodes,
        weight=0.6,
        threshold=0.7,
        summary=f"DAG judge rated novelty and testability for {len(hypotheses)} hypothesis/hypotheses.",
    )


def _grade_llm_open_ended_judgment_challenger(task: EvalTask, store: ArtifactStore) -> GraderResult:
    report = store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else ""
    progress = store.progress_path.read_text(encoding="utf-8") if store.progress_path.exists() else ""
    lower_report = report.lower()
    checks = [
        ("answers_user_prompt", "Does the report answer the user prompt?", any(term in lower_report for term in _keywords(task.prompt, limit=8)), "Report uses prompt-relevant terms.", "Report appears off-topic."),
        ("uses_evidence_language", "Does the report use evidence language?", any(term in lower_report for term in ["source", "claim", "evidence", "citation"]), "Report discusses source/claim/evidence/citation.", "Report does not show evidence use."),
        ("handles_uncertainty", "Does the report handle uncertainty?", any(term in lower_report for term in ["uncertain", "limitation", "caveat", "contradiction", "confidence"]), "Report handles uncertainty or limits.", "Report does not handle uncertainty or limits."),
        ("has_synthesis", "Does the report synthesize?", any(term in lower_report for term in ["summary", "synthesis", "findings", "recommendation"]), "Report includes synthesis/findings.", "Report lacks synthesis/findings."),
        ("run_reached_terminal_marker", "Did the run reach a terminal marker?", "<promise>complete</promise>" in progress.lower() or "stopped with" in progress.lower(), "Progress records completion or explicit stop.", "Progress lacks completion/stop marker."),
    ]
    nodes = [_binary_node(node_id, criteria, passed, right, wrong) for node_id, criteria, passed, right, wrong in checks]
    return dag_grader_result(
        "llm_open_ended_judgment_challenger",
        "model",
        "DAG LLM challenger open-ended judgment",
        nodes=nodes,
        weight=0.6,
        threshold=0.7,
        summary="DAG judge made an open-ended judgment on relevance, evidence use, uncertainty, synthesis, and terminal progress.",
    )


def _binary_node(node_id: str, criteria: str, passed: bool, right: str, wrong: str) -> dict[str, Any]:
    right_items, wrong_items = right_wrong(passed, right, wrong)
    return dag_node(
        node_id,
        criteria,
        "yes" if passed else "no",
        1.0 if passed else 0.0,
        right=right_items,
        wrong=wrong_items,
    )


def _keywords(text: str, limit: int = 8) -> list[str]:
    stop = {"the", "and", "for", "with", "that", "this", "from", "into", "about", "research", "optimize", "how"}
    words = [word.lower() for word in re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", text) if word.lower() not in stop]
    unique: list[str] = []
    for word in words:
        if word not in unique:
            unique.append(word)
    return unique[:limit]


def _topic_keywords(text: str, limit: int = 10) -> list[str]:
    """Extract domain-specific topic terms, filtering generic verbs and filler words."""
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "about",
        "research", "optimize", "find", "make", "give", "take", "show", "have",
        "using", "used", "uses", "based", "also", "more", "some", "many", "most",
        "these", "those", "they", "their", "each", "when", "what", "which",
        "where", "there", "then", "than", "can", "could", "will", "would",
        "should", "shall", "must", "need", "want", "like", "just", "even",
        "only", "well", "very", "much", "such", "both", "after", "before",
        "over", "under", "other", "same", "different", "new", "long", "high",
        "data", "model", "models", "system", "systems", "method", "paper",
        "task", "tasks", "result", "results", "approach", "work",
    }
    words = [word.lower() for word in re.findall(r"[a-zA-Z][a-zA-Z-]{4,}", text) if word.lower() not in stop]
    unique: list[str] = []
    for word in words:
        if word not in unique:
            unique.append(word)
    return unique[:limit]


def _grade_prompt_output_relevance(task: EvalTask, store: ArtifactStore) -> GraderResult:
    """Check whether the report, claims, and sources are topically relevant to the original prompt."""
    report = store.report_path.read_text(encoding="utf-8") if store.report_path.exists() else ""
    claims = store.list("claims")
    sources = store.list("sources")
    keywords = _topic_keywords(task.prompt, limit=10)
    if not keywords:
        return _result(
            "prompt_output_relevance", "code", "prompt-output topical relevance",
            0.0, False, 1.0,
            "Could not extract topic keywords from prompt.",
            [{"check": "keywords_extracted", "passed": False}],
        )
    lower_report = report.lower()
    report_hits = sum(1 for kw in keywords if kw in lower_report)
    report_ratio = report_hits / len(keywords)
    relevant_claims = [
        claim for claim in claims
        if any(kw in str(claim.get("text", "")).lower() for kw in keywords)
    ]
    claim_ratio = len(relevant_claims) / max(len(claims), 1)
    relevant_sources = [
        source for source in sources
        if any(kw in str(source.get("title", "")).lower() for kw in keywords)
    ]
    source_ratio = len(relevant_sources) / max(len(sources), 1)
    score = round((report_ratio * 0.4) + (claim_ratio * 0.4) + (source_ratio * 0.2), 3)
    passed = score >= 0.4
    return _result(
        "prompt_output_relevance",
        "code",
        "prompt-output topical relevance",
        score,
        passed,
        1.0,
        (
            f"Prompt keywords={keywords}; report={report_hits}/{len(keywords)} hits; "
            f"claims={len(relevant_claims)}/{len(claims)} relevant; "
            f"sources={len(relevant_sources)}/{len(sources)} relevant."
        ),
        [
            {"check": "report_keyword_ratio", "keywords": keywords, "hits": report_hits, "ratio": round(report_ratio, 3), "passed": report_ratio >= 0.4},
            {"check": "claim_relevance", "relevant": len(relevant_claims), "total": len(claims), "ratio": round(claim_ratio, 3), "passed": claim_ratio >= 0.4},
            {"check": "source_relevance", "relevant": len(relevant_sources), "total": len(sources), "ratio": round(source_ratio, 3), "passed": source_ratio >= 0.2},
        ],
    )
