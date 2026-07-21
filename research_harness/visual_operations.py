"""Semantic operation taxonomy shared by run visualizations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


CATEGORY_COLORS: Mapping[str, str] = {
    "model": "#7c3aed",
    "search": "#2563eb",
    "retrieval": "#0891b2",
    "transformation": "#0f766e",
    "analysis": "#9333ea",
    "evaluation": "#d97706",
    "system": "#64748b",
}


@dataclass(frozen=True)
class OperationDefinition:
    kind: str
    label: str
    category: str
    ordinal_scope: str
    tooltip_metadata: tuple[str, ...] = ()

    @property
    def color(self) -> str:
        return CATEGORY_COLORS[self.category]


OPERATIONS: Mapping[str, OperationDefinition] = {
    "model_turn": OperationDefinition("main_llm_call", "Main LLM call", "model", "main_llm_call", ("model_call_id",)),
    "web_search": OperationDefinition("web_search", "Web search", "search", "web_search", ("tool_name", "tool_call_id")),
    "arxiv_api_search": OperationDefinition("arxiv_search", "arXiv search", "search", "arxiv_search", ("tool_name", "tool_call_id")),
    "semantic_scholar_api_search": OperationDefinition("semantic_scholar_search", "Semantic Scholar search", "search", "semantic_scholar_search", ("tool_name", "tool_call_id")),
    "openalex_api_search": OperationDefinition("openalex_search", "OpenAlex search", "search", "openalex_search", ("tool_name", "tool_call_id")),
    "docs_blogs_search": OperationDefinition("docs_search", "Docs search", "search", "docs_search", ("tool_name", "tool_call_id")),
    "fetch_document": OperationDefinition("document_fetch", "Document fetch", "retrieval", "document_fetch", ("tool_name", "tool_call_id")),
    "inspect_document_figures": OperationDefinition("figure_inspection", "Inspect document figures", "retrieval", "figure_inspection", ("tool_name", "tool_call_id")),
    "extract_structured_data": OperationDefinition("structured_extraction", "Extract structured data", "transformation", "structured_extraction", ("tool_name", "tool_call_id")),
    "generate_svg_chart": OperationDefinition("chart_generation", "Generate chart", "transformation", "chart_generation", ("tool_name", "tool_call_id")),
    "analyze_research_document": OperationDefinition("document_analysis", "Analyze research document", "analysis", "document_analysis", ("tool_name", "tool_call_id")),
}


def operation_for(*, event_type: str = "", tool_name: Optional[str] = None) -> OperationDefinition:
    key = tool_name or event_type
    if key in OPERATIONS:
        return OPERATIONS[key]
    label = (tool_name or event_type or "system work").replace("_", " ").strip().title()
    kind = (tool_name or event_type or "system_work").strip().lower()
    return OperationDefinition(kind, label, "system", kind, ("tool_name", "tool_call_id"))
