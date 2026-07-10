from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

ALL_CONFIGURED_MODEL_ID = "all-configured"


@dataclass(frozen=True)
class ModelOption:
    id: str
    provider: str
    model: str
    lab: str
    label: str


DEFAULT_MODEL_CATALOG: tuple[ModelOption, ...] = (
    ModelOption("openai/gpt-5.5", "openai", "gpt-5.5", "OpenAI", "OpenAI - GPT-5.5"),
    ModelOption("openai/gpt-5.4", "openai", "gpt-5.4", "OpenAI", "OpenAI - GPT-5.4"),
    ModelOption("openai/gpt-5.2", "openai", "gpt-5.2", "OpenAI", "OpenAI - GPT-5.2"),
    ModelOption("openai/gpt-5.2-pro", "openai", "gpt-5.2-pro", "OpenAI", "OpenAI - GPT-5.2 Pro"),
    ModelOption("openai/gpt-5.1", "openai", "gpt-5.1", "OpenAI", "OpenAI - GPT-5.1"),
    ModelOption("openai/gpt-5", "openai", "gpt-5", "OpenAI", "OpenAI - GPT-5"),
    ModelOption("openai/gpt-5-mini", "openai", "gpt-5-mini", "OpenAI", "OpenAI - GPT-5 mini"),
    ModelOption("openai/gpt-5-nano", "openai", "gpt-5-nano", "OpenAI", "OpenAI - GPT-5 nano"),
    ModelOption("openai/gpt-4o", "openai", "gpt-4o", "OpenAI", "OpenAI - GPT-4o"),
    ModelOption("openai/gpt-4o-mini", "openai", "gpt-4o-mini", "OpenAI", "OpenAI - GPT-4o mini"),
    ModelOption("anthropic/claude-opus-4-7", "anthropic", "claude-opus-4-7", "Anthropic", "Anthropic - Claude Opus 4.7"),
    ModelOption("anthropic/claude-opus-4-6", "anthropic", "claude-opus-4-6", "Anthropic", "Anthropic - Claude Opus 4.6"),
    ModelOption("anthropic/claude-sonnet-4-6", "anthropic", "claude-sonnet-4-6", "Anthropic", "Anthropic - Claude Sonnet 4.6"),
    ModelOption("anthropic/claude-sonnet-4-5", "anthropic", "claude-sonnet-4-5", "Anthropic", "Anthropic - Claude Sonnet 4.5"),
    ModelOption("anthropic/claude-haiku-4-5", "anthropic", "claude-haiku-4-5", "Anthropic", "Anthropic - Claude Haiku 4.5"),
    ModelOption("kimi/kimi-k2.6", "kimi", "kimi-k2.6", "Kimi", "Kimi - K2.6"),
    ModelOption("ollama/qwen3.5:latest", "ollama", "qwen3.5:latest", "Ollama", "Ollama - qwen3.5"),
    ModelOption("ollama/qwen", "ollama", "qwen", "Ollama", "Ollama - qwen"),
    ModelOption("ollama/qwen2.5-coder", "ollama", "qwen2.5-coder", "Ollama", "Ollama - qwen2.5-coder"),
    ModelOption("local/local-deterministic-fallback", "local", "local-deterministic-fallback", "Local", "Local deterministic fallback"),
)


def model_catalog() -> list[ModelOption]:
    configured = _configured_model_entries()
    if not configured:
        return list(DEFAULT_MODEL_CATALOG)
    merged: list[ModelOption] = []
    seen: set[str] = set()
    for option in [*configured, *DEFAULT_MODEL_CATALOG]:
        if option.id in seen:
            continue
        seen.add(option.id)
        merged.append(option)
    return merged


def _configured_model_entries() -> list[ModelOption]:
    configured = os.environ.get("RESEARCH_HARNESS_LLM_MODELS", "").strip()
    if not configured:
        return []
    parsed = [_parse_model_entry(entry) for entry in configured.split(",") if entry.strip()]
    return [option for option in parsed if option]


def model_choices() -> list[tuple[str, str]]:
    return [(ALL_CONFIGURED_MODEL_ID, "Use all configured models (round-robin)")] + [
        (option.id, option.label) for option in model_catalog()
    ]


def resolve_model_selection(provider: Optional[str], model: Optional[str]) -> tuple[str, str]:
    selected_provider = (provider or "auto").lower()
    selected_model = model or "openai/gpt-5.2"
    if _is_all_configured_selection(selected_provider, selected_model):
        return "multi", ALL_CONFIGURED_MODEL_ID
    option = find_model_option(selected_model)
    if option:
        if selected_provider in {"auto", "", option.provider}:
            return option.provider, option.model
        if "/" in selected_model:
            return option.provider, option.model
        return selected_provider, option.model
    if "/" in selected_model:
        prefix, model_name = selected_model.split("/", 1)
        if prefix in {"openai", "anthropic", "kimi", "ollama", "local"}:
            if selected_provider in {"auto", "", prefix}:
                return prefix, model_name
            return selected_provider, model_name
    return selected_provider, selected_model


def configured_model_pool() -> list[ModelOption]:
    return _configured_model_entries() or list(DEFAULT_MODEL_CATALOG)


def is_all_configured_selection(provider: Optional[str], model: Optional[str]) -> bool:
    return _is_all_configured_selection((provider or "auto").lower(), model or "")


def find_model_option(model_id_or_name: str) -> Optional[ModelOption]:
    normalized = model_id_or_name.strip()
    for option in model_catalog():
        if normalized in {option.id, option.model, option.label}:
            return option
    return None


def format_model_catalog() -> str:
    lines = ["Available research harness models:"]
    lines.append(f"- {ALL_CONFIGURED_MODEL_ID} (Multi-provider; uses every configured available model round-robin)")
    for option in model_catalog():
        lines.append(f"- {option.id} ({option.lab}; provider={option.provider}; model={option.model})")
    return "\n".join(lines)


def _parse_model_entry(entry: str) -> Optional[ModelOption]:
    raw = entry.strip()
    if not raw:
        return None
    parts = [part.strip() for part in raw.split(":")]
    if len(parts) >= 2:
        provider, model = parts[0].lower(), parts[1]
        label = parts[2] if len(parts) >= 3 and parts[2] else _default_label(provider, model)
        lab = parts[3] if len(parts) >= 4 and parts[3] else _default_lab(provider)
        return ModelOption(f"{provider}/{model}", provider, model, lab, label)
    if "/" in raw:
        provider, model = raw.split("/", 1)
        provider = provider.lower()
        return ModelOption(raw, provider, model, _default_lab(provider), _default_label(provider, model))
    return ModelOption(f"openai/{raw}", "openai", raw, "OpenAI", _default_label("openai", raw))


def _is_all_configured_selection(provider: str, model: str) -> bool:
    normalized_model = model.strip().lower()
    return provider == "multi" or normalized_model in {
        "all",
        "all-configured",
        "all_models",
        "all-models",
        "all_configured",
        "multi",
        "multi-provider",
    }


def _default_lab(provider: str) -> str:
    return {"openai": "OpenAI", "anthropic": "Anthropic", "kimi": "Kimi", "ollama": "Ollama", "local": "Local"}.get(provider, provider.title())


def _default_label(provider: str, model: str) -> str:
    return f"{_default_lab(provider)} - {model}"
