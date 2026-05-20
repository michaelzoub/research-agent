from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .model_catalog import ALL_CONFIGURED_MODEL_ID, configured_model_pool, resolve_model_selection


class LLMError(RuntimeError):
    pass


# USD per token for known models. Used to compute real per-run costs.
# Prices sourced from public pricing pages; marked as estimates for unreleased models.
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50e-6, "output": 10.00e-6},
    "gpt-4o-mini": {"input": 0.15e-6, "output": 0.60e-6},
    "gpt-4-turbo": {"input": 10.00e-6, "output": 30.00e-6},
    "gpt-4": {"input": 30.00e-6, "output": 60.00e-6},
    "gpt-3.5-turbo": {"input": 0.50e-6, "output": 1.50e-6},
    "o1": {"input": 15.00e-6, "output": 60.00e-6},
    "o1-mini": {"input": 3.00e-6, "output": 12.00e-6},
    "o3": {"input": 10.00e-6, "output": 40.00e-6},
    "o3-mini": {"input": 1.10e-6, "output": 4.40e-6},
    "o4-mini": {"input": 1.10e-6, "output": 4.40e-6},
    "gpt-5": {"input": 10.00e-6, "output": 30.00e-6},   # estimate
    "gpt-5.5": {"input": 10.00e-6, "output": 30.00e-6},  # estimate
    "gpt-5.4": {"input": 10.00e-6, "output": 30.00e-6},  # estimate
    "gpt-5.2": {"input": 10.00e-6, "output": 30.00e-6},  # estimate
    "gpt-5.1": {"input": 10.00e-6, "output": 30.00e-6},  # estimate
    "gpt-5-mini": {"input": 1.00e-6, "output": 4.00e-6},  # estimate
    "gpt-5-nano": {"input": 0.20e-6, "output": 0.80e-6},  # estimate
    "claude-opus-4-7": {"input": 5.00e-6, "output": 25.00e-6},  # estimate
    "claude-opus-4-6": {"input": 5.00e-6, "output": 25.00e-6},  # estimate
    "claude-sonnet-4-6": {"input": 3.00e-6, "output": 15.00e-6},  # estimate
    "claude-sonnet-4-5": {"input": 3.00e-6, "output": 15.00e-6},  # estimate
    "claude-haiku-4-5": {"input": 0.80e-6, "output": 4.00e-6},  # estimate
    "kimi-k2.6": {"input": 0.95e-6, "output": 4.00e-6},
}
_DEFAULT_PRICING: dict[str, float] = {"input": 10.00e-6, "output": 30.00e-6}


def _pricing_for(model: str) -> dict[str, float]:
    # Match by prefix so "gpt-4o-2024-05-13" resolves to "gpt-4o".
    for key, pricing in _MODEL_PRICING.items():
        if model.startswith(key):
            return pricing
    return _DEFAULT_PRICING


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0


class LLMClient:
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = 60.0,
    ):
        _load_local_env_defaults()
        raw_provider = provider or os.environ.get("RESEARCH_HARNESS_LLM_PROVIDER") or "auto"
        raw_model = model or os.environ.get("RESEARCH_HARNESS_LLM_MODEL") or "openai/gpt-5.2"
        self.provider, self.model = resolve_model_selection(raw_provider, raw_model)
        self.model_pool = [(option.provider, option.model) for option in configured_model_pool()] if self.provider == "multi" else []
        self._model_pool_cursor = 0
        self.openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.anthropic_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.kimi_api_key = api_key or os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY")
        if self.provider == "kimi":
            self.api_key = self.kimi_api_key
        else:
            self.api_key = self.openai_api_key if self.provider in {"auto", "openai"} else self.anthropic_api_key
        self.timeout_seconds = timeout_seconds
        # Accumulated real token counts across all calls on this client instance.
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.call_history: list[dict[str, Any]] = []

    @property
    def is_live(self) -> bool:
        if self.provider == "multi":
            return any(self._provider_available(provider) for provider, _model in self.model_pool)
        if self.provider == "local":
            return False
        if self.provider == "openai":
            return _looks_like_openai_key(self.openai_api_key)
        if self.provider == "anthropic":
            return _looks_like_anthropic_key(self.anthropic_api_key)
        if self.provider == "kimi":
            return _looks_like_kimi_key(self.kimi_api_key)
        if self.provider == "auto":
            return (
                _looks_like_openai_key(self.openai_api_key)
                or _looks_like_anthropic_key(self.anthropic_api_key)
                or _looks_like_kimi_key(self.kimi_api_key)
            )
        return False

    @property
    def model_label(self) -> str:
        if self.provider == "multi":
            available = len(self._available_model_pool())
            return f"{ALL_CONFIGURED_MODEL_ID} ({available}/{len(self.model_pool)} available)"
        if self.is_live:
            return self.model
        return "local-deterministic-fallback"

    def complete(self, system: str, user: str, *, max_output_tokens: int = 900, temperature: float = 0.7) -> LLMResponse:
        active_provider, active_model = self._select_execution_model()
        stored_provider, stored_model = self.provider, self.model
        self.provider, self.model = active_provider, active_model
        try:
            if not self.is_live:
                response = LLMResponse(
                    text=self._local_response(system, user),
                    model=self.model_label,
                    provider="local",
                    prompt_tokens=_estimate_tokens(system + "\n" + user),
                    completion_tokens=80,
                )
            else:
                response = self._live_response(system, user, max_output_tokens=max_output_tokens, temperature=temperature)
            response.cost = self._response_cost(response)
            self.total_prompt_tokens += response.prompt_tokens
            self.total_completion_tokens += response.completion_tokens
            self.call_history.append(
                {
                    "provider": response.provider,
                    "model": response.model,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "total_tokens": response.prompt_tokens + response.completion_tokens,
                    "cost_usd": round(response.cost, 6),
                    "is_live": response.provider != "local",
                    "status": "completed",
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "configured_provider": stored_provider,
                    "configured_model": stored_model,
                }
            )
            return response
        except Exception as exc:
            self.call_history.append(
                {
                    "provider": self.provider,
                    "model": self.model,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "is_live": self.provider != "local",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "configured_provider": stored_provider,
                    "configured_model": stored_model,
                }
            )
            raise
        finally:
            self.provider, self.model = stored_provider, stored_model

    def total_cost(self) -> float:
        """Return accumulated cost in USD based on model pricing table."""
        if not self.call_history:
            return 0.0
        return sum(float(call.get("cost_usd") or 0.0) for call in self.call_history)

    def _response_cost(self, response: LLMResponse) -> float:
        if response.provider == "local":
            return 0.0
        pricing = _pricing_for(self.model)
        return (
            response.prompt_tokens * pricing["input"]
            + response.completion_tokens * pricing["output"]
        )

    def cost_breakdown(self) -> dict[str, object]:
        pricing = _pricing_for(self.model)
        return {
            "model": self.model,
            "provider": self.provider,
            "model_pool": [f"{provider}/{model}" for provider, model in self.model_pool],
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "cost_usd": round(self.total_cost(), 6),
            "model_call_count": len(self.call_history),
            "model_calls": self.call_history,
            "pricing_input_per_token": pricing["input"],
            "pricing_output_per_token": pricing["output"],
            "pricing_note": "Local deterministic fallback calls are recorded with zero cost; live-provider prices are estimates until verified against billing.",
        }

    def complete_json(self, system: str, user: str, *, max_output_tokens: int = 900, temperature: float = 0.7) -> dict[str, object]:
        response = self.complete(system, user, max_output_tokens=max_output_tokens, temperature=temperature)
        try:
            return json.loads(_extract_json(response.text))
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model did not return valid JSON: {exc}") from exc

    def _openai_response(self, system: str, user: str, *, max_output_tokens: int, temperature: float = 0.7) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_output_tokens,
            "temperature": round(max(0.0, min(2.0, temperature)), 2),
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json",
                "User-Agent": "research-harness/0.1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"HTTP {exc.code} from OpenAI Chat Completions: {body[:1000]}") from exc
        text = str(data["choices"][0]["message"]["content"] or "")
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=str(data.get("model") or self.model),
            provider="openai",
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )

    def _anthropic_response(self, system: str, user: str, *, max_output_tokens: int, temperature: float = 0.7) -> LLMResponse:
        payload = {
            "model": self.model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": max_output_tokens,
            "temperature": round(max(0.0, min(1.0, temperature)), 2),
        }
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": str(self.anthropic_api_key),
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
                "User-Agent": "research-harness/0.1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"HTTP {exc.code} from Anthropic Messages: {body[:1000]}") from exc
        blocks = data.get("content") or []
        text = "".join(str(block.get("text", "")) for block in blocks if isinstance(block, dict) and block.get("type") == "text")
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=str(data.get("model") or self.model),
            provider="anthropic",
            prompt_tokens=int(usage.get("input_tokens") or 0),
            completion_tokens=int(usage.get("output_tokens") or 0),
        )

    def _kimi_response(self, system: str, user: str, *, max_output_tokens: int, temperature: float = 0.7) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_output_tokens,
            # Kimi K2.6 currently rejects other values with
            # "invalid temperature: only 1 is allowed for this model".
            "temperature": 1,
        }
        data = self._kimi_chat_completions(payload)
        text = str(data["choices"][0]["message"]["content"] or "")
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=str(data.get("model") or self.model),
            provider="kimi",
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )

    def _kimi_chat_completions(self, payload: dict[str, object]) -> dict[str, object]:
        last_error: Optional[urllib.error.HTTPError] = None
        last_body = ""
        for attempt in range(3):
            request = urllib.request.Request(
                "https://api.moonshot.ai/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.kimi_api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "research-harness/0.1.0",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code != 429 or attempt >= 2:
                    raise LLMError(f"HTTP {exc.code} from Kimi Chat Completions: {body[:1000]}") from exc
                last_error = exc
                last_body = body
                retry_after = _retry_after_seconds(exc.headers.get("Retry-After") if exc.headers else None)
                time.sleep(retry_after if retry_after is not None else 1.0 + attempt)
        raise LLMError(f"HTTP 429 from Kimi Chat Completions after retries: {last_body[:1000]}") from last_error

    def _live_response(self, system: str, user: str, *, max_output_tokens: int, temperature: float) -> LLMResponse:
        if self.provider == "anthropic":
            return self._anthropic_response(system, user, max_output_tokens=max_output_tokens, temperature=temperature)
        if self.provider == "kimi":
            return self._kimi_response(system, user, max_output_tokens=max_output_tokens, temperature=temperature)
        return self._openai_response(system, user, max_output_tokens=max_output_tokens, temperature=temperature)

    def _select_execution_model(self) -> tuple[str, str]:
        if self.provider != "multi":
            return self.provider, self.model
        available = self._available_model_pool()
        if not available:
            return "local", "local-deterministic-fallback"
        provider, model = available[self._model_pool_cursor % len(available)]
        self._model_pool_cursor += 1
        return provider, model

    def _available_model_pool(self) -> list[tuple[str, str]]:
        return [
            (provider, model)
            for provider, model in self.model_pool
            if self._provider_available(provider)
        ]

    def _provider_available(self, provider: str) -> bool:
        if provider == "local":
            return True
        if provider == "openai":
            return _looks_like_openai_key(self.openai_api_key)
        if provider == "anthropic":
            return _looks_like_anthropic_key(self.anthropic_api_key)
        if provider == "kimi":
            return _looks_like_kimi_key(self.kimi_api_key)
        return False

    def validate(self) -> bool:
        if self.provider == "multi":
            return bool(self._available_model_pool())
        if not self.is_live:
            return False
        if self.provider == "anthropic":
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Return ok."}],
                "max_tokens": 8,
            }
            request = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "x-api-key": str(self.anthropic_api_key),
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                    "User-Agent": "research-harness/0.1.0",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=min(self.timeout_seconds, 15.0)):
                    return True
            except urllib.error.HTTPError as exc:
                if exc.code in {401, 403}:
                    return False
                raise
        if self.provider == "kimi":
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Return ok."}],
                "max_completion_tokens": 8,
            }
            request = urllib.request.Request(
                "https://api.moonshot.ai/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.kimi_api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "research-harness/0.1.0",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=min(self.timeout_seconds, 15.0)):
                    return True
            except urllib.error.HTTPError as exc:
                if exc.code in {401, 403}:
                    return False
                raise
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Return ok."}],
            "max_completion_tokens": 8,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json",
                "User-Agent": "research-harness/0.1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout_seconds, 15.0)):
                return True
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                return False
            raise

    def _local_response(self, system: str, user: str) -> str:
        if "json" in system.lower():
            return json.dumps({"score": 0.5, "rationale": "Local fallback score; configure OPENAI_API_KEY for live judging."})
        return (
            "Local deterministic fallback response. Configure OPENAI_API_KEY, ANTHROPIC_API_KEY, or MOONSHOT_API_KEY "
            "and choose a live model such as openai/gpt-5.2, anthropic/claude-sonnet-4-5, or kimi/kimi-k2.6.\n\n"
            f"Prompt excerpt: {user[:500]}"
        )


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _looks_like_openai_key(api_key: Optional[str]) -> bool:
    if not api_key:
        return False
    cleaned = api_key.strip()
    if cleaned in {"", "...", "changeme", "your-key-here"}:
        return False
    return cleaned.startswith(("sk-", "sess-"))


def _looks_like_anthropic_key(api_key: Optional[str]) -> bool:
    if not api_key:
        return False
    cleaned = api_key.strip()
    if cleaned in {"", "...", "changeme", "your-key-here"}:
        return False
    return cleaned.startswith("sk-ant-")


def _looks_like_kimi_key(api_key: Optional[str]) -> bool:
    if not api_key:
        return False
    cleaned = api_key.strip()
    return cleaned not in {"", "...", "changeme", "your-key-here"}


def _retry_after_seconds(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return max(0.0, min(10.0, float(value)))
    except ValueError:
        return None


_ENV_DEFAULTS_LOADED = False


def _load_local_env_defaults() -> None:
    global _ENV_DEFAULTS_LOADED
    if _ENV_DEFAULTS_LOADED:
        return
    _ENV_DEFAULTS_LOADED = True
    for path in (Path(".env"), Path(".env.local")):
        _load_env_file(path)


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        clean_key = key.strip()
        clean_value = value.strip().strip('"').strip("'")
        if clean_key and clean_key not in os.environ:
            os.environ[clean_key] = clean_value
