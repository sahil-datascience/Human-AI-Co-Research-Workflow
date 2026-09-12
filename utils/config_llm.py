
import os
from typing import Any
import warnings

from langchain_openai import ChatOpenAI


def _normalise_reasoning_config(reasoning: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Build a valid OpenRouter reasoning config.

    Rules:
    - Accept one of `effort` or `max_tokens` (not both; effort wins if both provided).
    - `enabled` defaults to inferred from effort/max_tokens.
    - Include `exclude` only when explicitly set.
    """
    if not isinstance(reasoning, dict):
        return None

    enabled = bool(reasoning.get("enabled", False))
    if not enabled:
        return None

    cfg: dict[str, Any] = {
        "enabled": True,
        "exclude": bool(reasoning.get("exclude", False)),
    }

    allowed_efforts = {"xhigh", "high", "medium", "low", "minimal", "none"}

    effort = reasoning.get("effort")
    if isinstance(effort, str):
        effort_value = effort.strip().lower()
        if effort_value in allowed_efforts and effort_value != "none":
            cfg["effort"] = effort_value

    if "effort" not in cfg:
        max_tokens = reasoning.get("max_tokens")
        if isinstance(max_tokens, int) and max_tokens > 0:
            cfg["max_tokens"] = max_tokens

    return cfg


def set_llm(
    model: str | None = None,
    temperature: float = 0,
    model_kwargs: dict | None = None,
    reasoning: dict[str, Any] | None = None,
):
    # Priority: explicit arg > env var > project default.
    selected_model = model or os.getenv("DSRP_LLM_MODEL") or "gpt-4o-mini"
    provider_env = os.getenv("DSRP_LLM_PROVIDER")
    selected_provider = (
        provider_env.strip().lower()
        if provider_env
        else ("openrouter" if "/" in selected_model else "openai")
    )

    if os.getenv("DSRP_LLM_DEBUG", "").strip().lower() in {"1", "true", "yes"}:
        print(f"Using LLM provider: {selected_provider}, model: {selected_model}")

    effective_model_kwargs = dict(model_kwargs or {})

    if selected_provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        reasoning_config = _normalise_reasoning_config(reasoning)
        if reasoning_config is None:
            return ChatOpenRouter(
                model=selected_model,
                temperature=temperature,
                model_kwargs=effective_model_kwargs,
            )

        # Compatibility fallbacks across langchain_openrouter versions.
        candidates: list[dict[str, Any]] = []

        # 1) Nested reasoning under model_kwargs (current preferred style).
        mk_nested = dict(effective_model_kwargs)
        mk_nested["reasoning"] = reasoning_config
        candidates.append({"model_kwargs": mk_nested})

        # 2) Top-level `reasoning` argument (some wrappers expose this).
        candidates.append({"model_kwargs": dict(effective_model_kwargs), "reasoning": reasoning_config})

        # 3) Flattened reasoning_* keys in model_kwargs (legacy/alternate wrappers).
        mk_flat = dict(effective_model_kwargs)
        if "effort" in reasoning_config:
            mk_flat["reasoning_effort"] = reasoning_config["effort"]
        if "max_tokens" in reasoning_config:
            mk_flat["reasoning_max_tokens"] = reasoning_config["max_tokens"]
        if "exclude" in reasoning_config:
            mk_flat["reasoning_exclude"] = reasoning_config["exclude"]
        mk_flat["reasoning_enabled"] = reasoning_config.get("enabled", True)
        candidates.append({"model_kwargs": mk_flat})

        last_exc: Exception | None = None
        for kwargs in candidates:
            try:
                return ChatOpenRouter(
                    model=selected_model,
                    temperature=temperature,
                    **kwargs,
                )
            except Exception as exc:
                last_exc = exc

        warnings.warn(
            "OpenRouter reasoning config was rejected by ChatOpenRouter; "
            "continuing without reasoning for compatibility. "
            f"Last error: {last_exc}",
            RuntimeWarning,
        )
        return ChatOpenRouter(
            model=selected_model,
            temperature=temperature,
            model_kwargs=effective_model_kwargs,
        )

    # Guard against accidentally forwarding OpenRouter-only knobs to OpenAI.
    effective_model_kwargs.pop("reasoning", None)

    return ChatOpenAI(
        model=selected_model,
        temperature=temperature,
        model_kwargs=effective_model_kwargs,
    )

