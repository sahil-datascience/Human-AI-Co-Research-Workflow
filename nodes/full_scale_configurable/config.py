from __future__ import annotations

import os
from typing import Any, Mapping

from utils.config_llm import set_llm


CONFIG_KEYS = {
    "llm_model",
    "model",
    "llm_reasoning",
    "reasoning",
    "retrieval_k",
    "reranker_top_k",
    "top_k",
    "reranker_model",
}

DEFAULT_RETRIEVAL_K = 25


def normalise_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = dict(config or {})
    if "model" in cfg and "llm_model" not in cfg:
        cfg["llm_model"] = cfg["model"]
    if "reasoning" in cfg and "llm_reasoning" not in cfg:
        cfg["llm_reasoning"] = cfg["reasoning"]
    if "top_k" in cfg and "reranker_top_k" not in cfg:
        cfg["reranker_top_k"] = cfg["top_k"]
    return cfg


def _config_only(config: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = normalise_config(config)
    return {key: value for key, value in cfg.items() if key in CONFIG_KEYS}


def node_config_entry(state: Mapping[str, Any], node_name: str) -> dict[str, Any]:
    if state.get("_active_node_name") == node_name and isinstance(
        state.get("_active_node_config_entry"), Mapping
    ):
        return dict(state.get("_active_node_config_entry") or {})
    return dict((state.get("node_configs", {}) or {}).get(node_name, {}) or {})


def get_config(state: Mapping[str, Any], node_name: str, *parts: str) -> dict[str, Any]:
    """Return merged config for a node or internal stage.

    Supports both flat configs:
        {"data_understanding": {"llm_model": "..."}}

    and nested configs:
        {"data_understanding": {"default": {...}, "data_category": {...}}}
    """
    merged = _config_only(
        state.get("_active_default_node_config", state.get("default_node_config", {}))
    )
    entry = node_config_entry(state, node_name)

    if any(key in entry for key in CONFIG_KEYS):
        merged.update(_config_only(entry))
    merged.update(_config_only(entry.get("default", {})))

    current: Any = entry
    for part in parts:
        if not isinstance(current, Mapping):
            break
        stage = current.get(part, {})
        merged.update(_config_only(stage))
        current = stage

    return merged


def get_llm(state: Mapping[str, Any], node_name: str, *parts: str):
    config = get_config(state, node_name, *parts)
    resolved_model = config.get("llm_model")
    if resolved_model is None:
        resolved_model = "gpt-4o-mini"
    stage = ".".join(parts) if parts else "default"
    trace_item = {
        "node": node_name,
        "stage": stage,
        "llm_model": resolved_model,
        "llm_reasoning": config.get("llm_reasoning"),
    }
    if isinstance(state, dict):
        state.setdefault("_llm_call_trace", []).append(trace_item)
        dsrp_outputs = state.get("dsrp_outputs")
        if isinstance(dsrp_outputs, dict):
            dsrp_outputs.setdefault("_llm_call_trace", []).append(trace_item)
            dsrp_outputs.setdefault(f"_llm_call_trace_{node_name}", []).append(trace_item)
    if os.getenv("DSRP_LLM_DEBUG", "").strip().lower() in {"1", "true", "yes"}:
        print(
            f"[LLM config] node={node_name} stage={stage} "
            f"model={resolved_model} reasoning={config.get('llm_reasoning')}"
        )
    return set_llm(
        model=resolved_model,
        reasoning=config.get("llm_reasoning"),
    )


def get_retrieval_k(
    state: Mapping[str, Any],
    node_name: str,
    default: int,
    *parts: str,
) -> int:
    config = get_config(state, node_name, *parts)
    value = config.get("retrieval_k", DEFAULT_RETRIEVAL_K)
    try:
        value = int(value)
    except (TypeError, ValueError):
        return DEFAULT_RETRIEVAL_K
    return value if value > 0 else DEFAULT_RETRIEVAL_K


def get_reranker_top_k(
    state: Mapping[str, Any],
    node_name: str,
    default: int = 10,
    *parts: str,
) -> int:
    config = get_config(state, node_name, *parts)
    value = config.get("reranker_top_k", default)
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def get_reranker_model(
    state: Mapping[str, Any],
    node_name: str,
    default: str = "cohere/rerank-4-pro",
    *parts: str,
) -> str:
    config = get_config(state, node_name, *parts)
    value = config.get("reranker_model", default)
    if not isinstance(value, str) or not value.strip():
        return default
    return value.strip()
