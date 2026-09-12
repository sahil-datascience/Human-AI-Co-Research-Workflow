"""Configurable full-scale DSRP graph.

This keeps the original node files unchanged while allowing each graph node to
receive its own LLM, reasoning, retrieval, and reranker settings.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Mapping

from langgraph.graph import END, StateGraph

import nodes.full_scale_configurable.data_preprocessing_node as data_preprocessing_module
import nodes.full_scale_configurable.data_understanding_node as data_understanding_module
import nodes.full_scale_configurable.evaluation_nodes as evaluation_module
import nodes.full_scale_configurable.modelling_node as modelling_module
import nodes.full_scale_configurable.research_question_node as research_question_module
import nodes.gatekeeper_node as gatekeeper_module
from utils.dsrp_state import DSRPState


NODE_NAMES = [
    "gatekeeper",
    "research_question",
    "data_understanding",
    "data_preprocessing",
    "modelling",
    "evaluation_metrics",
    "evaluation_theoretical_orientation",
    "evaluation_interpretability",
    "evaluation_ethical_social",
]

DEFAULT_RETRIEVAL_K = 25


def route_after_gatekeeper(state: DSRPState):
    decision = str(state.get("gatekeeper", {}).get("final_classification", "")).strip().lower()
    if decision == "include":
        return "workflow_start"
    return END


def _normalise_node_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = dict(config or {})

    # Friendly aliases for notebook editing.
    if "model" in cfg and "llm_model" not in cfg:
        cfg["llm_model"] = cfg.pop("model")
    if "reasoning" in cfg and "llm_reasoning" not in cfg:
        cfg["llm_reasoning"] = cfg.pop("reasoning")
    if "top_k" in cfg and "reranker_top_k" not in cfg:
        cfg["reranker_top_k"] = cfg.pop("top_k")

    return cfg


def _config_for_node(state: Mapping[str, Any], node_name: str) -> dict[str, Any]:
    default_config = _normalise_node_config(state.get("default_node_config", {}))
    node_configs = state.get("node_configs", {}) or {}
    node_entry = dict(node_configs.get(node_name, {}) or {})
    node_config = _normalise_node_config(node_entry)
    nested_default = _normalise_node_config(node_entry.get("default", {}))
    return {**default_config, **node_config, **nested_default}


@contextmanager
def _temporary_vector_k(module: Any, retrieval_k: Any):
    """Temporarily override vector_query.yaml k for a node module.

    Several existing nodes bind `load_vector_query` at import time. Patching that
    module-level symbol lets the new graph support per-node retrieval_k without
    editing old node files.
    """
    if retrieval_k is None or not hasattr(module, "load_vector_query"):
        yield
        return

    original = module.load_vector_query

    def load_vector_query_with_k(path):
        config = original(path)
        if isinstance(config, dict):
            config = dict(config)
            config["k"] = int(retrieval_k)
        return config

    module.load_vector_query = load_vector_query_with_k
    try:
        yield
    finally:
        module.load_vector_query = original


def _configured_node(node_name: str, module: Any, func_name: str) -> Callable[[DSRPState], dict]:
    def run(state: DSRPState) -> dict:
        node_configs = state.get("node_configs", {}) or {}
        node_entry = dict(node_configs.get(node_name, {}) or {})
        node_config = _config_for_node(state, node_name)
        retrieval_k = node_config.pop("retrieval_k", DEFAULT_RETRIEVAL_K)

        node_state = dict(state)
        node_state["dsrp_outputs"] = dict(state.get("dsrp_outputs", {}))
        node_state["_active_node_name"] = node_name
        node_state["_active_node_config_entry"] = node_entry
        node_state["_active_default_node_config"] = dict(state.get("default_node_config", {}) or {})
        node_state.update(node_config)

        node_func = getattr(module, func_name)
        with _temporary_vector_k(module, retrieval_k):
            return node_func(node_state)

    return run


builder = StateGraph(DSRPState)

builder.add_node("gatekeeper", _configured_node("gatekeeper", gatekeeper_module, "gatekeeper_node"))
builder.add_node("workflow_start", lambda state: {})
builder.add_node("workflow_complete", lambda state: {})
builder.add_node(
    "research_question",
    _configured_node("research_question", research_question_module, "research_question_node"),
)
builder.add_node(
    "data_understanding",
    _configured_node("data_understanding", data_understanding_module, "data_understanding_node"),
)
builder.add_node(
    "data_preprocessing",
    _configured_node("data_preprocessing", data_preprocessing_module, "data_preprocessing_node"),
)
builder.add_node("modelling", _configured_node("modelling", modelling_module, "modelling_node"))
builder.add_node(
    "evaluation_metrics_foundational_node",
    _configured_node(
        "evaluation_metrics",
        evaluation_module,
        "evaluation_metrics_foundational_node",
    ),
)
builder.add_node(
    "evaluation_theoretical_orientation_node",
    _configured_node(
        "evaluation_theoretical_orientation",
        evaluation_module,
        "evaluation_theoretical_orientation_node",
    ),
)
builder.add_node(
    "evaluation_interpretability_node",
    _configured_node(
        "evaluation_interpretability",
        evaluation_module,
        "evaluation_interpretability_node",
    ),
)
builder.add_node(
    "evaluation_ethical_social_node",
    _configured_node(
        "evaluation_ethical_social",
        evaluation_module,
        "evaluation_ethical_social_node",
    ),
)

builder.set_entry_point("gatekeeper")
builder.add_conditional_edges("gatekeeper", route_after_gatekeeper)

builder.add_edge("workflow_start", "research_question")
builder.add_edge("workflow_start", "data_understanding")
builder.add_edge("workflow_start", "data_preprocessing")
builder.add_edge("workflow_start", "modelling")
builder.add_edge("workflow_start", "evaluation_theoretical_orientation_node")
builder.add_edge("workflow_start", "evaluation_interpretability_node")
builder.add_edge("workflow_start", "evaluation_ethical_social_node")

builder.add_edge("modelling", "evaluation_metrics_foundational_node")

builder.add_edge("research_question", "workflow_complete")
builder.add_edge("data_understanding", "workflow_complete")
builder.add_edge("data_preprocessing", "workflow_complete")
builder.add_edge("evaluation_metrics_foundational_node", "workflow_complete")
builder.add_edge("evaluation_theoretical_orientation_node", "workflow_complete")
builder.add_edge("evaluation_interpretability_node", "workflow_complete")
builder.add_edge("evaluation_ethical_social_node", "workflow_complete")

builder.add_edge("workflow_complete", END)

graph = builder.compile()
