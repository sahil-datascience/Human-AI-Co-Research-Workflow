from __future__ import annotations

from pathlib import Path

from utils.config_llm import set_llm
from utils.dsrp_state import DSRPState

from .data_understanding_v2_common import build_context_text, invoke_prompt_with_json, invoke_prompt_with_text


def data_category_node(state: DSRPState) -> dict:
    prompt_root = Path("prompts/dsrp/data_understanding/v2/category")
    llm = set_llm(model=state.get("llm_model"), reasoning=state.get("llm_reasoning"))

    context_text = build_context_text(state, str(prompt_root / "vector_query.yaml"))

    evidence_json = invoke_prompt_with_text(llm, str(prompt_root / "retriever.yaml"), context_text)
    classification_json = invoke_prompt_with_json(llm, str(prompt_root / "classifier.yaml"), evidence_json)
    audit_json = invoke_prompt_with_json(
        llm,
        str(prompt_root / "auditor.yaml"),
        {"classification": classification_json, "evidence": evidence_json},
    )

    return {
        "data_category": audit_json,
        "data_category_pre_audit": {
            "evidence": evidence_json,
            "classification": classification_json,
        },
    }
