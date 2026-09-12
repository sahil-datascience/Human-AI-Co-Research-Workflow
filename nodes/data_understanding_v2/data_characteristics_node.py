from __future__ import annotations

from pathlib import Path

from utils.config_llm import set_llm
from utils.dsrp_state import DSRPState

from .data_understanding_v2_common import (
    build_context_text,
    invoke_prompt_with_json,
    invoke_prompt_with_text,
    run_classifiers_parallel,
)


def data_characteristics_node(state: DSRPState) -> dict:
    prompt_root = Path("prompts/dsrp/data_understanding/v2/characteristics")
    llm = set_llm(model=state.get("llm_model"), reasoning=state.get("llm_reasoning"))

    context_text = build_context_text(state, str(prompt_root / "vector_query.yaml"))

    label_jobs = [
        ("Temporal", str(prompt_root / "temporal" / "classifier.yaml")),
        ("Spatial", str(prompt_root / "spatial" / "classifier.yaml")),
        ("Textual", str(prompt_root / "textual" / "classifier.yaml")),
        ("Visual", str(prompt_root / "visual" / "classifier.yaml")),
        ("Networked", str(prompt_root / "networked" / "classifier.yaml")),
    ]

    parallel_workers = state.get("parallel_workers", 4)
    try:
        parallel_workers = int(parallel_workers)
    except (TypeError, ValueError):
        parallel_workers = 4

    label_outputs = run_classifiers_parallel(llm=llm, jobs=label_jobs, context_text=context_text, max_workers=max(1, parallel_workers))

    if (prompt_root / "retriever.yaml").exists():
        evidence_json = invoke_prompt_with_text(llm, str(prompt_root / "retriever.yaml"), context_text)
    else:
        evidence_json = {"candidate_evidence": []}

    audit_json = invoke_prompt_with_json(
        llm,
        str(prompt_root / "auditor.yaml"),
        {"label_outputs": label_outputs, "labels": [label_name for label_name, _ in label_jobs], "evidence": evidence_json},
    )

    return {
        "data_characteristics": audit_json,
        "data_characteristics_pre_audit": {
            "evidence": evidence_json,
            "classification": label_outputs,
        },
    }
