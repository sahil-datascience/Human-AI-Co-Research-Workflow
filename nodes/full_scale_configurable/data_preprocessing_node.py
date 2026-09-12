from __future__ import annotations

import json

from utils.dsrp_state import DSRPState
from utils.load_yaml_prompt import load_yaml_prompt
from utils.llm_output_schemas.data_preprocessing_node_schemas import (
    CandidateEvidenceSchema,
    DataPreprocessingAuditorSchema,
    DataPreprocessingClassifierSchema,
)

from .config import get_llm
from .retrieval import format_docs_context, retrieve_and_rerank


NODE_NAME = "data_preprocessing"


def _structured_output_to_dict(output):
    if hasattr(output, "model_dump"):
        return output.model_dump()
    if hasattr(output, "dict"):
        return output.dict()
    return output


def _ensure_audit_defaults(audit_json):
    if not isinstance(audit_json, dict):
        return audit_json

    for key in (
        "validated_data_cleaning",
        "validated_data_reduction",
        "validated_data_transformation",
    ):
        value = audit_json.get(key)
        if not isinstance(value, dict):
            value = {}
        value.setdefault("status", "Not Reported")
        value.setdefault("justification", "")
        audit_json[key] = value

    audit_json.setdefault("confidence", 0.0)
    audit_json.setdefault("validated_reasoning", "")
    audit_json.setdefault("validated_bibliography", [])
    audit_json.setdefault("audit_commentary", "")
    return audit_json


def data_preprocessing_node(state: DSRPState):
    docs = retrieve_and_rerank(
        state,
        NODE_NAME,
        f"prompts/dsrp/{NODE_NAME}/vector_query.yaml",
        default_reranker_top_k=8,
    )
    context_text = format_docs_context(docs)

    retriever_prompt = load_yaml_prompt(f"prompts/dsrp/{NODE_NAME}/retriever.yaml")
    evidence_llm = get_llm(state, NODE_NAME, "evidence").with_structured_output(CandidateEvidenceSchema)
    evidence_response = evidence_llm.invoke(retriever_prompt.format_messages(input=context_text))
    evidence_json = _structured_output_to_dict(evidence_response)

    classifier_prompt = load_yaml_prompt(f"prompts/dsrp/{NODE_NAME}/classifier.yaml")
    classifier_llm = get_llm(state, NODE_NAME, "classifier").with_structured_output(
        DataPreprocessingClassifierSchema
    )
    classification_response = classifier_llm.invoke(
        classifier_prompt.format_messages(input=json.dumps(evidence_json))
    )
    classification_json = _structured_output_to_dict(classification_response)

    auditor_prompt = load_yaml_prompt(f"prompts/dsrp/{NODE_NAME}/auditor.yaml")
    auditor_llm = get_llm(state, NODE_NAME, "auditor").with_structured_output(
        DataPreprocessingAuditorSchema
    )
    audit_response = auditor_llm.invoke(
        auditor_prompt.format_messages(input=json.dumps(classification_json))
    )
    audit_json = _ensure_audit_defaults(_structured_output_to_dict(audit_response))

    state["dsrp_outputs"][NODE_NAME] = audit_json
    state["dsrp_outputs"][f"{NODE_NAME}_pre_audit"] = {
        "evidence": evidence_json,
        "classification": classification_json,
    }
    return {"dsrp_outputs": state["dsrp_outputs"]}
