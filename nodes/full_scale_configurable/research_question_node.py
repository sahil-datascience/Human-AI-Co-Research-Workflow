from __future__ import annotations

import json

from utils.dsrp_state import DSRPState
from utils.load_yaml_prompt import load_yaml_prompt
from utils.llm_output_schemas.research_question_node_schemas import (
    ResearchQuestionAuditorSchema,
    ResearchQuestionClassifierSchema,
    ResearchQuestionEvidenceSchema,
    normalize_research_question_label,
)

from .config import get_llm
from .retrieval import format_docs_context, retrieve_and_rerank


NODE_NAME = "research_question"


def _structured_output_to_dict(output):
    if output is None:
        return {}
    if hasattr(output, "model_dump"):
        return output.model_dump()
    if hasattr(output, "dict"):
        return output.dict()
    return output if isinstance(output, dict) else {}


def _ensure_evidence_payload(payload):
    if not isinstance(payload, dict):
        payload = {}
    evidence = payload.get("candidate_evidence")
    if not isinstance(evidence, list):
        evidence = []
    payload["candidate_evidence"] = evidence
    return payload


def _ensure_classifier_payload(payload):
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("classification", "Exploratory")
    payload.setdefault("confidence", 0.0)
    payload.setdefault("reasoning_explanation", "")
    payload.setdefault("bibliography", [])
    return payload


def _ensure_audit_payload(payload):
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("final_classification", "Exploratory")
    payload.setdefault("confidence", 0.0)
    payload.setdefault("validated_reasoning", "")
    payload.setdefault("validated_bibliography", [])
    payload.setdefault("audit_commentary", "")
    return payload


def research_question_node(state: DSRPState):
    docs = retrieve_and_rerank(
        state,
        NODE_NAME,
        f"prompts/dsrp/{NODE_NAME}/vector_query.yaml",
        default_reranker_top_k=10,
    )
    context_text = format_docs_context(docs)

    retriever_prompt = load_yaml_prompt(f"prompts/dsrp/{NODE_NAME}/retriever.yaml")
    evidence_llm = get_llm(state, NODE_NAME, "evidence").with_structured_output(
        ResearchQuestionEvidenceSchema
    )
    evidence_response = evidence_llm.invoke(retriever_prompt.format_messages(input=context_text))
    evidence_json = _ensure_evidence_payload(_structured_output_to_dict(evidence_response))

    classifier_prompt = load_yaml_prompt(f"prompts/dsrp/{NODE_NAME}/classifier.yaml")
    classifier_llm = get_llm(state, NODE_NAME, "classifier").with_structured_output(
        ResearchQuestionClassifierSchema
    )
    classification_response = classifier_llm.invoke(
        classifier_prompt.format_messages(input=json.dumps(evidence_json))
    )
    classification_json = _ensure_classifier_payload(
        _structured_output_to_dict(classification_response)
    )

    auditor_prompt = load_yaml_prompt(f"prompts/dsrp/{NODE_NAME}/auditor.yaml")
    auditor_llm = get_llm(state, NODE_NAME, "auditor").with_structured_output(
        ResearchQuestionAuditorSchema
    )
    audit_response = auditor_llm.invoke(
        auditor_prompt.format_messages(input=json.dumps(classification_json))
    )
    audit_json = _ensure_audit_payload(_structured_output_to_dict(audit_response))

    final_label = normalize_research_question_label(audit_json.get("final_classification"))
    if final_label is None:
        final_label = normalize_research_question_label(classification_json.get("classification"))
    audit_json["final_classification"] = final_label or "Exploratory"

    state["dsrp_outputs"][NODE_NAME] = audit_json
    state["dsrp_outputs"][f"{NODE_NAME}_pre_audit"] = {
        "evidence": evidence_json,
        "classification": classification_json,
    }

    return {"dsrp_outputs": state["dsrp_outputs"]}

