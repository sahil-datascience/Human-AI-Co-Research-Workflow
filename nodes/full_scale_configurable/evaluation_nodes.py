from __future__ import annotations

import json
import os

from utils.dsrp_state import DSRPState
from utils.load_yaml_prompt import load_yaml_prompt
from llm_output_schemas.evaluation_node_schemas import (
    EthicalEvidenceSchema,
    EthicalSocialSchema,
    InterpretabilityEvidenceSchema,
    InterpretabilitySchema,
    MetricsEvaluationSchema,
    MetricsEvidenceSchema,
    TheoryEvidenceSchema,
    TheoryOrientationSchema,
)

from .config import get_llm
from .retrieval import format_docs_context, retrieve_and_rerank


def _structured_output_to_dict(output):
    if output is None:
        return {}
    if hasattr(output, "model_dump"):
        return output.model_dump()
    if hasattr(output, "dict"):
        return output.dict()
    return output if isinstance(output, dict) else {}


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _normalize_bibliography(value) -> list:
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "id": item.get("id", ""),
                "page": item.get("page", ""),
                "section": item.get("section", ""),
                "direct_quote": item.get("direct_quote", ""),
            }
        )
    return normalized


def _strategy_from_paradigm(foundational_paradigm) -> str:
    paradigm = str(foundational_paradigm or "").strip().lower()
    if "classical" in paradigm or "statistical" in paradigm:
        return "Statistical Model Diagnostics"
    if "mixed" in paradigm:
        return "Mixed Evaluation"
    return "Machine Learning Evaluation"


def _is_statistical_paradigm(foundational_paradigm) -> bool:
    paradigm = str(foundational_paradigm or "").strip().lower()
    return ("classical" in paradigm) or ("statistical" in paradigm)


def _ensure_metrics_payload(payload):
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("evaluation_strategy", "Machine Learning Evaluation")
    for key in ("learning_type", "problem_type", "validation_procedure"):
        value = payload.get(key, [])
        if value is None:
            value = []
        elif not isinstance(value, list):
            value = [value]
        payload[key] = value
    payload.setdefault("evaluation_metrics_present", "No")
    payload.setdefault("effect_size_reported", "Not applicable")
    payload.setdefault("assumption_checks_reported", "Not applicable")
    payload.setdefault("confidence", 0.0)
    payload.setdefault("reasoning", "")
    payload.setdefault("bibliography", [])
    payload.setdefault("audit_commentary", "")
    return payload


def _ensure_theory_payload(payload):
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("explicit_theory", "No")
    payload.setdefault("implicit_theory_detected", "No")
    payload.setdefault("epistemological_orientation", "Data-driven discovery")
    payload.setdefault("primary_research_orientation", "Method-oriented research")
    payload.setdefault("confidence", 0.0)
    payload.setdefault("reasoning_explanation", "")
    payload.setdefault("bibliography", [])
    payload.setdefault("validated_reasoning", payload.get("reasoning_explanation", ""))
    payload.setdefault("validated_bibliography", payload.get("bibliography", []))
    payload.setdefault("audit_commentary", "")
    return payload


def _ensure_interpretability_payload(payload):
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("interpretability_discussed", "No")
    payload.setdefault("interpretability_approach", payload.get("interpretability_method", ""))
    payload.setdefault("interpretability_method", payload.get("interpretability_approach", ""))
    payload.setdefault("model_transparency_level", "Low transparency")
    payload.setdefault("confidence", 0.0)
    payload.setdefault("reasoning_explanation", "")
    payload.setdefault("bibliography", [])
    payload.setdefault("validated_reasoning", payload.get("reasoning_explanation", ""))
    payload.setdefault("validated_bibliography", payload.get("bibliography", []))
    payload.setdefault("audit_commentary", "")
    return payload


def _ensure_ethical_payload(payload):
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("privacy_protection_reported", "No")
    payload.setdefault("bias_fairness_considered", "No")
    payload.setdefault("societal_impact_discussed", "No")
    payload.setdefault("ethical_reflection_level", "Low")
    payload.setdefault("confidence", 0.0)
    payload.setdefault("reasoning_explanation", "")
    payload.setdefault("bibliography", [])
    payload.setdefault("validated_reasoning", payload.get("reasoning_explanation", ""))
    payload.setdefault("validated_bibliography", payload.get("bibliography", []))
    payload.setdefault("audit_commentary", "")
    return payload


def _evidence_context(state: DSRPState, node_name: str, base_path: str, default_reranker_top_k: int):
    docs = retrieve_and_rerank(
        state,
        node_name,
        f"{base_path}/vector_query.yaml",
        default_reranker_top_k=default_reranker_top_k,
    )
    return format_docs_context(docs)


def evaluation_metrics_foundational_node(state: DSRPState):
    node_name = "evaluation_metrics"
    base_path = "prompts/dsrp/evaluation/metrics"
    context_text = _evidence_context(state, node_name, base_path, default_reranker_top_k=10)

    retriever_prompt = load_yaml_prompt(f"{base_path}/retriever.yaml")
    evidence_llm = get_llm(state, node_name, "evidence").with_structured_output(
        MetricsEvidenceSchema
    )
    evidence_response = evidence_llm.invoke(retriever_prompt.format_messages(input=context_text))
    evidence_json = _structured_output_to_dict(evidence_response)
    if not isinstance(evidence_json.get("evaluation_evidence"), list):
        evidence_json["evaluation_evidence"] = []
    evidence_list = evidence_json["evaluation_evidence"]

    modelling_output = state["dsrp_outputs"].get("modelling", {})
    foundational_paradigm = modelling_output.get("foundational_paradigm", "")
    ml_learning_type = _as_list(modelling_output.get("ml_learning_type", []))
    ml_problem_type = _as_list(modelling_output.get("ml_problem_type", []))

    classifier_prompt = load_yaml_prompt(f"{base_path}/classifier.yaml")
    classifier_input = {
        "modelling_context": {
            "foundational_paradigm": foundational_paradigm,
            "ml_learning_type": ml_learning_type,
            "ml_problem_type": ml_problem_type,
        },
        "evidence": evidence_list,
    }
    classifier_llm = get_llm(state, node_name, "classifier").with_structured_output(
        MetricsEvaluationSchema
    )
    classification_response = classifier_llm.invoke(
        classifier_prompt.format_messages(input=json.dumps(classifier_input))
    )
    classification_json = _ensure_metrics_payload(_structured_output_to_dict(classification_response))

    auditor_path = f"{base_path}/auditor.yaml"
    if os.path.exists(auditor_path):
        auditor_prompt = load_yaml_prompt(auditor_path)
        audit_input = {
            "modelling_context": classifier_input["modelling_context"],
            "evidence": evidence_list,
            "classification": classification_json,
        }
        auditor_llm = get_llm(state, node_name, "auditor").with_structured_output(
            MetricsEvaluationSchema
        )
        audit_response = auditor_llm.invoke(
            auditor_prompt.format_messages(input=json.dumps(audit_input))
        )
        output_payload = _ensure_metrics_payload(_structured_output_to_dict(audit_response))
    else:
        output_payload = classification_json

    classifier_reasoning = str(classification_json.get("reasoning", "")).strip()
    classifier_bibliography = _normalize_bibliography(classification_json.get("bibliography", []))
    audited_reasoning = str(output_payload.get("reasoning", "")).strip()
    audited_bibliography = _normalize_bibliography(output_payload.get("bibliography", []))
    output_payload["reasoning"] = audited_reasoning or classifier_reasoning
    output_payload["bibliography"] = audited_bibliography or classifier_bibliography
    output_payload["validated_reasoning"] = output_payload.get("reasoning", "")
    output_payload["validated_bibliography"] = output_payload.get("bibliography", [])
    output_payload["evaluation_strategy"] = _strategy_from_paradigm(foundational_paradigm)

    if _is_statistical_paradigm(foundational_paradigm):
        output_payload["learning_type"] = ["Not applicable"]
        output_payload["problem_type"] = ["Not applicable"]
    else:
        output_payload["learning_type"] = ml_learning_type
        output_payload["problem_type"] = ml_problem_type

    state["dsrp_outputs"]["evaluation_metrics"] = output_payload
    return {"dsrp_outputs": state["dsrp_outputs"]}


def evaluation_theoretical_orientation_node(state: DSRPState):
    node_name = "evaluation_theoretical_orientation"
    base_path = "prompts/dsrp/evaluation/epistemological/theoretical_orientation"
    context_text = _evidence_context(state, node_name, base_path, default_reranker_top_k=12)

    retriever_prompt = load_yaml_prompt(f"{base_path}/retriever.yaml")
    evidence_llm = get_llm(state, node_name, "evidence").with_structured_output(
        TheoryEvidenceSchema
    )
    evidence_response = evidence_llm.invoke(retriever_prompt.format_messages(input=context_text))
    evidence_json = _structured_output_to_dict(evidence_response)
    if not isinstance(evidence_json.get("theory_evidence"), list):
        evidence_json["theory_evidence"] = []

    classifier_prompt = load_yaml_prompt(f"{base_path}/classifier.yaml")
    classifier_llm = get_llm(state, node_name, "classifier").with_structured_output(
        TheoryOrientationSchema
    )
    classification_response = classifier_llm.invoke(
        classifier_prompt.format_messages(input=json.dumps(evidence_json))
    )
    classification_json = _ensure_theory_payload(_structured_output_to_dict(classification_response))

    auditor_prompt = load_yaml_prompt(f"{base_path}/auditor.yaml")
    auditor_llm = get_llm(state, node_name, "auditor").with_structured_output(
        TheoryOrientationSchema
    )
    audit_response = auditor_llm.invoke(
        auditor_prompt.format_messages(input=json.dumps(classification_json))
    )
    audit_json = _ensure_theory_payload(_structured_output_to_dict(audit_response))
    state["dsrp_outputs"][node_name] = audit_json
    return {"dsrp_outputs": state["dsrp_outputs"]}


def evaluation_interpretability_node(state: DSRPState):
    node_name = "evaluation_interpretability"
    base_path = "prompts/dsrp/evaluation/epistemological/interpretability"
    context_text = _evidence_context(state, node_name, base_path, default_reranker_top_k=12)

    retriever_prompt = load_yaml_prompt(f"{base_path}/retriever.yaml")
    evidence_llm = get_llm(state, node_name, "evidence").with_structured_output(
        InterpretabilityEvidenceSchema
    )
    evidence_response = evidence_llm.invoke(retriever_prompt.format_messages(input=context_text))
    evidence_json = _structured_output_to_dict(evidence_response)
    if not isinstance(evidence_json.get("interpretability_evidence"), list):
        evidence_json["interpretability_evidence"] = []

    classifier_prompt = load_yaml_prompt(f"{base_path}/classifier.yaml")
    classifier_llm = get_llm(state, node_name, "classifier").with_structured_output(
        InterpretabilitySchema
    )
    classification_response = classifier_llm.invoke(
        classifier_prompt.format_messages(input=json.dumps(evidence_json))
    )
    classification_json = _ensure_interpretability_payload(
        _structured_output_to_dict(classification_response)
    )

    auditor_prompt = load_yaml_prompt(f"{base_path}/auditor.yaml")
    auditor_llm = get_llm(state, node_name, "auditor").with_structured_output(
        InterpretabilitySchema
    )
    audit_response = auditor_llm.invoke(
        auditor_prompt.format_messages(input=json.dumps(classification_json))
    )
    audit_json = _ensure_interpretability_payload(_structured_output_to_dict(audit_response))
    if not audit_json.get("interpretability_approach") and audit_json.get("interpretability_method"):
        audit_json["interpretability_approach"] = audit_json["interpretability_method"]
    state["dsrp_outputs"][node_name] = audit_json
    return {"dsrp_outputs": state["dsrp_outputs"]}


def evaluation_ethical_social_node(state: DSRPState):
    node_name = "evaluation_ethical_social"
    base_path = "prompts/dsrp/evaluation/ethical_social"
    context_text = _evidence_context(state, node_name, base_path, default_reranker_top_k=10)

    retriever_prompt = load_yaml_prompt(f"{base_path}/retriever.yaml")
    evidence_llm = get_llm(state, node_name, "evidence").with_structured_output(
        EthicalEvidenceSchema
    )
    evidence_response = evidence_llm.invoke(retriever_prompt.format_messages(input=context_text))
    evidence_json = _structured_output_to_dict(evidence_response)
    if not isinstance(evidence_json.get("ethical_evidence"), list):
        evidence_json["ethical_evidence"] = []

    classifier_prompt = load_yaml_prompt(f"{base_path}/classifier.yaml")
    classifier_llm = get_llm(state, node_name, "classifier").with_structured_output(
        EthicalSocialSchema
    )
    classification_response = classifier_llm.invoke(
        classifier_prompt.format_messages(input=json.dumps(evidence_json))
    )
    classification_json = _ensure_ethical_payload(_structured_output_to_dict(classification_response))

    auditor_prompt = load_yaml_prompt(f"{base_path}/auditor.yaml")
    auditor_llm = get_llm(state, node_name, "auditor").with_structured_output(
        EthicalSocialSchema
    )
    audit_response = auditor_llm.invoke(
        auditor_prompt.format_messages(input=json.dumps(classification_json))
    )
    audit_json = _ensure_ethical_payload(_structured_output_to_dict(audit_response))
    state["dsrp_outputs"][node_name] = audit_json
    return {"dsrp_outputs": state["dsrp_outputs"]}

