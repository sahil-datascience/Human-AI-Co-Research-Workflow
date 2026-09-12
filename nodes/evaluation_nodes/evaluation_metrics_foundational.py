from utils.load_vector_query import load_vector_query
from utils.paper_retriever import PaperRetriever
from utils.load_yaml_prompt import load_yaml_prompt
from utils.config_llm import set_llm
import json
import os
from utils.dsrp_state import DSRPState
from utils.llm_output_schemas.evaluation_node_schemas import (
    MetricsEvaluationSchema,
    MetricsEvidenceSchema,
)


def _structured_output_to_dict(output):
    if output is None:
        return {}
    if hasattr(output, "model_dump"):
        return output.model_dump()
    if hasattr(output, "dict"):
        return output.dict()
    return output if isinstance(output, dict) else {}


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


def _message_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _extract_evidence_list(evidence_json) -> list:
    if isinstance(evidence_json, list):
        return evidence_json

    if isinstance(evidence_json, dict):
        for key in ["evaluation_evidence", "candidate_evidence", "evidence"]:
            value = evidence_json.get(key, [])
            if isinstance(value, list):
                return value

    return []


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
        normalized.append({
            "id": item.get("id", ""),
            "page": item.get("page", ""),
            "section": item.get("section", ""),
            "direct_quote": item.get("direct_quote", ""),
        })
    return normalized


# =====================================================
# FOUNDATIONAL EVALUATION METRICS NODE
# =====================================================
def evaluation_metrics_foundational_node(state: DSRPState):

    base_path = "prompts/dsrp/evaluation/metrics"

    collection_name = state["collection_name"]
    persist_directory = state["persist_directory"]
    embedding_model = state["embedding_model"]
    llm_model = state.get("llm_model")
    llm_reasoning = state.get("llm_reasoning")
    retrieval_k = state.get("reranker_top_k")

    llm = set_llm(model=llm_model, reasoning=llm_reasoning)

    # =====================================================
    # 1) VECTOR RETRIEVAL
    # =====================================================

    vector_config = load_vector_query(
        f"{base_path}/vector_query.yaml"
    )

    retriever_tool = PaperRetriever(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model=embedding_model
    ).for_paper(
        state["paper_id"],
        k=retrieval_k or vector_config["k"]
    )

    docs = retriever_tool.invoke(vector_config["query"])

    context_text = "\n\n".join(
        f"[Page {d.metadata.get('page_no')} | {d.metadata.get('section_heading')}]\n{d.page_content}"
        for d in docs
    )

    # =====================================================
    # 2) EVIDENCE EXTRACTION
    # =====================================================

    retriever_prompt = load_yaml_prompt(
        f"{base_path}/retriever.yaml"
    )

    evidence_llm = llm.with_structured_output(MetricsEvidenceSchema)
    evidence_response = evidence_llm.invoke(
        retriever_prompt.format_messages(
            input=context_text
        )
    )

    evidence_json = _structured_output_to_dict(evidence_response)
    if not isinstance(evidence_json.get("evaluation_evidence"), list):
        evidence_json["evaluation_evidence"] = []
    evidence_list = _extract_evidence_list(evidence_json)

    # =====================================================
    # 3) LOAD MODELLING OUTPUT (FOR STRATEGY DEPENDENCY)
    # =====================================================

    modelling_output = state["dsrp_outputs"].get("modelling", {})
    foundational_paradigm = modelling_output.get("foundational_paradigm", "")
    ml_learning_type = _as_list(modelling_output.get("ml_learning_type", []))
    ml_problem_type = _as_list(modelling_output.get("ml_problem_type", []))

    # =====================================================
    # 4) METRICS CLASSIFICATION
    # =====================================================

    classifier_prompt = load_yaml_prompt(
        f"{base_path}/classifier.yaml"
    )

    classifier_input = {
        "modelling_context": {
            "foundational_paradigm": foundational_paradigm,
            "ml_learning_type": ml_learning_type,
            "ml_problem_type": ml_problem_type,
        },
        "evidence": evidence_list
    }

    classifier_llm = llm.with_structured_output(MetricsEvaluationSchema)
    classification_response = classifier_llm.invoke(
        classifier_prompt.format_messages(
            input=json.dumps(classifier_input)
        )
    )

    classification_json = _ensure_metrics_payload(_structured_output_to_dict(classification_response))

    # =====================================================
    # 5) AUDIT VALIDATION
    # =====================================================

    auditor_path = f"{base_path}/auditor.yaml"
    if os.path.exists(auditor_path):
        auditor_prompt = load_yaml_prompt(auditor_path)

        audit_input = {
            "modelling_context": {
                "foundational_paradigm": foundational_paradigm,
                "ml_learning_type": ml_learning_type,
                "ml_problem_type": ml_problem_type,
            },
            "evidence": evidence_list,
            "classification": classification_json,
        }

        auditor_llm = llm.with_structured_output(MetricsEvaluationSchema)
        audit_response = auditor_llm.invoke(
            auditor_prompt.format_messages(
                input=json.dumps(audit_input)
            )
        )

        output_payload = _ensure_metrics_payload(_structured_output_to_dict(audit_response))
    else:
        output_payload = classification_json

    if not isinstance(output_payload, dict):
        output_payload = {}

    classifier_reasoning = ""
    classifier_bibliography = []
    if isinstance(classification_json, dict):
        classifier_reasoning = str(classification_json.get("reasoning", "")).strip()
        classifier_bibliography = _normalize_bibliography(
            classification_json.get("bibliography", [])
        )

    audited_reasoning = str(output_payload.get("reasoning", "")).strip()
    audited_bibliography = _normalize_bibliography(output_payload.get("bibliography", []))

    # Ensure citation traceability survives auditing even when the audit output is sparse.
    if not audited_reasoning and classifier_reasoning:
        output_payload["reasoning"] = classifier_reasoning
    else:
        output_payload["reasoning"] = audited_reasoning

    if not audited_bibliography and classifier_bibliography:
        output_payload["bibliography"] = classifier_bibliography
    else:
        output_payload["bibliography"] = audited_bibliography

    output_payload["validated_reasoning"] = output_payload.get("reasoning", "")
    output_payload["validated_bibliography"] = output_payload.get("bibliography", [])

    # Evaluation strategy must follow modelling paradigm.
    output_payload["evaluation_strategy"] = _strategy_from_paradigm(foundational_paradigm)

    # Keep recent foundational output shape under unified key.
    if _is_statistical_paradigm(foundational_paradigm):
        output_payload["learning_type"] = ["Not applicable"]
        output_payload["problem_type"] = ["Not applicable"]
    else:
        output_payload["learning_type"] = ml_learning_type
        output_payload["problem_type"] = ml_problem_type

    # =====================================================
    # 6) STORE RESULT
    # =====================================================

    state["dsrp_outputs"]["evaluation_metrics"] = output_payload
    state["dsrp_outputs"].pop("evaluation_metrics_foundational", None)

    return {"dsrp_outputs": state["dsrp_outputs"]}
