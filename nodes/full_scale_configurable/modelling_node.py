from __future__ import annotations

import json

from utils.dsrp_state import DSRPState
from utils.load_yaml_prompt import load_yaml_prompt
from utils.llm_output_schemas.modelling_node_schemas import (
    FoundationalAuditSchema,
    FoundationalClassifierSchema,
    GlobalModellingAuditSchema,
    MLLearningClassifierSchema,
    MLProblemClassifierSchema,
    ModellingEvidenceSchema,
    SpecialisedAuditSchema,
    SpecialisedClassifierSchema,
)

from .config import get_llm
from .retrieval import format_docs_context, retrieve_and_rerank


NODE_NAME = "modelling"


def _structured_output_to_dict(output):
    if output is None:
        return {}
    if hasattr(output, "model_dump"):
        return output.model_dump()
    if hasattr(output, "dict"):
        return output.dict()
    return output if isinstance(output, dict) else {}


def _ensure_modelling_evidence(payload):
    if not isinstance(payload, dict):
        payload = {}
    evidence = payload.get("modelling_evidence")
    if not isinstance(evidence, list):
        evidence = []
    payload["modelling_evidence"] = evidence
    return payload


def _ensure_foundational(payload):
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("foundational_paradigm", "Machine Learning")
    payload.setdefault("confidence", 0.0)
    payload.setdefault("reasoning_explanation", "")
    payload.setdefault("bibliography", [])
    return payload


def _ensure_ml_learning(payload):
    if not isinstance(payload, dict):
        payload = {}
    learning = payload.get("ml_learning_type", [])
    if learning is None:
        learning = []
    elif not isinstance(learning, list):
        learning = [learning]
    payload["ml_learning_type"] = learning
    payload.setdefault("deep_learning_used", False)
    payload.setdefault("confidence", 0.0)
    payload.setdefault("reasoning_explanation", "")
    payload.setdefault("bibliography", [])
    return payload


def _ensure_ml_problem(payload):
    if not isinstance(payload, dict):
        payload = {}
    problem = payload.get("ml_problem_type", [])
    if problem is None:
        problem = []
    elif not isinstance(problem, list):
        problem = [problem]
    payload["ml_problem_type"] = problem
    payload.setdefault("confidence", 0.0)
    payload.setdefault("reasoning_explanation", "")
    payload.setdefault("bibliography", [])
    return payload


def _ensure_foundational_audit(payload):
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("foundational_paradigm", "Machine Learning")
    for key in ("ml_learning_type", "ml_problem_type"):
        value = payload.get(key, [])
        if value is None:
            value = []
        elif not isinstance(value, list):
            value = [value]
        payload[key] = value
    payload.setdefault("deep_learning_used", False)
    payload.setdefault("confidence", 0.0)
    payload.setdefault("validated_reasoning", "")
    payload.setdefault("validated_bibliography", [])
    payload.setdefault("audit_commentary", "")
    return payload


def _ensure_specialised(payload):
    if not isinstance(payload, dict):
        payload = {}
    paradigms = payload.get("specialised_paradigms", [])
    if paradigms is None:
        paradigms = []
    elif not isinstance(paradigms, list):
        paradigms = [paradigms]
    payload["specialised_paradigms"] = paradigms
    payload.setdefault("confidence", 0.0)
    payload.setdefault("reasoning", "")
    payload.setdefault("bibliography", [])
    payload.setdefault("validated_reasoning", "")
    payload.setdefault("validated_bibliography", [])
    payload.setdefault("audit_commentary", "")
    return payload


def _ensure_global_audit(payload):
    payload = _ensure_foundational_audit(payload)
    paradigms = payload.get("specialised_paradigms", [])
    if paradigms is None:
        paradigms = []
    elif not isinstance(paradigms, list):
        paradigms = [paradigms]
    payload["specialised_paradigms"] = paradigms
    return payload


def _retrieve_context(state: DSRPState, base_path: str, config_part: str):
    docs = retrieve_and_rerank(
        state,
        NODE_NAME,
        f"{base_path}/vector_query.yaml",
        config_part,
        default_reranker_top_k=10,
    )
    return format_docs_context(docs)


def modelling_node(state: DSRPState):
    base_path = "prompts/dsrp/modelling"
    foundational_path = f"{base_path}/foundational"
    specialised_path = f"{base_path}/specialised"

    context_text = _retrieve_context(state, foundational_path, "foundational")
    retriever_prompt = load_yaml_prompt(f"{foundational_path}/retriever.yaml")
    evidence_llm = get_llm(state, NODE_NAME, "foundational", "evidence").with_structured_output(
        ModellingEvidenceSchema
    )
    evidence_response = evidence_llm.invoke(retriever_prompt.format_messages(input=context_text))
    evidence_json = _ensure_modelling_evidence(_structured_output_to_dict(evidence_response))
    evidence_json["modelling_evidence"] = [
        e for e in evidence_json["modelling_evidence"] if e.get("method_used", True)
    ]

    foundational_prompt = load_yaml_prompt(f"{foundational_path}/foundational_classifier.yaml")
    foundational_llm = get_llm(
        state, NODE_NAME, "foundational", "classifier"
    ).with_structured_output(FoundationalClassifierSchema)
    foundational_response = foundational_llm.invoke(
        foundational_prompt.format_messages(input=json.dumps(evidence_json))
    )
    foundational_json = _ensure_foundational(_structured_output_to_dict(foundational_response))

    ml_json = {
        "ml_learning_type": [],
        "ml_problem_type": [],
        "deep_learning_used": False,
    }

    if foundational_json.get("foundational_paradigm") in ["Machine Learning", "Mixed"]:
        ml_learning_prompt = load_yaml_prompt(f"{foundational_path}/ml_learning_classifier.yaml")
        ml_learning_llm = get_llm(
            state, NODE_NAME, "ml_learning"
        ).with_structured_output(MLLearningClassifierSchema)
        ml_learning_response = ml_learning_llm.invoke(
            ml_learning_prompt.format_messages(input=json.dumps(evidence_json))
        )
        ml_learning_json = _ensure_ml_learning(_structured_output_to_dict(ml_learning_response))

        ml_problem_prompt = load_yaml_prompt(f"{foundational_path}/ml_problem_classifier.yaml")
        ml_problem_llm = get_llm(
            state, NODE_NAME, "ml_problem"
        ).with_structured_output(MLProblemClassifierSchema)
        ml_problem_response = ml_problem_llm.invoke(
            ml_problem_prompt.format_messages(input=json.dumps(evidence_json))
        )
        ml_problem_json = _ensure_ml_problem(_structured_output_to_dict(ml_problem_response))

        ml_json = {
            "ml_learning_type": ml_learning_json.get("ml_learning_type", []),
            "ml_problem_type": ml_problem_json.get("ml_problem_type", []),
            "deep_learning_used": ml_learning_json.get("deep_learning_used", False),
            "confidence": {
                "learning": ml_learning_json.get("confidence", 0.0),
                "problem": ml_problem_json.get("confidence", 0.0),
            },
            "reasoning_explanation": {
                "learning": ml_learning_json.get("reasoning_explanation", ""),
                "problem": ml_problem_json.get("reasoning_explanation", ""),
            },
            "bibliography": {
                "learning": ml_learning_json.get("bibliography", []),
                "problem": ml_problem_json.get("bibliography", []),
            },
        }

    foundational_audit_prompt = load_yaml_prompt(f"{foundational_path}/auditor.yaml")
    foundational_audit_llm = get_llm(
        state, NODE_NAME, "foundational", "auditor"
    ).with_structured_output(FoundationalAuditSchema)
    foundational_audit_response = foundational_audit_llm.invoke(
        foundational_audit_prompt.format_messages(
            input=json.dumps(
                {
                    "foundational": foundational_json,
                    "ml_details": ml_json,
                    "evidence": evidence_json,
                }
            )
        )
    )
    foundational_audit_json = _ensure_foundational_audit(
        _structured_output_to_dict(foundational_audit_response)
    )

    specialised_context = _retrieve_context(state, specialised_path, "specialised")
    retriever_prompt = load_yaml_prompt(f"{specialised_path}/retriever.yaml")
    specialised_evidence_llm = get_llm(
        state, NODE_NAME, "specialised", "evidence"
    ).with_structured_output(ModellingEvidenceSchema)
    specialised_evidence_response = specialised_evidence_llm.invoke(
        retriever_prompt.format_messages(input=specialised_context)
    )
    specialised_evidence_json = _ensure_modelling_evidence(
        _structured_output_to_dict(specialised_evidence_response)
    )
    specialised_evidence_json["modelling_evidence"] = [
        e for e in specialised_evidence_json["modelling_evidence"]
        if e.get("method_used", True)
    ]

    specialised_prompt = load_yaml_prompt(f"{specialised_path}/specialised_classifier.yaml")
    specialised_llm = get_llm(
        state, NODE_NAME, "specialised", "classifier"
    ).with_structured_output(SpecialisedClassifierSchema)
    specialised_response = specialised_llm.invoke(
        specialised_prompt.format_messages(input=json.dumps(specialised_evidence_json))
    )
    specialised_json = _ensure_specialised(_structured_output_to_dict(specialised_response))

    specialised_audit_prompt = load_yaml_prompt(f"{specialised_path}/auditor.yaml")
    specialised_audit_llm = get_llm(
        state, NODE_NAME, "specialised", "auditor"
    ).with_structured_output(SpecialisedAuditSchema)
    specialised_audit_response = specialised_audit_llm.invoke(
        specialised_audit_prompt.format_messages(
            input=json.dumps(
                {
                    "specialised": specialised_json,
                    "evidence": specialised_evidence_json,
                }
            )
        )
    )
    specialised_audit_json = _ensure_specialised(
        _structured_output_to_dict(specialised_audit_response)
    )

    global_audit_prompt = load_yaml_prompt(f"{base_path}/auditor.yaml")
    global_audit_llm = get_llm(state, NODE_NAME, "auditor").with_structured_output(
        GlobalModellingAuditSchema
    )
    audit_response = global_audit_llm.invoke(
        global_audit_prompt.format_messages(
            input=json.dumps(
                {
                    "foundational": foundational_audit_json,
                    "specialised": specialised_audit_json,
                }
            )
        )
    )
    audit_json = _ensure_global_audit(_structured_output_to_dict(audit_response))

    state["dsrp_outputs"]["modelling"] = audit_json
    state["dsrp_outputs"]["modelling_pre_audit"] = {
        "foundational": foundational_json,
        "ml_details": ml_json,
        "foundational_audit": foundational_audit_json,
        "specialised": specialised_json,
        "specialised_audit": specialised_audit_json,
    }
    return {"dsrp_outputs": state["dsrp_outputs"]}
