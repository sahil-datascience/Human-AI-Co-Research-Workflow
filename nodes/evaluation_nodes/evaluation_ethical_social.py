
from utils.load_vector_query import load_vector_query
from utils.paper_retriever import PaperRetriever
from utils.load_yaml_prompt import load_yaml_prompt
from utils.config_llm import set_llm
import json
from utils.dsrp_state import DSRPState
from utils.llm_output_schemas.evaluation_node_schemas import (
    EthicalEvidenceSchema,
    EthicalSocialSchema,
)


def _structured_output_to_dict(output):
    if output is None:
        return {}
    if hasattr(output, "model_dump"):
        return output.model_dump()
    if hasattr(output, "dict"):
        return output.dict()
    return output if isinstance(output, dict) else {}


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


def evaluation_ethical_social_node(state: DSRPState):

    base_path = "prompts/dsrp/evaluation/ethical_social"
    collection_name = state["collection_name"]
    persist_directory = state["persist_directory"]
    embedding_model = state["embedding_model"]
    llm_model = state.get("llm_model")
    llm_reasoning = state.get("llm_reasoning")
    retrieval_k = state.get("reranker_top_k")
    llm = set_llm(model=llm_model, reasoning=llm_reasoning)

    # VECTOR RETRIEVAL
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

    # EVIDENCE EXTRACTION
    retriever_prompt = load_yaml_prompt(
        f"{base_path}/retriever.yaml"
    )

    evidence_llm = llm.with_structured_output(EthicalEvidenceSchema)
    evidence_response = evidence_llm.invoke(
        retriever_prompt.format_messages(input=context_text)
    )

    evidence_json = _structured_output_to_dict(evidence_response)
    if not isinstance(evidence_json.get("ethical_evidence"), list):
        evidence_json["ethical_evidence"] = []

    # CLASSIFICATION
    classifier_prompt = load_yaml_prompt(
        f"{base_path}/classifier.yaml"
    )

    classifier_llm = llm.with_structured_output(EthicalSocialSchema)
    classification_response = classifier_llm.invoke(
        classifier_prompt.format_messages(
            input=json.dumps(evidence_json)
        )
    )

    classification_json = _ensure_ethical_payload(_structured_output_to_dict(classification_response))

    # AUDIT
    auditor_prompt = load_yaml_prompt(
        f"{base_path}/auditor.yaml"
    )

    auditor_llm = llm.with_structured_output(EthicalSocialSchema)
    audit_response = auditor_llm.invoke(
        auditor_prompt.format_messages(
            input=json.dumps(classification_json)
        )
    )

    audit_json = _ensure_ethical_payload(_structured_output_to_dict(audit_response))

    # STORE OUTPUT
    state["dsrp_outputs"]["evaluation_ethical_social"] = audit_json

    return {"dsrp_outputs": state["dsrp_outputs"]}
