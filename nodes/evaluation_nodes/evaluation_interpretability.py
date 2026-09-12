
from utils.load_vector_query import load_vector_query
from utils.paper_retriever import PaperRetriever
from utils.load_yaml_prompt import load_yaml_prompt
from utils.config_llm import set_llm
import json
from utils.dsrp_state import DSRPState
from utils.llm_output_schemas.evaluation_node_schemas import (
    InterpretabilityEvidenceSchema,
    InterpretabilitySchema,
)


def _structured_output_to_dict(output):
    if output is None:
        return {}
    if hasattr(output, "model_dump"):
        return output.model_dump()
    if hasattr(output, "dict"):
        return output.dict()
    return output if isinstance(output, dict) else {}


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

def evaluation_interpretability_node(state: DSRPState):

    base_path = "prompts/dsrp/evaluation/epistemological/interpretability"
    collection_name = state["collection_name"]
    persist_directory = state["persist_directory"]
    embedding_model = state["embedding_model"]
    llm_model = state.get("llm_model")
    llm_reasoning = state.get("llm_reasoning")
    retrieval_k = state.get("reranker_top_k")
    llm = set_llm(model=llm_model, reasoning=llm_reasoning)

    # =====================================================
    # 1️⃣ VECTOR RETRIEVAL
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
    # 2️⃣ EVIDENCE EXTRACTION
    # =====================================================

    retriever_prompt = load_yaml_prompt(
        f"{base_path}/retriever.yaml"
    )

    evidence_llm = llm.with_structured_output(InterpretabilityEvidenceSchema)
    evidence_response = evidence_llm.invoke(
        retriever_prompt.format_messages(input=context_text)
    )

    evidence_json = _structured_output_to_dict(evidence_response)
    if not isinstance(evidence_json.get("interpretability_evidence"), list):
        evidence_json["interpretability_evidence"] = []

    # =====================================================
    # 3️⃣ INTERPRETABILITY CLASSIFICATION
    # =====================================================

    classifier_prompt = load_yaml_prompt(
        f"{base_path}/classifier.yaml"
    )

    classifier_llm = llm.with_structured_output(InterpretabilitySchema)
    classification_response = classifier_llm.invoke(
        classifier_prompt.format_messages(
            input=json.dumps(evidence_json)
        )
    )

    classification_json = _ensure_interpretability_payload(_structured_output_to_dict(classification_response))

    # =====================================================
    # 4️⃣ AUDIT VALIDATION
    # =====================================================

    auditor_prompt = load_yaml_prompt(
        f"{base_path}/auditor.yaml"
    )

    auditor_llm = llm.with_structured_output(InterpretabilitySchema)
    audit_response = auditor_llm.invoke(
        auditor_prompt.format_messages(
            input=json.dumps(classification_json)
        )
    )

    audit_json = _ensure_interpretability_payload(_structured_output_to_dict(audit_response))
    if not audit_json.get("interpretability_approach") and audit_json.get("interpretability_method"):
        audit_json["interpretability_approach"] = audit_json["interpretability_method"]

    # =====================================================
    # 5️⃣ STORE OUTPUT
    # =====================================================

    state["dsrp_outputs"]["evaluation_interpretability"] = audit_json

    return {"dsrp_outputs": state["dsrp_outputs"]}
