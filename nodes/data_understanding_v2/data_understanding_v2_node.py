from __future__ import annotations

import json
from pathlib import Path

from utils.config_llm import set_llm
from utils.dsrp_state import DSRPState
from utils.load_yaml_prompt import load_yaml_prompt
from utils.parse_llm_json import parse_llm_json

from .data_category_node import data_category_node
from .data_characteristics_node import data_characteristics_node
from .data_format_node import data_format_node


def data_understanding_v2_node(state: DSRPState):
    dimension_name = "data_understanding"
    llm = set_llm(model=state.get("llm_model"), reasoning=state.get("llm_reasoning"))
    state.setdefault("dsrp_outputs", {})

    category_result = data_category_node(state)
    format_result = data_format_node(state)
    characteristics_result = data_characteristics_node(state)

    category_audit = category_result["data_category"]
    format_audit = format_result["data_format"]
    characteristics_audit = characteristics_result["data_characteristics"]

    evidence_json = {
        "data_category": category_result["data_category_pre_audit"]["evidence"],
        "data_format": format_result["data_format_pre_audit"]["evidence"],
        "data_characteristics": characteristics_result["data_characteristics_pre_audit"]["evidence"],
    }

    combined_classification = {
        "data_category": category_audit.get("data_category", []),
        "data_format": format_audit.get("data_format", []),
        "data_characteristics": characteristics_audit.get("data_characteristics", []),
        "confidence": max(
            float(category_audit.get("confidence", 0.0) or 0.0),
            float(format_audit.get("confidence", 0.0) or 0.0),
            float(characteristics_audit.get("confidence", 0.0) or 0.0),
        ),
        "reasoning_explanation": "",
        "evidence": evidence_json,
    }

    auditor_prompt = load_yaml_prompt(str(Path("prompts/dsrp/data_understanding/v2/auditor.yaml")))
    auditor_input = {"classification_outputs": combined_classification, "evidence": evidence_json}

    audit_response = llm.invoke(auditor_prompt.format_messages(input=json.dumps(auditor_input)))
    audit_json = parse_llm_json(audit_response.content)

    state["dsrp_outputs"][f"{dimension_name}_pre_audit"] = {
        "classification_outputs": {
            "data_category": category_result,
            "data_format": format_result,
            "data_characteristics": characteristics_result,
        },
        "combined_classification": combined_classification,
        "evidence": evidence_json,
    }
    state["dsrp_outputs"][dimension_name] = audit_json

    return {"dsrp_outputs": state["dsrp_outputs"]}


def data_understanding_node(state: DSRPState):
    return data_understanding_v2_node(state)
