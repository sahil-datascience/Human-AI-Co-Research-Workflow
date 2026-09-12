from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from utils.dsrp_state import DSRPState
from utils.load_yaml_prompt import load_yaml_prompt
from utils.parse_llm_json import parse_llm_json

from .config import get_llm
from .retrieval import format_docs_context, retrieve_and_rerank


NODE_NAME = "data_understanding"


def _run_prompt_pair(llm, retriever_path: str, classifier_path: str, context_text: str):
    retriever_prompt = load_yaml_prompt(retriever_path)
    retrieval_response = llm.invoke(retriever_prompt.format_messages(input=context_text))
    evidence_json = parse_llm_json(retrieval_response.content)

    classifier_prompt = load_yaml_prompt(classifier_path)
    classification_response = llm.invoke(
        classifier_prompt.format_messages(input=json.dumps(evidence_json))
    )
    classification_json = parse_llm_json(classification_response.content)

    return evidence_json, classification_json


def _run_classifier_only(llm, classifier_path: str, context_text: str):
    classifier_prompt = load_yaml_prompt(classifier_path)
    classification_response = llm.invoke(classifier_prompt.format_messages(input=context_text))
    return parse_llm_json(classification_response.content)


def _run_prompt_pairs_parallel(jobs, context_text: str, max_workers: int):
    evidence_by_key = {}
    classification_by_key = {}

    worker_count = max(1, min(max_workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_key = {
            executor.submit(_run_prompt_pair, llm, retriever_path, classifier_path, context_text): key
            for key, llm, retriever_path, classifier_path in jobs
        }

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                evidence_json, classification_json = future.result()
            except Exception as exc:
                raise RuntimeError(f"Parallel prompt pair failed for '{key}': {exc}") from exc
            evidence_by_key[key] = evidence_json
            classification_by_key[key] = classification_json

    return evidence_by_key, classification_by_key


def _run_classifiers_parallel(jobs, context_text: str, max_workers: int):
    classification_by_key = {}

    worker_count = max(1, min(max_workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_key = {
            executor.submit(_run_classifier_only, llm, classifier_path, context_text): key
            for key, llm, classifier_path in jobs
        }

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                classification_json = future.result()
            except Exception as exc:
                raise RuntimeError(f"Parallel classifier failed for '{key}': {exc}") from exc
            classification_by_key[key] = classification_json

    return classification_by_key


def data_understanding_node(state: DSRPState):
    parallel_workers = max(1, int(os.getenv("DSRP_PARALLEL_WORKERS", "4")))

    docs = retrieve_and_rerank(
        state,
        NODE_NAME,
        f"prompts/dsrp/{NODE_NAME}/vector_query.yaml",
        default_reranker_top_k=8,
    )
    context_text = format_docs_context(docs)

    evidence_json = {}
    base_retriever_path = Path(f"prompts/dsrp/{NODE_NAME}/retriever.yaml")
    if base_retriever_path.exists():
        retriever_prompt = load_yaml_prompt(str(base_retriever_path))
        llm = get_llm(state, NODE_NAME, "evidence")
        evidence_response = llm.invoke(retriever_prompt.format_messages(input=context_text))
        evidence_json = parse_llm_json(evidence_response.content)

    dimension_configs = [
        ("category", "data_category"),
        ("format", "data_format"),
    ]

    classification_outputs = {}
    per_dimension_evidence = {}

    base_jobs = []
    for dimension_folder, output_key in dimension_configs:
        retriever_path = f"prompts/dsrp/{NODE_NAME}/{dimension_folder}/retriever.yaml"
        classifier_path = f"prompts/dsrp/{NODE_NAME}/{dimension_folder}/classifier.yaml"
        base_jobs.append((output_key, get_llm(state, NODE_NAME, output_key), retriever_path, classifier_path))

    base_evidence, base_classifications = _run_prompt_pairs_parallel(
        jobs=base_jobs,
        context_text=context_text,
        max_workers=parallel_workers,
    )
    per_dimension_evidence.update(base_evidence)
    classification_outputs.update(base_classifications)

    characteristics_base = Path(f"prompts/dsrp/{NODE_NAME}/characteristics")
    characteristics_label_units = [
        ("temporal", "Temporal"),
        ("spatial", "Spatial"),
        ("textual", "Textual"),
        ("visual", "Visual"),
        ("networked", "Networked"),
    ]
    split_prompt_available = all(
        (characteristics_base / folder / "classifier.yaml").exists()
        for folder, _ in characteristics_label_units
    )

    if split_prompt_available:
        characteristic_jobs = []
        for folder, label_name in characteristics_label_units:
            classifier_path = str(characteristics_base / folder / "classifier.yaml")
            llm = get_llm(state, NODE_NAME, "data_characteristics", folder)
            characteristic_jobs.append((label_name, llm, classifier_path))

        characteristics_by_label = _run_classifiers_parallel(
            jobs=characteristic_jobs,
            context_text=context_text,
            max_workers=parallel_workers,
        )

        characteristics_auditor_path = characteristics_base / "auditor.yaml"
        if characteristics_auditor_path.exists():
            characteristics_auditor_prompt = load_yaml_prompt(str(characteristics_auditor_path))
            characteristics_auditor_input = {
                "label_outputs": characteristics_by_label,
                "labels": [label_name for _, label_name in characteristics_label_units],
            }
            llm = get_llm(state, NODE_NAME, "data_characteristics", "auditor")
            characteristics_audit_response = llm.invoke(
                characteristics_auditor_prompt.format_messages(
                    input=json.dumps(characteristics_auditor_input)
                )
            )
            characteristics_audit_json = parse_llm_json(characteristics_audit_response.content)

            audited_labels = characteristics_audit_json.get("data_characteristics", [])
            audited_confidence = float(characteristics_audit_json.get("confidence", 0.0) or 0.0)
            audited_reasoning = characteristics_audit_json.get("validated_reasoning", "")
            audited_bibliography = characteristics_audit_json.get("validated_bibliography", [])
        else:
            audited_labels = None
            audited_confidence = 0.0
            audited_reasoning = ""
            audited_bibliography = []

        selected_labels = []
        max_conf = 0.0
        bibliography = []
        for _, label_name in characteristics_label_units:
            out = characteristics_by_label.get(label_name, {})
            label_biblio = out.get("bibliography", [])
            label_present = bool(out.get("is_present", False))
            label_supported = bool(label_biblio)

            if label_present and label_supported:
                selected_labels.append(label_name)
                bibliography.extend(label_biblio)

            max_conf = max(max_conf, float(out.get("confidence", 0.0) or 0.0))

        if isinstance(audited_labels, list):
            selected_labels = [
                label for label in audited_labels
                if label in {label_name for _, label_name in characteristics_label_units}
            ]
            bibliography = audited_bibliography if isinstance(audited_bibliography, list) else bibliography
            max_conf = max(max_conf, audited_confidence)
            reasoning_explanation = (
                audited_reasoning
                if isinstance(audited_reasoning, str) and audited_reasoning.strip()
                else "Validated by characteristics-level auditor and aggregated from split label-specific classifiers."
            )
        else:
            reasoning_explanation = "Strictly aggregated from split label-specific classifiers with explicit-evidence gating."

        classification_outputs["data_characteristics"] = {
            "data_characteristics": selected_labels,
            "confidence": max_conf,
            "reasoning_explanation": reasoning_explanation,
            "bibliography": bibliography,
        }
        per_dimension_evidence["data_characteristics"] = {
            "by_label": {
                label_name: characteristics_by_label.get(label_name, {}).get("bibliography", [])
                for _, label_name in characteristics_label_units
            },
            "per_label_classifiers": characteristics_by_label,
        }
    else:
        legacy_retriever_path = f"prompts/dsrp/{NODE_NAME}/characteristics/retriever.yaml"
        legacy_classifier_path = f"prompts/dsrp/{NODE_NAME}/characteristics/classifier.yaml"
        characteristics_evidence_json, characteristics_classification_json = _run_prompt_pair(
            llm=get_llm(state, NODE_NAME, "data_characteristics"),
            retriever_path=legacy_retriever_path,
            classifier_path=legacy_classifier_path,
            context_text=context_text,
        )
        per_dimension_evidence["data_characteristics"] = characteristics_evidence_json
        classification_outputs["data_characteristics"] = characteristics_classification_json

    if not evidence_json:
        evidence_json = {
            "data_category": per_dimension_evidence.get("data_category", {}),
            "data_format": per_dimension_evidence.get("data_format", {}),
            "data_characteristics": per_dimension_evidence.get("data_characteristics", {}),
        }

    combined_classification = {
        "data_category": classification_outputs.get("data_category", {}).get("data_category", []),
        "data_format": classification_outputs.get("data_format", {}).get("data_format", []),
        "data_characteristics": classification_outputs.get("data_characteristics", {}).get("data_characteristics", []),
        "confidence": max(
            classification_outputs.get("data_category", {}).get("confidence", 0),
            classification_outputs.get("data_format", {}).get("confidence", 0),
            classification_outputs.get("data_characteristics", {}).get("confidence", 0),
        ),
        "reasoning_explanation": "",
        "evidence": evidence_json,
    }

    auditor_prompt = load_yaml_prompt(f"prompts/dsrp/{NODE_NAME}/auditor.yaml")
    auditor_input = {
        "classification_outputs": combined_classification,
        "evidence": evidence_json,
    }
    llm = get_llm(state, NODE_NAME, "auditor")
    audit_response = llm.invoke(auditor_prompt.format_messages(input=json.dumps(auditor_input)))
    audit_json = parse_llm_json(audit_response.content)

    state["dsrp_outputs"][f"{NODE_NAME}_pre_audit"] = {
        "classification_outputs": classification_outputs,
        "combined_classification": combined_classification,
        "evidence": evidence_json,
    }
    state["dsrp_outputs"][NODE_NAME] = audit_json

    return {"dsrp_outputs": state["dsrp_outputs"]}
