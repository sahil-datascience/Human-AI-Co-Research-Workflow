from typing import Any, List, Literal

from pydantic import BaseModel, Field, field_validator


ResearchQuestionLabel = Literal[
    "Descriptive",
    "Exploratory",
    "Inferential",
    "Predictive",
    "Causal",
    "Prescriptive",
]


class ResearchQuestionEvidenceItem(BaseModel):
    id: int = Field(default=0)
    page: str = Field(default="")
    section: str = Field(default="")
    direct_quote: str = Field(default="")

    @field_validator("page", "section", "direct_quote", mode="before")
    @classmethod
    def stringify_optional_text(cls, value):
        if value is None:
            return ""
        return str(value)


class ResearchQuestionEvidenceSchema(BaseModel):
    candidate_evidence: List[ResearchQuestionEvidenceItem] = Field(default_factory=list)


class ResearchQuestionClassifierSchema(BaseModel):
    classification: ResearchQuestionLabel = Field(default="Exploratory")
    confidence: float = Field(default=0.0)
    reasoning_explanation: str = Field(default="")
    bibliography: List[ResearchQuestionEvidenceItem] = Field(default_factory=list)

    @field_validator("classification", mode="before")
    @classmethod
    def normalize_label(cls, value):
        return normalize_research_question_label(value) or "Exploratory"


class ResearchQuestionAuditorSchema(BaseModel):
    final_classification: ResearchQuestionLabel = Field(default="Exploratory")
    confidence: float = Field(default=0.0)
    validated_reasoning: str = Field(default="")
    validated_bibliography: List[ResearchQuestionEvidenceItem] = Field(default_factory=list)
    audit_commentary: str = Field(default="")

    @field_validator("final_classification", mode="before")
    @classmethod
    def normalize_label(cls, value):
        return normalize_research_question_label(value) or "Exploratory"


def normalize_research_question_label(value: Any) -> ResearchQuestionLabel | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    aliases = {
        "descriptive": "Descriptive",
        "description": "Descriptive",
        "exploratory": "Exploratory",
        "exploration": "Exploratory",
        "inferential": "Inferential",
        "inference": "Inferential",
        "predictive": "Predictive",
        "prediction": "Predictive",
        "forecasting": "Predictive",
        "causal": "Causal",
        "causality": "Causal",
        "prescriptive": "Prescriptive",
        "prescription": "Prescriptive",
        "optimization": "Prescriptive",
        "optimisation": "Prescriptive",
        "decision-support": "Prescriptive",
        "decision support": "Prescriptive",
    }

    if text in aliases:
        return aliases[text]  # type: ignore[return-value]

    for key, label in aliases.items():
        if key in text:
            return label  # type: ignore[return-value]

    return None
