from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


GatekeeperLabel = Literal["Include", "Exclude", "Borderline"]


class GatekeeperEvidenceItem(BaseModel):
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


class GatekeeperEvidenceSchema(BaseModel):
    candidate_evidence: List[GatekeeperEvidenceItem] = Field(default_factory=list)


class GatekeeperClassifierSchema(BaseModel):
    paper_id: str = Field(default="")
    classification: GatekeeperLabel = Field(default="Borderline")
    inclusion_basis: str = Field(default="")
    identified_modelling_paradigm: str = Field(default="")
    confidence: float = Field(default=0.0)
    reasoning_explanation: str = Field(default="")
    bibliography: List[GatekeeperEvidenceItem] = Field(default_factory=list)
    borderline_explanation_if_applicable: str = Field(default="")

    @field_validator("classification", mode="before")
    @classmethod
    def normalize_label(cls, value):
        if value is None:
            return "Borderline"
        text = str(value).strip().lower()
        if "include" in text or text in {"yes", "eligible", "in-scope", "inscope"}:
            return "Include"
        if "exclude" in text or text in {"no", "out-of-scope", "out of scope"}:
            return "Exclude"
        if "border" in text or text in {"uncertain", "mixed"}:
            return "Borderline"
        return "Borderline"


class GatekeeperAuditorSchema(BaseModel):
    final_classification: GatekeeperLabel = Field(default="Borderline")
    confidence: float = Field(default=0.0)
    validated_reasoning: str = Field(default="")
    validated_bibliography: List[GatekeeperEvidenceItem] = Field(default_factory=list)
    audit_commentary: str = Field(default="")

    @field_validator("final_classification", mode="before")
    @classmethod
    def normalize_label(cls, value):
        return GatekeeperClassifierSchema.normalize_label(value)
