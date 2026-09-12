from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


PreprocessingStatus = Literal[
    "Not Reported",
    "Mentioned",
    "Transparently Described",
]


class BibliographyItem(BaseModel):
    id: int = Field(default=0, description="Unique incremental identifier starting from 1")
    page: str = Field(
        default="",
        description="Page number or range where the information was found",
    )
    section: str = Field(
        default="",
        description="Name or number of the section/chapter",
    )
    direct_quote: str = Field(
        default="",
        description="Exact verbatim quote from the text source",
    )

    @field_validator("page", "section", "direct_quote", mode="before")
    @classmethod
    def stringify_optional_text(cls, value):
        if value is None:
            return ""
        return str(value)


class CandidateEvidenceSchema(BaseModel):
    candidate_evidence: List[BibliographyItem] = Field(
        default_factory=list,
        description="Verbatim passages that describe data preprocessing before modelling"
    )


class PreprocessingClassificationItem(BaseModel):
    status: PreprocessingStatus = Field(
        default="Not Reported",
        description="Transparency status for the preprocessing sub-component"
    )
    evidence_ids: List[int] = Field(
        default_factory=list,
        description="Evidence IDs from the bibliography that directly support this status"
    )

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value):
        status_map = {
            "not reported": "Not Reported",
            "not-reported": "Not Reported",
            "not_reported": "Not Reported",
            "mentioned": "Mentioned",
            "transparently described": "Transparently Described",
            "transparently-described": "Transparently Described",
            "transparently_described": "Transparently Described",
            "transparent": "Transparently Described",
        }
        if value is None:
            return "Not Reported"
        return status_map.get(str(value).strip().lower(), value)


class DataPreprocessingClassifierSchema(BaseModel):
    data_cleaning: PreprocessingClassificationItem = Field(
        default_factory=PreprocessingClassificationItem,
        description="Classification for data cleaning"
    )
    data_reduction: PreprocessingClassificationItem = Field(
        default_factory=PreprocessingClassificationItem,
        description="Classification for data reduction"
    )
    data_transformation: PreprocessingClassificationItem = Field(
        default_factory=PreprocessingClassificationItem,
        description="Classification for data transformation"
    )
    confidence: float = Field(default=0.0, description="Confidence score from 0.0 to 1.0")
    reasoning_explanation: str = Field(
        default="",
        description="Academic justification for the classifications"
    )
    bibliography: List[BibliographyItem] = Field(
        default_factory=list,
        description="Bibliography items actually used in the classification decisions"
    )


class ValidatedPreprocessingItem(BaseModel):
    status: PreprocessingStatus = Field(
        default="Not Reported",
        description="Audited transparency status for the preprocessing sub-component"
    )
    justification: str = Field(
        default="",
        description="Concise audited rationale for this sub-component's status"
    )

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value):
        status_map = {
            "not reported": "Not Reported",
            "not-reported": "Not Reported",
            "not_reported": "Not Reported",
            "mentioned": "Mentioned",
            "transparently described": "Transparently Described",
            "transparently-described": "Transparently Described",
            "transparently_described": "Transparently Described",
            "transparent": "Transparently Described",
        }
        if value is None:
            return "Not Reported"
        return status_map.get(str(value).strip().lower(), value)


class DataPreprocessingAuditorSchema(BaseModel):
    validated_data_cleaning: ValidatedPreprocessingItem = Field(
        default_factory=ValidatedPreprocessingItem,
        description="Audited classification for data cleaning"
    )
    validated_data_reduction: ValidatedPreprocessingItem = Field(
        default_factory=ValidatedPreprocessingItem,
        description="Audited classification for data reduction"
    )
    validated_data_transformation: ValidatedPreprocessingItem = Field(
        default_factory=ValidatedPreprocessingItem,
        description="Audited classification for data transformation"
    )
    confidence: float = Field(default=0.0, description="Auditor confidence score from 0.0 to 1.0")
    validated_reasoning: str = Field(
        default="",
        description="Concise academic reasoning validating the final labels"
    )
    validated_bibliography: List[BibliographyItem] = Field(
        default_factory=list,
        description="Evidence used to validate the final audited classifications"
    )
    audit_commentary: str = Field(
        default="",
        description="Corrections, downgrades, boundary issues, or other audit notes"
    )
