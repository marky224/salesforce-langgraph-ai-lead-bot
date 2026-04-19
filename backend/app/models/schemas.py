"""
Pydantic v2 models for the AI Sales Lead Bot.

Defines structured data schemas for lead information, qualification data,
lead scoring, and API request/response payloads. All models use Pydantic v2
with strict validation, serialization aliases for Salesforce field mapping,
and comprehensive docstrings.
"""

from __future__ import annotations

import enum
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, computed_field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConversationStage(str, enum.Enum):
    """Tracks which phase of the sales conversation the user is in."""

    GREETING = "greeting"
    DISCOVERY = "discovery"
    QUALIFICATION = "qualification"
    OBJECTION_HANDLING = "objection_handling"
    LEAD_CAPTURE = "lead_capture"
    CONFIRMATION = "confirmation"
    COMPLETE = "complete"


class BudgetRange(str, enum.Enum):
    """Standardised budget brackets surfaced during qualification."""

    UNDER_10K = "Under $10K"
    TEN_TO_FIFTY_K = "$10K-$50K"
    FIFTY_TO_HUNDRED_K = "$50K-$100K"
    OVER_100K = "$100K+"
    UNKNOWN = "Unknown"


class Timeline(str, enum.Enum):
    """Purchase-readiness timeline captured during qualification."""

    IMMEDIATE = "Immediate"
    ONE_TO_THREE_MONTHS = "1-3 months"
    THREE_TO_SIX_MONTHS = "3-6 months"
    SIX_PLUS_MONTHS = "6+ months"
    JUST_EXPLORING = "Just exploring"
    UNKNOWN = "Unknown"


class CompanySize(str, enum.Enum):
    """Employee-count bracket for the prospect's organisation."""

    MICRO = "1-10"
    SMALL = "11-50"
    MEDIUM = "51-200"
    LARGE = "201-1000"
    ENTERPRISE = "1000+"
    UNKNOWN = "Unknown"


class LeadPriority(str, enum.Enum):
    """Agentforce-assigned priority derived from lead score."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------

class LeadData(BaseModel):
    """
    Contact information collected from the prospect during the chat.

    Fields accumulate incrementally as the conversation progresses;
    every field is optional until the lead-capture stage.
    """

    first_name: Optional[str] = Field(default=None, max_length=40, description="Prospect's first name")
    last_name: Optional[str] = Field(default=None, max_length=80, description="Prospect's last name")
    email: Optional[EmailStr] = Field(default=None, description="Business email address")
    company: Optional[str] = Field(default=None, max_length=255, description="Company or organisation name")
    phone: Optional[str] = Field(default=None, max_length=40, description="Phone number (any format)")
    title: Optional[str] = Field(default=None, max_length=128, description="Job title / role")

    @computed_field
    @property
    def full_name(self) -> str:
        """Convenience accessor: 'First Last' or whichever parts are available."""
        parts = [p for p in (self.first_name, self.last_name) if p]
        return " ".join(parts) if parts else ""

    @computed_field
    @property
    def is_complete(self) -> bool:
        """True when the minimum required fields for a Salesforce Lead exist."""
        return bool(self.last_name and self.email and self.company)

    def to_salesforce_payload(self) -> dict:
        """
        Return a dict ready for the Salesforce Lead create call.

        Only includes fields that have been captured (non-None).
        """
        mapping: dict[str, Optional[str]] = {
            "FirstName": self.first_name,
            "LastName": self.last_name,
            "Email": self.email,
            "Company": self.company,
            "Phone": self.phone,
            "Title": self.title,
            "LeadSource": "Web Chat",
            "Status": "New",
        }
        return {k: v for k, v in mapping.items() if v is not None}


class QualificationData(BaseModel):
    """
    Structured qualification signals extracted from conversation context.

    Populated by the scoring/qualification extraction tool as the
    prospect reveals information during discovery and qualification stages.
    """

    budget_range: BudgetRange = Field(default=BudgetRange.UNKNOWN, description="Stated or inferred budget bracket")
    timeline: Timeline = Field(default=Timeline.UNKNOWN, description="Purchase-readiness timeline")
    company_size: CompanySize = Field(default=CompanySize.UNKNOWN, description="Employee-count bracket")
    pain_points: list[str] = Field(default_factory=list, description="List of pain points surfaced in conversation")
    decision_maker: Optional[bool] = Field(default=None, description="True if the prospect is a decision-maker")
    current_solution: Optional[str] = Field(default=None, max_length=500, description="Tools/solutions currently in use")
    goals: list[str] = Field(default_factory=list, description="Desired outcomes mentioned by the prospect")

    def to_salesforce_fields(self) -> dict:
        """
        Return custom-field values for the Salesforce Lead record.

        Maps qualification data to the custom fields defined on the Lead object.
        """
        fields: dict = {
            "Budget_Range__c": self.budget_range.value if self.budget_range != BudgetRange.UNKNOWN else None,
            "Timeline__c": self.timeline.value if self.timeline != Timeline.UNKNOWN else None,
            "Company_Size__c": self.company_size.value if self.company_size != CompanySize.UNKNOWN else None,
            "Pain_Points__c": "; ".join(self.pain_points) if self.pain_points else None,
        }
        return {k: v for k, v in fields.items() if v is not None}

    @computed_field
    @property
    def known_field_count(self) -> int:
        """Number of qualification fields that have been captured (not unknown/empty)."""
        count = 0
        if self.budget_range != BudgetRange.UNKNOWN:
            count += 1
        if self.timeline != Timeline.UNKNOWN:
            count += 1
        if self.company_size != CompanySize.UNKNOWN:
            count += 1
        if self.pain_points:
            count += 1
        if self.decision_maker is not None:
            count += 1
        if self.current_solution:
            count += 1
        return count


class LeadScore(BaseModel):
    """
    Computed lead score with breakdown rationale.

    Score ranges 0–100. The breakdown dict maps each scoring
    dimension to the points it contributed so the Agentforce agent
    can understand *why* a lead was scored the way it was.
    """

    score: int = Field(default=0, ge=0, le=100, description="Overall lead quality score 0-100")
    breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Points contributed by each scoring dimension",
    )
    priority: LeadPriority = Field(default=LeadPriority.LOW, description="Derived priority tier")

    @classmethod
    def from_score(cls, score: int, breakdown: dict[str, int] | None = None) -> LeadScore:
        """Factory that auto-assigns priority from the numeric score."""
        if score >= 70:
            priority = LeadPriority.HIGH
        elif score >= 40:
            priority = LeadPriority.MEDIUM
        else:
            priority = LeadPriority.LOW
        return cls(score=score, breakdown=breakdown or {}, priority=priority)


# ---------------------------------------------------------------------------
# Salesforce payload models
# ---------------------------------------------------------------------------

class SalesforceLeadPayload(BaseModel):
    """Complete payload assembled right before the Salesforce API call."""

    lead_fields: dict = Field(description="Core Lead field values")
    custom_fields: dict = Field(default_factory=dict, description="Custom field values (Lead_Score__c, etc.)")
    description: str = Field(default="", description="Qualification summary for the Description field")

    def to_api_body(self) -> dict:
        """Merge core + custom fields + description into a single dict."""
        body = {**self.lead_fields, **self.custom_fields}
        if self.description:
            body["Description"] = self.description
        return body


class SalesforceTaskPayload(BaseModel):
    """Payload for creating a Task record linked to a Lead."""

    who_id: str = Field(description="Salesforce Lead ID (WhoId)")
    subject: str = Field(description="Task subject line")
    description: str = Field(description="Full conversation transcript")
    activity_date: date = Field(default_factory=date.today, description="Task due date")
    status: str = Field(default="Completed")
    priority: str = Field(default="Normal")

    def to_api_body(self) -> dict:
        """Return dict ready for the Salesforce Task create call."""
        return {
            "WhoId": self.who_id,
            "Subject": self.subject,
            "Description": self.description,
            "Status": self.status,
            "Priority": self.priority,
            "ActivityDate": self.activity_date.isoformat(),
        }


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Inbound chat message from the frontend widget."""

    message: str = Field(min_length=1, max_length=4000, description="User's message text")
    thread_id: Optional[str] = Field(default=None, description="Conversation thread ID for state continuity")


class ChatResponse(BaseModel):
    """Outbound response sent back to the frontend widget."""

    reply: str = Field(description="Assistant's reply text")
    thread_id: str = Field(description="Thread ID (return to client for subsequent requests)")
    stage: ConversationStage = Field(description="Current conversation stage")
    is_complete: bool = Field(default=False, description="True when the conversation has ended")
    lead_id: Optional[str] = Field(default=None, description="Salesforce Lead ID once created")


class HealthResponse(BaseModel):
    """Health-check endpoint response."""

    status: str = "ok"
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
