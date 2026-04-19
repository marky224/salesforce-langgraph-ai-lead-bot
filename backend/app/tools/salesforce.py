"""
Salesforce integration tools for the AI Sales Lead Bot.

Provides async-friendly wrappers around ``simple_salesforce`` to create
Lead and Task records.  Authentication supports two flows:

1. **Username-Password Flow** — uses SF_USERNAME + SF_PASSWORD +
   SF_SECURITY_TOKEN.  Simplest for development.
2. **Client Credentials Flow** — uses SF_CLIENT_ID + SF_CLIENT_SECRET
   with a Connected App configured for server-to-server OAuth.
   Preferred for production (no user interaction required).

The module maintains a singleton ``Salesforce`` client that is lazily
initialised on first use and cached for the process lifetime.

Usage::

    from app.tools.salesforce import create_lead, create_transcript_task

    lead_id = await create_lead(
        lead_data={"last_name": "Chen", "email": "s@acme.com", "company": "Acme"},
        qualification_data={"budget_range": "$10K-$50K", "pain_points": ["slow CRM"]},
        lead_score=72,
        description="Prospect is looking for a CRM replacement...",
    )

    task_id = await create_transcript_task(
        lead_id=lead_id,
        transcript="Visitor: Hi...\nAlex: Welcome!...",
    )
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Salesforce client singleton
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_sf_client() -> Any:
    """
    Create and cache a ``simple_salesforce.Salesforce`` client.

    Tries Client Credentials Flow first (if SF_CLIENT_ID and
    SF_CLIENT_SECRET are set), then falls back to Username-Password Flow.

    Returns
    -------
    Salesforce
        An authenticated ``simple_salesforce.Salesforce`` instance.

    Raises
    ------
    ImportError
        If ``simple_salesforce`` is not installed.
    ValueError
        If insufficient credentials are configured.
    """
    try:
        from simple_salesforce import Salesforce
    except ImportError as exc:
        raise ImportError(
            "simple_salesforce is required for Salesforce integration. "
            "Install it with: pip install simple-salesforce"
        ) from exc

    from app.config import get_settings

    s = get_settings()

    if not s.salesforce_configured:
        raise ValueError(
            "Salesforce credentials are not fully configured. "
            "Set SF_CLIENT_ID, SF_CLIENT_SECRET, SF_USERNAME, and SF_PASSWORD "
            "in your environment variables."
        )

    # --- Attempt Client Credentials Flow ---
    if s.sf_client_id and s.sf_client_secret:
        try:
            sf = Salesforce(
                username=s.sf_username,
                password=s.sf_password.get_secret_value(),
                security_token=(
                    s.sf_security_token.get_secret_value()
                    if s.sf_security_token
                    else ""
                ),
                consumer_key=s.sf_client_id.get_secret_value(),
                consumer_secret=s.sf_client_secret.get_secret_value(),
                domain=(
                    "test"
                    if "sandbox" in s.sf_instance_url.lower()
                    or "test" in s.sf_instance_url.lower()
                    else "login"
                ),
            )
            logger.info(
                "Salesforce client authenticated via Connected App (user: %s)",
                s.sf_username,
            )
            return sf
        except Exception:
            logger.warning(
                "Connected App auth failed, falling back to username-password flow",
                exc_info=True,
            )

    # --- Fallback: Username-Password Flow ---
    sf = Salesforce(
        username=s.sf_username,
        password=s.sf_password.get_secret_value(),
        security_token=(
            s.sf_security_token.get_secret_value()
            if s.sf_security_token
            else ""
        ),
        domain=(
            "test"
            if "sandbox" in s.sf_instance_url.lower()
            or "test" in s.sf_instance_url.lower()
            else "login"
        ),
    )
    logger.info(
        "Salesforce client authenticated via username-password (user: %s)",
        s.sf_username,
    )
    return sf


def reset_sf_client() -> None:
    """Clear the cached Salesforce client (useful for testing or re-auth)."""
    _get_sf_client.cache_clear()
    logger.info("Salesforce client cache cleared")


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------

async def _run_sync(func: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Run a synchronous ``simple_salesforce`` call in a thread pool so it
    doesn't block the async event loop.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ---------------------------------------------------------------------------
# Lead creation
# ---------------------------------------------------------------------------

async def create_lead(
    lead_data: dict[str, Any],
    qualification_data: dict[str, Any],
    lead_score: int = 0,
    description: str = "",
) -> str:
    """
    Create a Lead record in Salesforce.

    Merges core contact fields, custom qualification fields, and the
    lead score into a single API payload.

    Parameters
    ----------
    lead_data : dict
        Contact info: first_name, last_name, email, company, phone, title.
    qualification_data : dict
        Qualification signals: budget_range, timeline, company_size,
        pain_points, decision_maker, current_solution.
    lead_score : int
        Computed lead quality score (0–100).
    description : str
        Transcript summary for the Lead Description field.

    Returns
    -------
    str
        The Salesforce Lead record ID (18-char).

    Raises
    ------
    SalesforceError
        If the API call fails.
    """
    sf = _get_sf_client()

    # --- Build core fields ---
    payload: dict[str, Any] = {
        "LeadSource": "Web Chat",
        "Status": "New",
    }

    # Map lead_data keys to Salesforce field names
    field_map = {
        "first_name": "FirstName",
        "last_name": "LastName",
        "email": "Email",
        "company": "Company",
        "phone": "Phone",
        "title": "Title",
    }
    for key, sf_field in field_map.items():
        value = lead_data.get(key)
        if value:
            payload[sf_field] = value

    # Salesforce requires LastName and Company at minimum
    if "LastName" not in payload:
        payload["LastName"] = "Unknown"
    if "Company" not in payload:
        payload["Company"] = lead_data.get("company", "Unknown")

    # --- Add custom fields ---
    qd = qualification_data

    if qd.get("budget_range") and qd["budget_range"] != "Unknown":
        payload["Budget_Range__c"] = qd["budget_range"]

    if qd.get("timeline") and qd["timeline"] != "Unknown":
        payload["Timeline__c"] = qd["timeline"]

    if qd.get("company_size") and qd["company_size"] != "Unknown":
        payload["Company_Size__c"] = qd["company_size"]

    if qd.get("pain_points"):
        # Join pain points into a semicolon-separated string
        # (Long Text Area field, max 32000 chars)
        pain_text = "; ".join(qd["pain_points"])
        payload["Pain_Points__c"] = pain_text[:32000]

    if lead_score > 0:
        payload["Lead_Score__c"] = lead_score

    if description:
        payload["Description"] = description[:32000]

    # --- Create the record ---
    logger.info(
        "Creating Salesforce Lead: %s %s at %s",
        payload.get("FirstName", ""),
        payload.get("LastName", ""),
        payload.get("Company", ""),
    )
    logger.debug("Lead payload: %s", payload)

    result = await _run_sync(sf.Lead.create, payload)

    if not result.get("success"):
        errors = result.get("errors", [])
        raise RuntimeError(f"Salesforce Lead creation failed: {errors}")

    lead_id = result["id"]
    logger.info("Salesforce Lead created successfully: %s", lead_id)
    return lead_id


# ---------------------------------------------------------------------------
# Task creation (transcript attachment)
# ---------------------------------------------------------------------------

async def create_transcript_task(
    lead_id: str,
    transcript: str,
    subject: Optional[str] = None,
) -> str:
    """
    Create a Task record linked to a Lead containing the chat transcript.

    Parameters
    ----------
    lead_id : str
        The Salesforce Lead ID to link the Task to (WhoId).
    transcript : str
        Full conversation transcript text.
    subject : str | None
        Task subject line.  Defaults to
        ``"AI Chat Transcript - {date}"``.

    Returns
    -------
    str
        The Salesforce Task record ID (18-char).

    Raises
    ------
    SalesforceError
        If the API call fails.
    """
    sf = _get_sf_client()

    if subject is None:
        subject = f"AI Chat Transcript - {date.today().isoformat()}"

    payload = {
        "WhoId": lead_id,
        "Subject": subject,
        "Description": transcript[:32000],
        "Status": "Completed",
        "Priority": "Normal",
        "ActivityDate": date.today().isoformat(),
        "Type": "Other",
    }

    logger.info("Creating Salesforce Task for Lead %s: %s", lead_id, subject)
    logger.debug("Task payload: %s", payload)

    result = await _run_sync(sf.Task.create, payload)

    if not result.get("success"):
        errors = result.get("errors", [])
        raise RuntimeError(f"Salesforce Task creation failed: {errors}")

    task_id = result["id"]
    logger.info("Salesforce Task created successfully: %s", task_id)

    # --- Update Lead with the transcript Task ID ---
    try:
        await _run_sync(
            sf.Lead.update,
            lead_id,
            {"Chat_Transcript_ID__c": task_id},
        )
        logger.info("Lead %s updated with Chat_Transcript_ID__c: %s", lead_id, task_id)
    except Exception:
        # Non-fatal — the Task is already linked via WhoId
        logger.warning(
            "Failed to update Chat_Transcript_ID__c on Lead %s (non-fatal)",
            lead_id,
            exc_info=True,
        )

    return task_id


# ---------------------------------------------------------------------------
# Utility: verify Salesforce connection
# ---------------------------------------------------------------------------

async def verify_connection() -> dict[str, Any]:
    """
    Test the Salesforce connection by querying org limits.

    Returns a dict with connection status and org info.  Useful for
    health checks and startup verification.
    """
    try:
        sf = _get_sf_client()
        limits = await _run_sync(lambda: sf.limits())

        daily_api = limits.get("DailyApiRequests", {})
        return {
            "connected": True,
            "instance_url": sf.sf_instance,
            "api_calls_remaining": daily_api.get("Remaining", "unknown"),
            "api_calls_max": daily_api.get("Max", "unknown"),
        }
    except Exception as exc:
        logger.error("Salesforce connection verification failed: %s", exc)
        return {
            "connected": False,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Utility: clean up test records
# ---------------------------------------------------------------------------

async def delete_record(sobject: str, record_id: str) -> bool:
    """
    Delete a Salesforce record by type and ID.

    Intended for test cleanup only — not used in production flows.

    Parameters
    ----------
    sobject : str
        Salesforce object type (e.g. "Lead", "Task").
    record_id : str
        The 18-character record ID.

    Returns
    -------
    bool
        True if deletion succeeded.
    """
    try:
        sf = _get_sf_client()
        await _run_sync(getattr(sf, sobject).delete, record_id)
        logger.info("Deleted %s record: %s", sobject, record_id)
        return True
    except Exception:
        logger.warning("Failed to delete %s %s", sobject, record_id, exc_info=True)
        return False
