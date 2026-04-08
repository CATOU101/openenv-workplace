from __future__ import annotations

from typing import Any


def _clamp_score(score: float) -> float:
    return max(0.01, min(score, 0.99))


def grade_email_triage(task_state: dict[str, Any]) -> tuple[float, str]:
    inbox = task_state["inbox"]
    score = 0.0
    messages: list[str] = []

    for email in inbox:
        status = email["status"]
        category = email["category"]
        if category in {"fyi", "newsletter"} and status == "archived":
            score += 1.0 / 3.0
        elif category == "requires_reply" and status == "replied":
            reply_text = (email.get("reply_text") or "").strip().lower()
            if email["expected_reply"] in reply_text:
                score += 1.0 / 3.0
            else:
                score += 1.0 / 6.0
                messages.append("Vendor email was replied to, but the response missed the requested commitment.")
        else:
            messages.append(f"Email {email['id']} is not resolved correctly yet.")

    return _clamp_score(score), " ".join(messages).strip()


def grade_meeting_scheduling(task_state: dict[str, Any]) -> tuple[float, str]:
    calendar = task_state["calendar"]
    proposed_slot = calendar.get("proposed_slot")
    confirmed = calendar.get("confirmed", False)
    earliest_common_slot = "2026-04-06T10:00"

    if not proposed_slot:
        return _clamp_score(0.0), "No meeting proposal has been created."

    if proposed_slot != earliest_common_slot:
        if confirmed:
            return _clamp_score(0.4), "Meeting was confirmed, but not at the earliest shared slot."
        return _clamp_score(0.25), "A meeting slot was proposed, but it is not the earliest shared availability."

    if confirmed:
        return _clamp_score(1.0), ""
    return _clamp_score(0.7), "Correct slot proposed, but the meeting still needs confirmation."


def grade_data_cleaning(task_state: dict[str, Any]) -> tuple[float, str]:
    cleaned_dataset = task_state.get("cleaned_dataset")
    if not cleaned_dataset:
        return _clamp_score(0.0), "No cleaned dataset has been submitted."

    expected = [
        {"name": "Alice Smith", "email": "alice@example.com", "department": "sales", "status": "active"},
        {"name": "Bob Jones", "email": "bob@example.com", "department": "operations", "status": "inactive"},
        {"name": "Carla Ng", "email": "carla@example.com", "department": "finance", "status": "active"},
        {"name": "Dinesh Patel", "email": "dinesh@example.com", "department": "support", "status": "pending"},
    ]

    normalized = [
        {
            "name": row.get("name"),
            "email": row.get("email"),
            "department": row.get("department"),
            "status": row.get("status"),
        }
        for row in cleaned_dataset
    ]

    matches = sum(1 for row in expected if row in normalized)
    score = matches / len(expected)

    if len(normalized) != len(expected):
        score = min(score, 0.75)
        return _clamp_score(score), "Row count is incorrect after cleaning."

    if score < 1.0:
        return _clamp_score(score), "Dataset is partially cleaned but still missing one or more required fixes."

    return _clamp_score(1.0), ""


GRADERS = {
    "email_triage": grade_email_triage,
    "meeting_scheduling": grade_meeting_scheduling,
    "data_cleaning": grade_data_cleaning,
}
