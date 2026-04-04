from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from openenv.models import ActionType, TaskDifficulty


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_name: str
    difficulty: TaskDifficulty
    objective: str
    max_steps: int
    available_actions: list[ActionType]
    initial_state: dict[str, Any]


EMAIL_TRIAGE_TASK = TaskSpec(
    task_id="email_triage_easy",
    task_name="Email Triage",
    difficulty=TaskDifficulty.EASY,
    objective=(
        "Review the inbox and resolve each message appropriately: archive FYI items, "
        "reply to the vendor update, and archive the newsletter."
    ),
    max_steps=8,
    available_actions=[
        ActionType.READ_EMAIL,
        ActionType.ARCHIVE_EMAIL,
        ActionType.REPLY_EMAIL,
        ActionType.SUBMIT,
    ],
    initial_state={
        "inbox": [
            {
                "id": "email_1",
                "from": "finance@company.com",
                "subject": "Expense policy reminder",
                "body": "Please note the updated expense submission deadline.",
                "category": "fyi",
                "status": "unread",
            },
            {
                "id": "email_2",
                "from": "vendor@acme.com",
                "subject": "Contract redline received",
                "body": "Can you confirm that legal will review by Friday?",
                "category": "requires_reply",
                "expected_reply": "legal will review by friday",
                "status": "unread",
            },
            {
                "id": "email_3",
                "from": "news@industry-weekly.com",
                "subject": "Top industry stories",
                "body": "Your weekly newsletter is ready.",
                "category": "newsletter",
                "status": "unread",
            },
        ],
        "resolved_ids": [],
    },
)


MEETING_SCHEDULING_TASK = TaskSpec(
    task_id="meeting_scheduling_medium",
    task_name="Meeting Scheduling",
    difficulty=TaskDifficulty.MEDIUM,
    objective=(
        "Schedule a 30-minute project sync for Alice, Bob, and Carla using the earliest "
        "shared availability and then confirm the meeting."
    ),
    max_steps=6,
    available_actions=[
        ActionType.PROPOSE_MEETING,
        ActionType.CONFIRM_MEETING,
        ActionType.SUBMIT,
    ],
    initial_state={
        "calendar": {
            "participants": ["Alice", "Bob", "Carla"],
            "duration_minutes": 30,
            "timezone": "UTC",
            "availability": {
                "Alice": ["2026-04-06T09:00", "2026-04-06T10:00", "2026-04-06T14:00"],
                "Bob": ["2026-04-06T10:00", "2026-04-06T11:00", "2026-04-06T14:00"],
                "Carla": ["2026-04-06T08:30", "2026-04-06T10:00", "2026-04-06T15:00"],
            },
            "proposed_slot": None,
            "confirmed": False,
        }
    },
)


DATA_CLEANING_TASK = TaskSpec(
    task_id="data_cleaning_hard",
    task_name="Data Cleaning",
    difficulty=TaskDifficulty.HARD,
    objective=(
        "Clean the CRM export by removing duplicate rows, normalizing names and emails, "
        "fixing missing departments from known mappings, and standardizing status values."
    ),
    max_steps=7,
    available_actions=[
        ActionType.CLEAN_DATA,
        ActionType.SUBMIT,
    ],
    initial_state={
        "dataset": [
            {"row_id": "r1", "name": " alice  smith ", "email": "ALICE@EXAMPLE.COM ", "department": "sales", "status": "Active"},
            {"row_id": "r2", "name": "Bob Jones", "email": "bob@example.com", "department": "", "status": "inactive"},
            {"row_id": "r3", "name": "Bob Jones", "email": " bob@example.com ", "department": "", "status": "Inactive"},
            {"row_id": "r4", "name": "CARLA NG", "email": "carla@example.com", "department": "finance", "status": "ACTIVE"},
            {"row_id": "r5", "name": "dinesh patel", "email": "dinesh@example.com", "department": None, "status": " pending "},
        ],
        "department_fill_map": {
            "bob@example.com": "operations",
            "dinesh@example.com": "support",
        },
        "cleaned_dataset": None,
    },
)


TASK_SPECS: list[TaskSpec] = [
    EMAIL_TRIAGE_TASK,
    MEETING_SCHEDULING_TASK,
    DATA_CLEANING_TASK,
]


def get_task_specs() -> list[TaskSpec]:
    return deepcopy(TASK_SPECS)
