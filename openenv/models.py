from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ActionType(str, Enum):
    READ_EMAIL = "read_email"
    ARCHIVE_EMAIL = "archive_email"
    REPLY_EMAIL = "reply_email"
    PROPOSE_MEETING = "propose_meeting"
    CONFIRM_MEETING = "confirm_meeting"
    CLEAN_DATA = "clean_data"
    SUBMIT = "submit"


class OpenEnvObservation(BaseModel):
    task_id: str
    task_name: str
    difficulty: TaskDifficulty
    objective: str
    available_actions: list[ActionType]
    episode_id: str
    inbox: list[dict[str, Any]] = Field(default_factory=list)
    calendar: dict[str, Any] = Field(default_factory=dict)
    dataset_preview: list[dict[str, Any]] = Field(default_factory=list)
    progress: float = 0.0
    steps_taken: int = 0
    max_steps: int
    action_history: list[str] = Field(default_factory=list)
    last_action_valid: bool = True
    last_feedback: str = ""


class OpenEnvAction(BaseModel):
    action_type: ActionType
    target_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class OpenEnvReward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    delta: float
    invalid_action_penalty: float = 0.0
    repeated_action_penalty: float = 0.0
    efficiency_penalty: float = 0.0
    completion_bonus: float = 0.0
    message: str
