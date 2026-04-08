from __future__ import annotations

from copy import deepcopy
from hashlib import sha1
from typing import Any

from openenv.graders import GRADERS
from openenv.models import ActionType, OpenEnvAction, OpenEnvObservation, OpenEnvReward
from openenv.rewards import build_reward
from openenv.tasks import TaskSpec, get_task_specs


def _canonical_task_name(task_id: str) -> str:
    if task_id.startswith("email_triage"):
        return "email_triage"
    if task_id.startswith("meeting_scheduling"):
        return "meeting_scheduling"
    if task_id.startswith("data_cleaning"):
        return "data_cleaning"
    raise ValueError(f"Unsupported task id: {task_id}")


TASKS = {
    "email_triage": GRADERS["email_triage"],
    "meeting_scheduling": GRADERS["meeting_scheduling"],
    "data_cleaning": GRADERS["data_cleaning"],
}


class OpenEnvWorkplace:
    def __init__(self, task_name: str = "email_triage") -> None:
        self._task_specs_by_name: dict[str, TaskSpec] = {
            _canonical_task_name(spec.task_id): spec for spec in get_task_specs()
        }
        if task_name not in TASKS:
            raise ValueError(f"Unknown task_name: {task_name}")
        self.task_name: str = task_name
        self._current_task_state: dict[str, Any] = {}
        self._steps_taken: int = 0
        self._done: bool = False
        self._last_feedback: str = ""
        self._last_action_valid: bool = True
        self._last_action_repeated: bool = False
        self._current_score: float = 0.0
        self._episode_id: str = ""
        self._episode_log: list[dict[str, Any]] = []
        self._action_history: list[str] = []
        self.reset()

    def reset(self, task_name: str | None = None) -> OpenEnvObservation:
        if task_name is not None:
            if task_name not in TASKS:
                raise ValueError(f"Unknown task_name: {task_name}")
            self.task_name = task_name

        self._current_task_state = deepcopy(self._current_spec.initial_state)
        self._steps_taken = 0
        self._done = False
        self._last_feedback = "Environment reset."
        self._last_action_valid = True
        self._last_action_repeated = False
        self._current_score = 0.0
        self._action_history = []
        self._episode_log = []
        self._episode_id = self._build_episode_id(self.task_name)
        return self._build_observation()

    def step(self, action: OpenEnvAction) -> tuple[OpenEnvObservation, OpenEnvReward, bool, dict[str, Any]]:
        if self._done:
            reward = build_reward(
                self._current_score,
                self._current_score,
                is_valid=False,
                is_repeated=False,
                steps_taken=self._steps_taken,
                max_steps=self._current_spec.max_steps,
                done=True,
                feedback="Task is already complete. Call reset() before taking more actions.",
            )
            return self._build_observation(), reward, True, self._build_info()

        previous_score = self._current_score
        self._steps_taken += 1
        action_signature = self._serialize_action(action)
        self._last_action_repeated = action_signature in self._action_history
        self._last_action_valid, self._last_feedback = self._apply_action(action)
        self._action_history.append(action_signature)
        current_score, grader_feedback = self._grade_current_task()
        self._current_score = current_score

        if grader_feedback:
            self._last_feedback = f"{self._last_feedback} {grader_feedback}".strip()

        if action.action_type == ActionType.SUBMIT or self._steps_taken >= self._current_spec.max_steps or current_score >= 0.99:
            self._done = True

        reward = build_reward(
            previous_score,
            self._current_score,
            is_valid=self._last_action_valid,
            is_repeated=self._last_action_repeated,
            steps_taken=self._steps_taken,
            max_steps=self._current_spec.max_steps,
            done=self._done,
            feedback=self._last_feedback,
        )
        self._episode_log.append(
            {
                "step": self._steps_taken,
                "action": action.model_dump(mode="json"),
                "valid": self._last_action_valid,
                "repeated": self._last_action_repeated,
                "raw_score": self._current_score,
                "reward": reward.model_dump(mode="json"),
                "feedback": self._last_feedback,
            }
        )
        return self._build_observation(), reward, self._done, self._build_info()

    def state(self) -> OpenEnvObservation:
        return self._build_observation()

    @property
    def task_specs(self) -> list[TaskSpec]:
        return deepcopy(list(self._task_specs_by_name.values()))

    @property
    def _current_spec(self) -> TaskSpec:
        return self._task_specs_by_name[self.task_name]

    def _build_observation(self) -> OpenEnvObservation:
        dataset = self._current_task_state.get("cleaned_dataset") or self._current_task_state.get("dataset") or []
        return OpenEnvObservation(
            task_id=self._current_spec.task_id,
            task_name=self._current_spec.task_name,
            difficulty=self._current_spec.difficulty,
            objective=self._current_spec.objective,
            available_actions=self._current_spec.available_actions,
            episode_id=self._episode_id,
            inbox=deepcopy(self._current_task_state.get("inbox", [])),
            calendar=deepcopy(self._current_task_state.get("calendar", {})),
            dataset_preview=deepcopy(dataset[:5]),
            progress=self._current_score,
            steps_taken=self._steps_taken,
            max_steps=self._current_spec.max_steps,
            action_history=deepcopy(self._action_history),
            last_action_valid=self._last_action_valid,
            last_feedback=self._last_feedback,
        )

    def _build_info(self) -> dict[str, Any]:
        valid_steps = sum(1 for event in self._episode_log if event["valid"])
        invalid_steps = sum(1 for event in self._episode_log if not event["valid"])
        repeated_steps = sum(1 for event in self._episode_log if event["repeated"])
        return {
            "episode_id": self._episode_id,
            "task_id": self._current_spec.task_id,
            "task_name": self._current_spec.task_name,
            "difficulty": self._current_spec.difficulty.value,
            "raw_score": self._current_score,
            "steps_taken": self._steps_taken,
            "max_steps": self._current_spec.max_steps,
            "valid_steps": valid_steps,
            "invalid_steps": invalid_steps,
            "repeated_actions": repeated_steps,
            "done": self._done,
            "action_history": deepcopy(self._action_history),
            "episode_log": deepcopy(self._episode_log),
        }

    def _build_episode_id(self, task_id: str) -> str:
        seed = f"{task_id}:{len(self._task_specs_by_name)}"
        return sha1(seed.encode("utf-8")).hexdigest()[:12]

    def _serialize_action(self, action: OpenEnvAction) -> str:
        return f"{action.action_type.value}|{action.target_id}|{sorted(action.payload.items())}"

    def evaluate(self) -> tuple[float, str]:
        if self.task_name not in TASKS:
            raise ValueError(f"No grader defined for task {self.task_name}")
        score, message = TASKS[self.task_name](self._current_task_state)
        return score, message

    def _grade_current_task(self) -> tuple[float, str]:
        return self.evaluate()

    def _apply_action(self, action: OpenEnvAction) -> tuple[bool, str]:
        if action.action_type not in self._current_spec.available_actions:
            return False, f"Action {action.action_type.value} is not available for this task."

        if self.task_name == "email_triage":
            return self._apply_email_action(action)
        if self.task_name == "meeting_scheduling":
            return self._apply_meeting_action(action)
        if self.task_name == "data_cleaning":
            return self._apply_data_action(action)
        return False, "Unsupported task."

    def _apply_email_action(self, action: OpenEnvAction) -> tuple[bool, str]:
        inbox = self._current_task_state["inbox"]
        if action.action_type == ActionType.SUBMIT:
            return True, "Submitted email triage results for grading."

        target = next((email for email in inbox if email["id"] == action.target_id), None)
        if target is None:
            return False, "Target email was not found."

        if action.action_type == ActionType.READ_EMAIL:
            target["status"] = "read"
            return True, f"Opened {target['subject']}."

        if action.action_type == ActionType.ARCHIVE_EMAIL:
            target["status"] = "archived"
            return True, f"Archived {target['subject']}."

        if action.action_type == ActionType.REPLY_EMAIL:
            reply_text = str(action.payload.get("reply_text", "")).strip()
            if not reply_text:
                return False, "Reply email action requires payload.reply_text."
            target["status"] = "replied"
            target["reply_text"] = reply_text
            return True, f"Replied to {target['subject']}."

        return False, "Unsupported email action."

    def _apply_meeting_action(self, action: OpenEnvAction) -> tuple[bool, str]:
        calendar = self._current_task_state["calendar"]

        if action.action_type == ActionType.SUBMIT:
            return True, "Submitted meeting schedule for grading."

        if action.action_type == ActionType.PROPOSE_MEETING:
            proposed_slot = str(action.payload.get("slot", "")).strip()
            if not proposed_slot:
                return False, "Propose meeting action requires payload.slot."
            calendar["proposed_slot"] = proposed_slot
            return True, f"Proposed meeting at {proposed_slot}."

        if action.action_type == ActionType.CONFIRM_MEETING:
            if not calendar.get("proposed_slot"):
                return False, "A meeting must be proposed before it can be confirmed."
            calendar["confirmed"] = True
            return True, f"Confirmed meeting at {calendar['proposed_slot']}."

        return False, "Unsupported scheduling action."

    def _apply_data_action(self, action: OpenEnvAction) -> tuple[bool, str]:
        if action.action_type == ActionType.SUBMIT:
            return True, "Submitted cleaned dataset for grading."

        if action.action_type != ActionType.CLEAN_DATA:
            return False, "Unsupported data action."

        operation = action.payload.get("operation")
        dataset = deepcopy(self._current_task_state["dataset"])
        fill_map = self._current_task_state["department_fill_map"]

        if operation == "trim_whitespace":
            for row in dataset:
                row["name"] = str(row["name"]).strip()
                row["email"] = str(row["email"]).strip()
                row["status"] = str(row["status"]).strip()
            self._current_task_state["dataset"] = dataset
            return True, "Trimmed whitespace in text fields."

        if operation == "normalize_case":
            for row in dataset:
                row["name"] = str(row["name"]).title()
                row["email"] = str(row["email"]).lower()
                row["status"] = str(row["status"]).strip().lower()
            self._current_task_state["dataset"] = dataset
            return True, "Normalized case in names, emails, and statuses."

        if operation == "fill_departments":
            for row in dataset:
                email = str(row["email"]).strip().lower()
                if row.get("department") in ("", None):
                    row["department"] = fill_map.get(email, row.get("department"))
            self._current_task_state["dataset"] = dataset
            return True, "Filled missing departments from the known mapping."

        if operation == "deduplicate":
            deduped: list[dict[str, Any]] = []
            seen_emails: set[str] = set()
            for row in dataset:
                email = str(row["email"]).strip().lower()
                if email not in seen_emails:
                    deduped.append(row)
                    seen_emails.add(email)
            self._current_task_state["dataset"] = deduped
            return True, "Removed duplicate contacts by email."

        if operation == "finalize":
            finalized = []
            for row in dataset:
                finalized.append(
                    {
                        "name": str(row["name"]).strip().title(),
                        "email": str(row["email"]).strip().lower(),
                        "department": row.get("department"),
                        "status": str(row["status"]).strip().lower(),
                    }
                )
            self._current_task_state["cleaned_dataset"] = finalized
            return True, "Stored cleaned dataset snapshot for grading."

        return False, "Unknown clean_data operation."
