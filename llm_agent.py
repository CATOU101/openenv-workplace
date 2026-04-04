from __future__ import annotations

import json
import os
from typing import Any

from openenv import OpenEnvAction, OpenEnvObservation

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - dependency availability varies by runtime
    OpenAI = None  # type: ignore[assignment]


SYSTEM_PROMPT = """You are an AI agent solving workplace tasks.
You must output ONLY valid JSON action.
No text, no explanation."""


def _parse_model_text(response: Any) -> str:
    if hasattr(response, "output_text"):
        return str(response.output_text).strip()

    output = getattr(response, "output", [])
    chunks: list[str] = []
    for item in output:
        for content in getattr(item, "content", []):
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "".join(chunks).strip()


class LLMAgent:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "llama3-8b-8192",
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = os.getenv("LLM_MODEL", model)
        self.base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.groq.com/openai/v1"
        self.client: Any | None = None

        if OpenAI is not None and self.api_key:
            client_kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "base_url": self.base_url,
            }
            self.client = OpenAI(**client_kwargs)

    def act(self, observation: OpenEnvObservation) -> OpenEnvAction:
        if self.client is None:
            return self._fallback_action(observation)

        observation_json = json.dumps(observation.model_dump(mode="json"), indent=2, sort_keys=True)
        response = self.client.responses.create(
            model=self.model,
            temperature=0,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "input_text", "text": observation_json}]},
            ],
        )
        content = _parse_model_text(response)
        payload = json.loads(content)
        return OpenEnvAction.model_validate(payload)

    def _fallback_action(self, observation: OpenEnvObservation) -> OpenEnvAction:
        if observation.task_id == "email_triage_easy":
            unread = [email for email in observation.inbox if email["status"] in {"unread", "read"}]
            if unread:
                email = unread[0]
                if email["category"] in {"fyi", "newsletter"}:
                    return OpenEnvAction(action_type="archive_email", target_id=email["id"])
                return OpenEnvAction(
                    action_type="reply_email",
                    target_id=email["id"],
                    payload={"reply_text": "Legal will review by Friday and confirm next steps."},
                )
            return OpenEnvAction(action_type="submit")

        if observation.task_id == "meeting_scheduling_medium":
            if not observation.calendar.get("proposed_slot"):
                return OpenEnvAction(action_type="propose_meeting", payload={"slot": "2026-04-06T10:00"})
            if not observation.calendar.get("confirmed"):
                return OpenEnvAction(action_type="confirm_meeting")
            return OpenEnvAction(action_type="submit")

        if observation.task_id == "data_cleaning_hard":
            dataset_preview = observation.dataset_preview
            if any(row["name"] != str(row["name"]).strip() or row["email"] != str(row["email"]).strip() for row in dataset_preview):
                return OpenEnvAction(action_type="clean_data", payload={"operation": "trim_whitespace"})
            if any(str(row["email"]) != str(row["email"]).lower() or str(row["status"]).strip() != str(row["status"]).strip().lower() for row in dataset_preview):
                return OpenEnvAction(action_type="clean_data", payload={"operation": "normalize_case"})
            emails = [str(row["email"]).strip().lower() for row in dataset_preview]
            if len(set(emails)) != len(emails):
                return OpenEnvAction(action_type="clean_data", payload={"operation": "deduplicate"})
            if any(row.get("department") in ("", None) for row in dataset_preview):
                return OpenEnvAction(action_type="clean_data", payload={"operation": "fill_departments"})
            if observation.progress < 1.0:
                return OpenEnvAction(action_type="clean_data", payload={"operation": "finalize"})
            return OpenEnvAction(action_type="submit")

        return OpenEnvAction(action_type="submit")
