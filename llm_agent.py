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
        self.base_url = base_url or os.getenv("LLM_BASE_URL")

        if OpenAI is None:
            raise RuntimeError("openai package is required to use LLMAgent.")
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is required to use LLMAgent.")

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = OpenAI(**client_kwargs)

    def act(self, observation: OpenEnvObservation) -> OpenEnvAction:
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
