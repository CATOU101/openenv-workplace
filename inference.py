from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from openenv.env import OpenEnvWorkplace
from openenv.models import OpenEnvAction


SYSTEM_PROMPT = """You are an AI agent solving workplace tasks.
You must output ONLY valid JSON action.
No text, no explanation."""


def parse_model_text(response: Any) -> str:
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


def build_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("HF_TOKEN", ""),
        base_url=os.getenv("API_BASE_URL"),
    )


def build_action(client: OpenAI, model_name: str, observation: Any) -> OpenEnvAction:
    response = client.responses.create(
        model=model_name,
        temperature=0,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(observation.model_dump(mode="json"), indent=2, sort_keys=True),
                    }
                ],
            },
        ],
    )
    payload = json.loads(parse_model_text(response))
    return OpenEnvAction.model_validate(payload)


def main() -> None:
    model_name = os.getenv("MODEL_NAME", "")
    client = build_client()
    env = OpenEnvWorkplace()

    observation = env.reset()
    print(f"[START] task={observation.task_id} env=openenv-workplace model={model_name}")

    rewards: list[float] = []
    done = False
    success = False
    step_count = 0

    while not done and step_count < 20:
        error: str | None = None
        try:
            action = build_action(client, model_name, observation)
            action_text = json.dumps(action.model_dump(mode="json"), separators=(",", ":"))
            observation, reward, done, _info = env.step(action)
            reward_value = float(reward.score)
        except Exception as exc:
            error = str(exc)
            action_text = "null"
            reward_value = 0.0
            done = True

        rewards.append(reward_value)
        step_count += 1
        print(
            f"[STEP] step={step_count} action={action_text} "
            f"reward={reward_value:.2f} done={'true' if done else 'false'} "
            f"error={error if error is not None else 'null'}"
        )

    if done and rewards:
        success = error is None

    score = sum(rewards) / len(rewards) if rewards else 0.0
    score = max(0.0, min(1.0, score))
    reward_list = ",".join(f"{value:.2f}" for value in rewards)
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step_count} score={score:.2f} rewards=<{reward_list}>"
    )


if __name__ == "__main__":
    main()
