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
    api_base_url = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("API_KEY")

    missing = [
        name
        for name, value in (
            ("API_BASE_URL", api_base_url),
            ("API_KEY", api_key),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
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


def run_task(client: OpenAI, model_name: str, task_name: str) -> tuple[bool, int, float, list[float]]:
    env = OpenEnvWorkplace(task_name=task_name)
    observation = env.reset(task_name=task_name)
    print(f"[START] task={observation.task_id} env=openenv-workplace model={model_name}")

    rewards: list[float] = []
    done = False
    success = False
    step_count = 0
    last_error: str | None = None

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
        last_error = error
        print(
            f"[STEP] step={step_count} action={action_text} "
            f"reward={reward_value:.2f} done={'true' if done else 'false'} "
            f"error={error if error is not None else 'null'}"
        )

    if done and rewards:
        success = last_error is None

    score = sum(rewards) / len(rewards) if rewards else 0.0
    score = max(0.01, min(0.99, score))
    reward_list = ",".join(f"{value:.2f}" for value in rewards)
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step_count} score={score:.2f} rewards=<{reward_list}>"
    )
    return success, step_count, score, rewards


def main() -> None:
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    try:
        client = build_client()
        for task_name in ("email_triage", "meeting_scheduling", "data_cleaning"):
            run_task(client, model_name, task_name)
    except Exception as exc:
        print("[START] task= env=openenv-workplace model=" + model_name)
        print(f"[STEP] step=0 action=null reward=0.00 done=true error={str(exc)}")
        print("[END] success=false steps=0 score=0.01 rewards=<>")
    finally:
        return None


if __name__ == "__main__":
    main()
