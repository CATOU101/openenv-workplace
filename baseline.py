from __future__ import annotations

import json
import os
from typing import Any

from openenv import OpenEnvAction, OpenEnvWorkplace

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - dependency availability varies by runtime
    OpenAI = None  # type: ignore[assignment]


SYSTEM_PROMPT = """
You are controlling a deterministic workplace simulation environment.
Return exactly one JSON object with keys:
- action_type: string
- target_id: string or null
- payload: object

Choose only from the available_actions shown in the observation.
Focus on efficient task completion.
""".strip()


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


def build_prompt(observation: dict[str, Any]) -> str:
    return json.dumps(observation, indent=2, sort_keys=True)


def fallback_policy(task_id: str, observation: dict[str, Any]) -> OpenEnvAction:
    if task_id == "email_triage_easy":
        inbox = observation["inbox"]
        unread = [email for email in inbox if email["status"] in {"unread", "read"}]
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

    if task_id == "meeting_scheduling_medium":
        calendar = observation["calendar"]
        if not calendar.get("proposed_slot"):
            return OpenEnvAction(action_type="propose_meeting", payload={"slot": "2026-04-06T10:00"})
        if not calendar.get("confirmed"):
            return OpenEnvAction(action_type="confirm_meeting")
        return OpenEnvAction(action_type="submit")

    if task_id == "data_cleaning_hard":
        dataset_preview = observation["dataset_preview"]
        if any(row["name"] != str(row["name"]).strip() or row["email"] != str(row["email"]).strip() for row in dataset_preview):
            return OpenEnvAction(action_type="clean_data", payload={"operation": "trim_whitespace"})
        if any(str(row["email"]) != str(row["email"]).lower() or str(row["status"]).strip() != str(row["status"]).strip().lower() for row in dataset_preview):
            return OpenEnvAction(action_type="clean_data", payload={"operation": "normalize_case"})
        emails = [str(row["email"]).strip().lower() for row in dataset_preview]
        if len(set(emails)) != len(emails):
            return OpenEnvAction(action_type="clean_data", payload={"operation": "deduplicate"})
        if any(row.get("department") in ("", None) for row in dataset_preview):
            return OpenEnvAction(action_type="clean_data", payload={"operation": "fill_departments"})
        if observation["progress"] < 1.0:
            return OpenEnvAction(action_type="clean_data", payload={"operation": "finalize"})
        return OpenEnvAction(action_type="submit")

    raise ValueError(f"Unsupported task_id: {task_id}")


def llm_policy(client: Any, model: str, observation: dict[str, Any]) -> OpenEnvAction:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "input_text", "text": build_prompt(observation)}]},
        ],
    )
    content = _parse_model_text(response)
    data = json.loads(content)
    return OpenEnvAction.model_validate(data)


def run_task(env: OpenEnvWorkplace, task_id: str, client: Any, model: str) -> dict[str, Any]:
    observation = env.reset(task_id=task_id)
    done = False
    last_raw_score = 0.0
    final_info: dict[str, Any] = {}

    while not done:
        observation_payload = observation.model_dump(mode="json")
        try:
            action = llm_policy(client, model, observation_payload) if client else fallback_policy(task_id, observation_payload)
        except Exception:
            action = fallback_policy(task_id, observation_payload)

        observation, reward, done, info = env.step(action)
        last_raw_score = float(info["raw_score"])
        final_info = info | {"last_reward": reward.model_dump(mode="json")}

    return {
        "task_id": task_id,
        "task_name": final_info["task_name"],
        "score": round(last_raw_score, 4),
        "steps_taken": final_info["steps_taken"],
        "invalid_steps": final_info["invalid_steps"],
        "repeated_actions": final_info["repeated_actions"],
        "episode_id": final_info["episode_id"],
    }


def print_leaderboard(results: list[dict[str, Any]]) -> None:
    print("Hackathon Leaderboard")
    print("====================")
    print(f"{'Task':<24} {'Score':>8} {'Steps':>8} {'Invalid':>8} {'Repeat':>8}")
    print("-" * 64)
    for result in sorted(results, key=lambda item: (-item["score"], item["steps_taken"], item["task_name"])):
        print(
            f"{result['task_name']:<24} "
            f"{result['score']:>8.4f} "
            f"{result['steps_taken']:>8} "
            f"{result['invalid_steps']:>8} "
            f"{result['repeated_actions']:>8}"
        )


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key) if api_key and OpenAI is not None else None

    env = OpenEnvWorkplace()
    results: list[dict[str, Any]] = []

    for task in env.task_specs:
        result = run_task(env, task.task_id, client, model)
        results.append(result)

    print("Per-task baseline scores")
    print("========================")
    total = 0.0
    for result in results:
        total += result["score"]
        print(
            f"{result['task_name']}: "
            f"score={result['score']:.4f}, "
            f"steps={result['steps_taken']}, "
            f"invalid={result['invalid_steps']}, "
            f"repeated={result['repeated_actions']}, "
            f"episode_id={result['episode_id']}"
        )
    average = total / len(results)
    print(f"Average: {average:.4f}")
    print()
    print_leaderboard(results)


if __name__ == "__main__":
    main()
