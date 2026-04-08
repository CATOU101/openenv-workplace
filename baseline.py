from __future__ import annotations

import os
from typing import Any

from openenv import OpenEnvAction, OpenEnvWorkplace
from llm_agent import LLMAgent


def fallback_policy(task_id: str, observation: dict[str, Any]) -> OpenEnvAction:
    if task_id == "email_triage":
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

    if task_id == "meeting_scheduling":
        calendar = observation["calendar"]
        if not calendar.get("proposed_slot"):
            return OpenEnvAction(action_type="propose_meeting", payload={"slot": "2026-04-06T10:00"})
        if not calendar.get("confirmed"):
            return OpenEnvAction(action_type="confirm_meeting")
        return OpenEnvAction(action_type="submit")

    if task_id == "data_cleaning":
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


class HeuristicAgent:
    def act(self, observation: Any) -> OpenEnvAction:
        observation_payload = observation.model_dump(mode="json") if hasattr(observation, "model_dump") else observation
        return fallback_policy(observation_payload["task_id"], observation_payload)


def build_agent() -> Any:
    agent_type = os.getenv("AGENT_TYPE", "heuristic").strip().lower()
    if agent_type == "llm":
        return LLMAgent()
    return HeuristicAgent()


def run_task(env: OpenEnvWorkplace, task_id: str, agent: Any) -> dict[str, Any]:
    observation = env.reset(task_name=task_id)
    done = False
    last_raw_score = 0.0
    final_info: dict[str, Any] = {}

    while not done:
        try:
            action = agent.act(observation)
        except Exception:
            action = fallback_policy(task_id, observation.model_dump(mode="json"))

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
    env = OpenEnvWorkplace()
    agent = build_agent()
    results: list[dict[str, Any]] = []

    for task in env.task_specs:
        result = run_task(env, task.task_id, agent)
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
