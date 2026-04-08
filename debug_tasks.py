from __future__ import annotations

from openenv.env import OpenEnvWorkplace, TASKS
from openenv.models import OpenEnvAction


def solve_task(env: OpenEnvWorkplace, task_name: str) -> int:
    steps = 0
    env.reset(task_name=task_name)

    if task_name == "email_triage":
        env.step(OpenEnvAction(action_type="archive_email", target_id="email_1"))
        env.step(
            OpenEnvAction(
                action_type="reply_email",
                target_id="email_2",
                payload={"reply_text": "Legal will review by Friday and confirm next steps."},
            )
        )
        env.step(OpenEnvAction(action_type="archive_email", target_id="email_3"))
        return 3

    if task_name == "meeting_scheduling":
        env.step(OpenEnvAction(action_type="propose_meeting", payload={"slot": "2026-04-06T10:00"}))
        env.step(OpenEnvAction(action_type="confirm_meeting"))
        return 2

    if task_name == "data_cleaning":
        env.step(OpenEnvAction(action_type="clean_data", payload={"operation": "trim_whitespace"}))
        env.step(OpenEnvAction(action_type="clean_data", payload={"operation": "normalize_case"}))
        env.step(OpenEnvAction(action_type="clean_data", payload={"operation": "deduplicate"}))
        env.step(OpenEnvAction(action_type="clean_data", payload={"operation": "fill_departments"}))
        env.step(OpenEnvAction(action_type="clean_data", payload={"operation": "finalize"}))
        return 5

    raise ValueError(f"Unknown task: {task_name}")


def main() -> None:
    for task_name in TASKS:
        env = OpenEnvWorkplace(task_name=task_name)
        steps = solve_task(env, task_name)
        raw_score, _message = env.evaluate()
        clamped_score = max(0.01, min(raw_score, 0.99))
        print(f"task name: {task_name}")
        print(f"raw score: {raw_score}")
        print(f"clamped score: {clamped_score}")
        print(f"number of steps: {steps}")


if __name__ == "__main__":
    main()
