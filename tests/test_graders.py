from __future__ import annotations

import unittest

from openenv.env import OpenEnvWorkplace, TASKS
from openenv.models import OpenEnvAction


def _solve_task(env: OpenEnvWorkplace, task_name: str) -> None:
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
        return

    if task_name == "meeting_scheduling":
        env.step(OpenEnvAction(action_type="propose_meeting", payload={"slot": "2026-04-06T10:00"}))
        env.step(OpenEnvAction(action_type="confirm_meeting"))
        return

    if task_name == "data_cleaning":
        env.step(OpenEnvAction(action_type="clean_data", payload={"operation": "trim_whitespace"}))
        env.step(OpenEnvAction(action_type="clean_data", payload={"operation": "normalize_case"}))
        env.step(OpenEnvAction(action_type="clean_data", payload={"operation": "deduplicate"}))
        env.step(OpenEnvAction(action_type="clean_data", payload={"operation": "fill_departments"}))
        env.step(OpenEnvAction(action_type="clean_data", payload={"operation": "finalize"}))
        return

    raise AssertionError(f"Unknown task: {task_name}")


class GraderComplianceTest(unittest.TestCase):
    def test_all_tasks_registered(self) -> None:
        self.assertEqual(set(TASKS.keys()), {"email_triage", "meeting_scheduling", "data_cleaning"})

    def test_all_task_scores_are_strictly_between_zero_and_one(self) -> None:
        for task_name in TASKS:
            env = OpenEnvWorkplace(task_name=task_name)
            _solve_task(env, task_name)
            score, _message = env.evaluate()
            self.assertIsInstance(score, float)
            self.assertGreater(score, 0.0)
            self.assertLess(score, 1.0)


if __name__ == "__main__":
    unittest.main()
