from __future__ import annotations

from openenv.models import OpenEnvReward


def build_reward(
    previous_score: float,
    current_score: float,
    *,
    is_valid: bool,
    is_repeated: bool,
    steps_taken: int,
    max_steps: int,
    done: bool,
    feedback: str,
) -> OpenEnvReward:
    delta = round(current_score - previous_score, 4)
    invalid_penalty = 0.0 if is_valid else 0.1
    repeated_action_penalty = 0.03 if is_repeated else 0.0
    efficiency_penalty = round(min(steps_taken / max_steps, 1.0) * 0.02, 4)
    completion_bonus = 0.1 if done and current_score >= 0.999 else 0.0

    shaped_score = current_score + delta + completion_bonus - invalid_penalty - repeated_action_penalty - efficiency_penalty
    shaped_score = round(max(0.0, min(1.0, shaped_score)), 4)

    if not is_valid:
        message = f"Invalid action. {feedback}".strip()
    elif is_repeated:
        message = f"Repeated action penalty applied. {feedback}".strip()
    elif done and current_score >= 0.999:
        message = "Task completed successfully."
    else:
        message = feedback or "Progress updated."

    return OpenEnvReward(
        score=shaped_score,
        delta=delta,
        invalid_action_penalty=invalid_penalty,
        repeated_action_penalty=repeated_action_penalty,
        efficiency_penalty=efficiency_penalty,
        completion_bonus=completion_bonus,
        message=message,
    )
