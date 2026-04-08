---
sdk: docker
tags:
  - openenv
  - hackathon
  - reinforcement-learning
  - ai-agents
---

# OpenEnv Workplace

OpenEnv Workplace is a hackathon-ready, deterministic OpenEnv environment built to showcase real-world AI agent training and evaluation. It models workplace operations instead of games, exposes typed observations/actions/rewards, logs full episodes, and produces reproducible leaderboard-style baseline metrics.

## Problem Statement

Most agent benchmarks still over-index on toy tasks, brittle puzzles, or narrow API exercises. This project demonstrates a more practical alternative: a compact but fully programmatic environment where an agent must complete routine knowledge-work tasks with measurable progress, deterministic grading, and reproducible scores.

## Why Real-World Tasks Matter

Real-world agent systems are judged by whether they can triage communication, coordinate schedules, and clean operational data efficiently and safely. These tasks capture the kinds of decisions deployed assistants and workflow agents actually make:

- Prioritize and resolve inbox items.
- Coordinate multiple participants with constraints.
- Transform noisy business data into standardized outputs.

## Environment Description

The environment models common operational workflows instead of toy mechanics:

- Email triage: resolve a small inbox by archiving non-actionable mail and replying to a vendor message.
- Meeting scheduling: identify the earliest shared timeslot across participants and confirm the event.
- Data cleaning: normalize a CRM export, fill missing values from known mappings, deduplicate records, and finalize a clean dataset.

All tasks are deterministic and scored by code-based graders that return values strictly within `(0.0, 1.0)`. Final grader outputs are clamped into the range `[0.01, 0.99]` for evaluator compatibility.

## Architecture Diagram

```text
+-----------------------+
| baseline.py           |
| OpenAI or fallback    |
+-----------+-----------+
            |
            v
+-----------------------+
| OpenEnvWorkplace      |
| reset / step / state  |
+-----------+-----------+
            |
            +-------------------+
            |                   |
            v                   v
+-------------------+   +-------------------+
| tasks.py          |   | rewards.py        |
| deterministic     |   | dense shaping     |
| task state        |   | penalties/bonus   |
+---------+---------+   +---------+---------+
          |                         |
          +------------+------------+
                       |
                       v
               +---------------+
               | graders.py    |
               | final scoring |
               +---------------+
```

## Project Structure

```text
openenv-workplace/
  openenv/
    __init__.py
    env.py
    models.py
    tasks.py
    graders.py
    rewards.py
  baseline.py
  openenv.yaml
  Dockerfile
  requirements.txt
  README.md
```

## OpenEnv Interface

The environment entrypoint is `openenv.env:OpenEnvWorkplace`.

Implemented methods:

- `reset() -> OpenEnvObservation`
- `step(action: OpenEnvAction) -> tuple[OpenEnvObservation, OpenEnvReward, bool, dict]`
- `state() -> OpenEnvObservation`

Typed models:

- Observation: `OpenEnvObservation`
- Action: `OpenEnvAction`
- Reward: `OpenEnvReward`

## Task Definitions

### 1. Email Triage (Easy)

Objective: process three inbox items using realistic actions such as reading, archiving, and replying.

Deterministic grader:

- Archives the FYI email correctly
- Replies to the vendor email with the required commitment
- Archives the newsletter

### 2. Meeting Scheduling (Medium)

Objective: schedule a 30-minute sync for Alice, Bob, and Carla at the earliest shared availability.

Deterministic grader:

- Rewards proposing the correct shared slot
- Rewards confirming the meeting
- Penalizes later but still valid-looking proposals

### 3. Data Cleaning (Hard)

Objective: clean a noisy CRM export by progressively applying transformation operations.

Deterministic grader:

- Validates canonical rows
- Checks row count after deduplication
- Verifies normalized names, emails, departments, and statuses

## Observation, Action, and Reward Spaces

### Observation

`OpenEnvObservation` includes:

- `task_id`, `task_name`, `difficulty`, `objective`, `episode_id`
- `available_actions`
- `inbox`, `calendar`, `dataset_preview`
- `progress`, `steps_taken`, `max_steps`
- `action_history`
- `last_action_valid`, `last_feedback`

### Action

`OpenEnvAction` includes:

- `action_type`
- `target_id`
- `payload`

Supported action types:

- `read_email`
- `archive_email`
- `reply_email`
- `propose_meeting`
- `confirm_meeting`
- `clean_data`
- `submit`

### Reward

`OpenEnvReward` includes:

- `score`
- `delta`
- `invalid_action_penalty`
- `repeated_action_penalty`
- `efficiency_penalty`
- `completion_bonus`
- `message`

## Reward Design

The reward function provides dense, shaped feedback:

- Positive delta for incremental task progress
- Penalty for invalid actions
- Penalty for repeated actions
- Small per-step efficiency penalty
- Completion bonus for fully solving a task
- Bounded final score in `[0.0, 1.0]`

This encourages correct intermediate actions while discouraging redundant behavior, invalid tool use, and slow completion.

## Hackathon Scoring Features

The environment includes evaluation-friendly instrumentation:

- Deterministic `episode_id` for each task run
- Per-step episode logging in `info["episode_log"]`
- Action validation and invalid-step accounting
- Repeated-action tracking
- Step counter and max-step budget
- Leaderboard-style baseline output for quick judging

`info` includes:

- `raw_score`
- `steps_taken`
- `valid_steps`
- `invalid_steps`
- `repeated_actions`
- `action_history`
- `episode_log`
- `done`

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python baseline.py
```

If `OPENAI_API_KEY` is set, `baseline.py` will try to use the OpenAI Responses API. If not, it falls back to a deterministic scripted baseline so the project stays runnable offline.

Optional environment variables:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` default: `gpt-4.1-mini`

## Docker Usage

Build and run:

```bash
docker build -t openenv-hackathon .
docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY openenv-hackathon
```

This configuration is compatible with HuggingFace Spaces using Docker SDK.

## HuggingFace Deployment

This repository is compatible with containerized HuggingFace Spaces:

- `README.md` includes the required `sdk: docker` header
- The Docker image launches the reproducible baseline runner by default
- `openenv.yaml` provides environment metadata for discoverability and integration

## Baseline Scores

The included fallback policy is deterministic and should produce reproducible metrics. Because final task graders clamp scores into the strict open interval `(0, 1)`, perfect task completion yields `0.9900` instead of `1.0000`.

| Task | Score |
|---|---:|
| Email Triage | 0.9900 |
| Meeting Scheduling | 0.9900 |
| Data Cleaning | 0.9900 |
| Average | 0.9900 |

When using an OpenAI model, results remain reproducible so long as the model output is stable and temperature is set to `0`.

## Future Improvements

- Add additional enterprise workflows such as CRM note-taking or ticket routing.
- Support multi-turn partial observability for longer-horizon agents.
- Add batched evaluation utilities for RL training loops.
- Export leaderboard logs to JSONL for experiment tracking.
