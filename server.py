from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from openenv.env import OpenEnvWorkplace
from openenv.models import OpenEnvAction


class ResetRequest(BaseModel):
    task_id: str | None = None


env = OpenEnvWorkplace()
app = FastAPI(title="OpenEnv Workplace")


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "name": "openenv-workplace"}


@app.post("/reset")
def reset(request: ResetRequest | None = None) -> dict[str, Any]:
    observation = env.reset(task_id=request.task_id if request else None)
    return observation.model_dump(mode="json")


@app.post("/step")
def step(action: OpenEnvAction) -> dict[str, Any]:
    observation, reward, done, info = env.step(action)
    return {
        "observation": observation.model_dump(mode="json"),
        "reward": reward.model_dump(mode="json"),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict[str, Any]:
    return env.state().model_dump(mode="json")
