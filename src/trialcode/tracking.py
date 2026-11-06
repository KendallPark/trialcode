"""Weights & Biases helpers shared by all TrialCode experiments."""
import os

import wandb
from dotenv import load_dotenv

load_dotenv()  # reads .env from the working directory tree; never overrides real env vars

DEFAULT_PROJECT = "trialcode"


def init_run(name=None, config=None, tags=None, job_type="train", **kwargs):
    """Start a W&B run. Entity/project/host come from WANDB_ENTITY / WANDB_PROJECT / WANDB_BASE_URL."""
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT),
        name=name,
        config=config or {},
        tags=tags or [],
        job_type=job_type,
        **kwargs,
    )
