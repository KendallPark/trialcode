"""Weights & Biases helpers shared by all TrialCode experiments."""
import os

import wandb

DEFAULT_PROJECT = "trialcode"


def init_run(name=None, config=None, tags=None, job_type="train", **kwargs):
    """Start a W&B run. Entity/project can be overridden via WANDB_ENTITY / WANDB_PROJECT."""
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT),
        name=name,
        config=config or {},
        tags=tags or [],
        job_type=job_type,
        **kwargs,
    )
