# MIT License
#
# Copyright (c) 2026 Nikhil Sunder
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Persistence of trained cells, so each algorithm notebook hands its policies to the comparison."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

import torch

from ..exceptions import ArtifactError
from .policies import OBS_DIM, DeterministicPolicy, TabularQPolicy, TrainedCell, make_ddpg_actor, make_sac_actor

__all__ = ["load_cell", "load_cells", "save_cell", "save_cells"]

_FORMAT_VERSION = 1


def save_cell(cell: TrainedCell, path: str | os.PathLike[str], *, sac_hidden: tuple[int, ...] = (64, 64)) -> Path:
    """Write a trained cell to one file readable with ``weights_only=True``; returns the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    module = cell.policy if isinstance(cell.policy, TabularQPolicy) else cell.policy.actor
    payload: dict[str, Any] = {
        "kind": "cell", "version": _FORMAT_VERSION, "algo": cell.algo, "env_name": cell.env_name,
        "obs_spec": cell.obs_spec, "policy_variant": cell.policy_variant, "critic_nodes": cell.critic_nodes,
        "actor_nodes": cell.actor_nodes, "steady_state_reward": float(cell.steady_state_reward),
        "action_high": float(cell.action_high), "training_rewards": [float(r) for r in cell.training_rewards],
        "sac_hidden": list(sac_hidden), "state": {k: v.detach().cpu() for k, v in module.state_dict().items()},
    }
    torch.save(payload, target)
    return target


def load_cell(path: str | os.PathLike[str]) -> TrainedCell:
    """Read a cell written by :func:`save_cell` and rebuild its policy module.

    Raises:
        ArtifactError: If the file is missing, damaged, or not a trained cell.
    """
    source = Path(path)
    if not source.is_file():
        raise ArtifactError(f"{source} not found; run the notebook that trains this cell first")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch raises several unrelated types for a damaged file
        raise ArtifactError(f"{source} could not be read: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("kind") != "cell" or payload.get("version") != _FORMAT_VERSION:
        raise ArtifactError(f"{source} is not a trained cell of format version {_FORMAT_VERSION}")

    algo, state = payload["algo"], payload["state"]
    if algo == "Q-learning":
        policy: torch.nn.Module = TabularQPolicy(
            payload["action_high"], n_bins=tuple(int(b) for b in state["bins"].tolist()),
            action_grid=state["action_grid"].tolist(), obs_low=tuple(state["obs_low"].tolist()),
            obs_high=tuple(state["obs_high"].tolist()))
        policy.load_state_dict(state)
    else:
        obs_dim = OBS_DIM[payload["obs_spec"]]
        actor = (make_ddpg_actor(obs_dim, payload["policy_variant"], int(payload["actor_nodes"])) if algo == "DDPG"
                 else make_sac_actor(obs_dim, payload["sac_hidden"]))
        actor.load_state_dict(state)
        policy = DeterministicPolicy(actor)
    return TrainedCell(algo, payload["env_name"], payload["obs_spec"], payload["policy_variant"], payload["critic_nodes"],
                       payload["actor_nodes"], payload["steady_state_reward"], policy.eval(), payload["action_high"],
                       list(payload["training_rewards"]))


def save_cells(cells: Iterable[TrainedCell], directory: str | os.PathLike[str], *,
               sac_hidden: tuple[int, ...] = (64, 64)) -> list[Path]:
    """Write every cell to ``<directory>/<cell.key>.pt``; returns the paths."""
    return [save_cell(cell, Path(directory) / f"{cell.key}.pt", sac_hidden=sac_hidden) for cell in cells]


def load_cells(directory: str | os.PathLike[str], *, algo: Optional[str] = None) -> list[TrainedCell]:
    """Load every cell saved in ``directory``, optionally only those of one algorithm.

    Raises:
        ArtifactError: If the directory holds no matching cell.
    """
    source = Path(directory)
    cells = [load_cell(path) for path in sorted(source.glob("*.pt"))] if source.is_dir() else []
    if algo is not None:
        cells = [cell for cell in cells if cell.algo == algo]
    if not cells:
        raise ArtifactError(f"no trained {algo or 'RL'} cells in {source}; run the algorithm notebooks first")
    return cells
