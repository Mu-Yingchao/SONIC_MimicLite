"""Playback helpers for interactive viewers (pause / reset / speed / ckpt)."""

from __future__ import annotations

import time
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from tensordict import TensorDictBase


def resolve_mjlab_viewer(env: Any):
    try:
        from active_adaptation.envs.backends.mjlab.viewer import MjLabViewer
    except ImportError:
        return None
    base = getattr(env, "base_env", env)
    sim = getattr(base, "sim", None)
    viewer = getattr(sim, "viewer", None)
    return viewer if isinstance(viewer, MjLabViewer) else None


def play_pre_step(
    env: Any,
    carry: "TensorDictBase",
    *,
    on_checkpoint_load: Callable[[str], None] | None = None,
) -> "TensorDictBase":
    """Handle checkpoint / reset / pause before a play-loop environment step."""
    viewer = resolve_mjlab_viewer(env)
    if viewer is None:
        return carry

    ckpt_path = viewer.consume_checkpoint_request()
    if ckpt_path is not None and on_checkpoint_load is not None:
        on_checkpoint_load(ckpt_path)
        viewer.notify_checkpoint_loaded(ckpt_path)
        carry = env.reset()
        viewer.on_env_reset()

    if viewer.consume_reset_request():
        carry = env.reset()
        viewer.on_env_reset()
    viewer.idle_while_paused()
    return carry


def play_post_step(env: Any, *, step_dt: float, timer: Any | None = None) -> None:
    """Book-keeping + rate control after a play-loop environment step."""
    viewer = resolve_mjlab_viewer(env)
    speed = 1.0
    if viewer is not None:
        viewer.note_step_done()
        speed = viewer.speed_scale
    if timer is not None and abs(speed - 1.0) < 1e-6:
        timer.sleep()
    else:
        time.sleep(max(float(step_dt) / max(speed, 1e-6), 0.0))
