from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import viser
from mjlab.sim import Simulation
from mjlab.viewer.viser import ViserMujocoScene
from mjlab.viewer.viser.term_plotter import ViserTermPlotter

from active_adaptation.envs.backends.mjlab.reward_bar import DarkRewardBarPanel
from active_adaptation.envs.env_base import _EnvBase
from active_adaptation.utils.profiling import ScopedTimer

SPEED_MULTIPLIERS = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0)
_UI_TEXT = "#1a1a1a"


def _rgba_to_rgb255(color: tuple[float, ...] | list[float]) -> tuple[int, int, int]:
    r, g, b = float(color[0]), float(color[1]), float(color[2])
    if max(r, g, b) <= 1.0:
        return (int(r * 255), int(g * 255), int(b * 255))
    return (int(r), int(g), int(b))


def _format_speed(multiplier: float) -> str:
    if multiplier >= 1.0:
        return f"{multiplier:g}x"
    return f"1/{int(round(1.0 / multiplier))}x"


def _checkpoint_sort_key(name: str) -> float:
    if name.endswith("_final.pt") or name == "checkpoint_final.pt":
        return float("inf")
    match = re.search(r"checkpoint_(\d+)", name)
    return float(match.group(1)) if match else -1.0


def discover_checkpoint_files(current_path: str | Path) -> list[Path]:
    path = Path(current_path).expanduser().resolve()
    if not path.is_file():
        return []
    files = [
        p
        for p in path.parent.glob("checkpoint_*.pt")
        if p.is_file()
    ]
    files.sort(key=lambda p: _checkpoint_sort_key(p.name))
    return files


@dataclass
class CheckpointBrowser:
    """Local checkpoint list + load callback for the Viser Checkpoints tab."""

    current_path: Path
    on_load: Callable[[str], None]
    available: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.current_path = Path(self.current_path).expanduser().resolve()
        self.refresh()

    def refresh(self) -> list[str]:
        self.available = discover_checkpoint_files(self.current_path)
        if self.current_path.is_file() and self.current_path not in self.available:
            self.available.append(self.current_path)
            self.available.sort(key=lambda p: _checkpoint_sort_key(p.name))
        return [p.name for p in self.available]

    def labels(self) -> list[str]:
        labels = [p.name for p in self.available]
        return labels or [self.current_path.name]

    def path_for_label(self, label: str) -> Path | None:
        for path in self.available:
            if path.name == label:
                return path
        if self.current_path.name == label:
            return self.current_path
        return None

    def set_current(self, path: Path) -> None:
        self.current_path = Path(path).expanduser().resolve()


class MjLabViewer:
    """
    Different from `mjlab.viewer.viser.viewer.ViserPlayViewer`, this
    viewer is not responsible for stepping the environment and is updated
    synchronously from the environment step loop.
    """

    def __init__(self, env: _EnvBase, sim: Simulation):
        self.env = env
        self.sim = sim

        self._server = viser.ViserServer(label="mjlab")
        self._is_setup = False

        self._cameras: dict[str, viser.CameraFrustumHandle] = {}
        self._line_handle = None
        self._point_handle = None
        self._debug_line_pts: list[np.ndarray] = []
        self._debug_line_cols: list[np.ndarray] = []
        self._debug_point_pts: list[np.ndarray] = []
        self._debug_point_cols: list[np.ndarray] = []
        self._debug_point_size: float = 0.02

        self._paused = False
        self._single_step = False
        self._reset_requested = False
        self._speed_index = SPEED_MULTIPLIERS.index(1.0)
        self._step_count = 0
        self._prev_env_idx = 0

        self._status_html = None
        self._pause_button = None
        self._reward_bar_panel: DarkRewardBarPanel | None = None
        self._reward_plotter: ViserTermPlotter | None = None
        self._checkpoint_browser: CheckpointBrowser | None = None
        self._ckpt_dropdown = None
        self._ckpt_status_html = None
        self._pending_checkpoint_path: str | None = None
        self._tabs = None
        self._checkpoints_tab_added = False
        self._motion_clip_hint = None
        self._motion_clip_name_html = None

    @property
    def speed_scale(self) -> float:
        return float(SPEED_MULTIPLIERS[self._speed_index])

    @property
    def is_paused(self) -> bool:
        return self._paused

    def setup(self):
        if self._is_setup:
            return

        self._scene = ViserMujocoScene(
            self._server,
            self.sim.mj_model,
            self.env.num_envs,
        )
        self._scene.debug_visualization_enabled = True
        self._scene.camera_tracking_enabled = False
        self._scene.show_all_envs = True
        self._scene.env_idx = 0
        self._prev_env_idx = 0

        viewer_cfg = getattr(self.env.sim, "_viewer_cfg", None)
        camera_kwargs = {}
        if viewer_cfg is not None:
            camera_kwargs = {
                "camera_distance": float(getattr(viewer_cfg, "distance", 3.0)),
                "camera_azimuth": float(getattr(viewer_cfg, "azimuth", 45.0)),
                "camera_elevation": float(getattr(viewer_cfg, "elevation", -30.0)),
            }

        # Default medium panel (~20em) truncates long motion names.
        self._server.gui.configure_theme(control_width="large")

        tabs = self._server.gui.add_tab_group()
        self._tabs = tabs
        with tabs.add_tab("Controls", icon=viser.Icon.SETTINGS):
            with self._server.gui.add_folder("Info"):
                self._status_html = self._server.gui.add_html("")

            with self._server.gui.add_folder("Simulation"):
                self._pause_button = self._server.gui.add_button(
                    "Pause",
                    icon=viser.Icon.PLAYER_PAUSE,
                )

                @self._pause_button.on_click
                def _(_) -> None:
                    self._paused = not self._paused
                    self._single_step = False
                    self._sync_pause_button()
                    self._update_status_display()

                step_button = self._server.gui.add_button(
                    "Step",
                    icon=viser.Icon.PLAYER_TRACK_NEXT,
                )

                @step_button.on_click
                def _(_) -> None:
                    self._single_step = True
                    self._paused = True
                    self._sync_pause_button()
                    self._update_status_display()

                reset_button = self._server.gui.add_button(
                    "Reset Environment",
                    icon=viser.Icon.REFRESH,
                )

                @reset_button.on_click
                def _(_) -> None:
                    self._reset_requested = True

                speed_buttons = self._server.gui.add_button_group(
                    "Speed",
                    options=["Slower", "1x", "Faster"],
                )

                @speed_buttons.on_click
                def _(event) -> None:
                    if event.target.value == "Slower":
                        self._speed_index = max(0, self._speed_index - 1)
                    elif event.target.value == "1x":
                        self._speed_index = SPEED_MULTIPLIERS.index(1.0)
                    else:
                        self._speed_index = min(
                            len(SPEED_MULTIPLIERS) - 1, self._speed_index + 1
                        )
                    self._update_status_display()

            if hasattr(getattr(self.env, "command_manager", None), "viz"):
                self._create_mimic_gui()

            with self._server.gui.add_folder("Scene"):
                self._scene.create_scene_gui(**camera_kwargs)

        with tabs.add_tab("Visualization", icon=viser.Icon.EYE):
            self._scene.create_overlay_gui()

        self._setup_reward_tabs(tabs)

        with tabs.add_tab("Groups", icon=viser.Icon.LAYERS_INTERSECT):
            self._scene.create_groups_gui()

        if self._checkpoint_browser is not None:
            self._setup_checkpoint_tab(tabs)

        self._update_status_display()
        self._is_setup = True

    def attach_checkpoints(
        self,
        current_path: str | Path,
        on_load: Callable[[str], None],
    ) -> None:
        """Enable Checkpoints tab (call before or after setup)."""
        self._checkpoint_browser = CheckpointBrowser(
            current_path=Path(current_path),
            on_load=on_load,
        )
        if self._is_setup and self._tabs is not None and not self._checkpoints_tab_added:
            self._setup_checkpoint_tab(self._tabs)
            self._update_status_display()

    def _setup_checkpoint_tab(self, tabs) -> None:
        if self._checkpoint_browser is None or self._checkpoints_tab_added:
            return
        browser = self._checkpoint_browser
        labels = browser.labels()
        with tabs.add_tab("Checkpoints", icon=viser.Icon.DATABASE):
            self._ckpt_status_html = self._server.gui.add_html("")
            self._ckpt_dropdown = self._server.gui.add_dropdown(
                "Checkpoint",
                options=labels,
                initial_value=browser.current_path.name
                if browser.current_path.name in labels
                else labels[0],
            )

            @self._ckpt_dropdown.on_update
            def _(_) -> None:
                path = browser.path_for_label(self._ckpt_dropdown.value)
                if path is not None:
                    self._pending_checkpoint_path = str(path)

            buttons = self._server.gui.add_button_group(
                "Actions",
                options=["Load", "Random", "Refresh"],
            )

            @buttons.on_click
            def _(event) -> None:
                if event.target.value == "Refresh":
                    new_labels = browser.refresh() or browser.labels()
                    self._ckpt_dropdown.options = new_labels
                    current = browser.current_path.name
                    if current in new_labels:
                        self._ckpt_dropdown.value = current
                    self._update_checkpoint_status()
                    return
                if event.target.value == "Random":
                    choices = list(browser.available) or [browser.current_path]
                    if len(choices) > 1:
                        others = [p for p in choices if p != browser.current_path]
                        pick = random.choice(others or choices)
                    else:
                        pick = choices[0]
                    self._ckpt_dropdown.value = pick.name
                    self._pending_checkpoint_path = str(pick)
                    return
                # Load selected
                path = browser.path_for_label(self._ckpt_dropdown.value)
                if path is not None:
                    self._pending_checkpoint_path = str(path)

            self._server.gui.add_markdown(
                "Loads sibling `checkpoint_*.pt` from the same directory. "
                "**Random** picks another file; applied on the next play step."
            )
        self._checkpoints_tab_added = True
        self._update_checkpoint_status()

    def consume_checkpoint_request(self) -> str | None:
        path = self._pending_checkpoint_path
        self._pending_checkpoint_path = None
        return path

    def notify_checkpoint_loaded(self, path: str) -> None:
        if self._checkpoint_browser is None:
            return
        loaded = Path(path).expanduser().resolve()
        self._checkpoint_browser.set_current(loaded)
        labels = self._checkpoint_browser.refresh()
        if self._ckpt_dropdown is not None:
            self._ckpt_dropdown.options = labels or [loaded.name]
            if loaded.name in (labels or [loaded.name]):
                self._ckpt_dropdown.value = loaded.name
        self._update_checkpoint_status()
        self._update_status_display()

    def _update_checkpoint_status(self) -> None:
        if self._ckpt_status_html is None or self._checkpoint_browser is None:
            return
        browser = self._checkpoint_browser
        n = len(browser.available)
        self._ckpt_status_html.content = (
            f'<div style="font-size:0.85em;line-height:1.25;padding:0 1em 0.5em 1em;'
            f'color:{_UI_TEXT};">'
            f"<strong>Current:</strong> {browser.current_path.name}<br/>"
            f"<strong>Found:</strong> {n} checkpoint(s) in "
            f"{browser.current_path.parent}"
            f"</div>"
        )

    def _create_mimic_gui(self) -> None:
        command = self.env.command_manager
        viz = getattr(command, "viz", None)
        if viz is None:
            return

        with self._server.gui.add_folder("Mimic"):
            mode_dropdown = self._server.gui.add_dropdown(
                "Reference Viz",
                options=["ghost", "frames", "off"],
                initial_value=getattr(viz, "mode", "ghost"),
                hint="Ghost mesh / body frames / disabled.",
            )

            @mode_dropdown.on_update
            def _(_) -> None:
                command.viz.mode = mode_dropdown.value
                if hasattr(self._scene, "clear_debug_all"):
                    self._scene.clear_debug_all()

            track_cb = self._server.gui.add_checkbox(
                "Camera Tracking",
                initial_value=bool(self._scene.camera_tracking_enabled),
                hint="Keep camera centered on the tracked body.",
            )

            @track_cb.on_update
            def _(_) -> None:
                self._scene.camera_tracking_enabled = bool(track_cb.value)

            # Viewer setup runs during env __init__, before play.py calls eval().
            if hasattr(command, "search_motion_clips"):
                try:
                    self._create_motion_clip_gui(command)
                except Exception as error:
                    print(f"[viewer] failed to create motion clip GUI: {error}")

    def _create_motion_clip_gui(self, command) -> None:
        """Play-time clip picker: search, then apply a pin and reset."""
        placeholder = "(no matches)"
        matches = command.search_motion_clips("")
        initial = matches[0] if matches else placeholder
        self._motion_clip_hint = self._server.gui.add_markdown("")
        self._motion_clip_name_html = self._server.gui.add_html("")
        search_box = self._server.gui.add_text(
            "Clip Search",
            initial_value="",
            hint="Filter by id, folder, or clip name. Apply to pin and reset.",
        )
        dropdown = self._server.gui.add_dropdown(
            "Clip",
            options=matches or [placeholder],
            initial_value=initial,
            hint="Filtered matches. Full name is shown above. Click Apply to pin.",
        )
        actions = self._server.gui.add_button_group(
            "Clip Actions",
            options=["Apply", "Random", "Unpin"],
        )

        def _clip_name_html(selected: str, pinned: str) -> str:
            def _esc(text: str) -> str:
                return (
                    str(text)
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )

            return (
                f'<div style="font-size:0.8em;line-height:1.35;padding:0 0.15em 0.35em 0.15em;'
                f'color:{_UI_TEXT};word-break:break-all;overflow-wrap:anywhere;">'
                f"<strong>Selected:</strong><br/>{_esc(selected)}<br/>"
                f"<strong>Pinned:</strong><br/>{_esc(pinned)}"
                f"</div>"
            )

        def _set_matches(query: str, *, preferred: str | None = None) -> None:
            found = command.search_motion_clips(query)
            options = found or [placeholder]
            dropdown.options = options
            if preferred is not None and preferred in options:
                dropdown.value = preferred
            elif dropdown.value not in options:
                dropdown.value = options[0]
            total = int(getattr(command, "num_motion_clips", 0))
            shown = 0 if options == [placeholder] else len(found)
            pinned = getattr(command, "pinned_motion_clip_id", None)
            pin_text = (
                command.motion_clip_label(pinned)
                if pinned is not None and hasattr(command, "motion_clip_label")
                else "(random)"
            )
            self._motion_clip_hint.content = (
                f"Showing **{shown}** of **{total}** clips."
            )
            if self._motion_clip_name_html is not None:
                self._motion_clip_name_html.content = _clip_name_html(
                    str(dropdown.value), pin_text
                )

        def _apply_clip_id(clip_id: int | None) -> None:
            command.pin_motion_clip(clip_id)
            self._reset_requested = True
            preferred = (
                command.motion_clip_label(clip_id)
                if clip_id is not None
                else None
            )
            _set_matches(
                search_box.value if clip_id is None else str(clip_id),
                preferred=preferred,
            )
            self._update_status_display()

        @search_box.on_update
        def _(_) -> None:
            _set_matches(search_box.value)

        @dropdown.on_update
        def _(_) -> None:
            pinned = getattr(command, "pinned_motion_clip_id", None)
            pin_text = (
                command.motion_clip_label(pinned)
                if pinned is not None and hasattr(command, "motion_clip_label")
                else "(random)"
            )
            if self._motion_clip_name_html is not None:
                self._motion_clip_name_html.content = _clip_name_html(
                    str(dropdown.value), pin_text
                )

        @actions.on_click
        def _(event) -> None:
            action = event.target.value
            if action == "Unpin":
                _apply_clip_id(None)
                return
            if action == "Random":
                n = int(getattr(command, "num_motion_clips", 0))
                if n <= 0:
                    return
                _apply_clip_id(random.randrange(n))
                return
            head = str(dropdown.value).split(":", 1)[0].strip()
            if not head.isdigit():
                return
            _apply_clip_id(int(head))

        _set_matches("")

    def _setup_reward_tabs(self, tabs) -> None:
        term_names = []
        if hasattr(self.env, "get_reward_term_names"):
            term_names = list(self.env.get_reward_term_names())
        if not term_names:
            return

        frame_time = float(getattr(self.env, "step_dt", 0.02))
        with tabs.add_tab("Rewards", icon=viser.Icon.CHART_LINE):
            self._reward_bar_panel = DarkRewardBarPanel(
                self._server,
                term_names,
                update_dt=frame_time,
                max_terms=max(40, len(term_names)),
            )
            self._reward_plotter = ViserTermPlotter(
                self._server,
                term_names,
                name="Reward",
                env_idx=self._scene.env_idx,
            )

    @property
    def scene(self) -> ViserMujocoScene | None:
        return getattr(self, "_scene", None)

    def add_batched_axes(self, name: str):
        axes_handle = self._server.scene.add_batched_axes(
            name=name,
            batched_wxyzs=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(
                self.env.num_envs, 4
            ),
            batched_positions=torch.tensor([[0.0, 0.0, 0.0]]).expand(
                self.env.num_envs, 3
            ),
            batched_scales=torch.tensor([[1.0, 1.0, 1.0]]).expand(
                self.env.num_envs, 3
            ),
        )
        return axes_handle

    def add_line_segments(
        self, name: str, colors: tuple[float, float, float] | torch.Tensor
    ):
        lines_handle = self._server.scene.add_line_segments(
            name=name,
            points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).expand(
                self.env.num_envs, 2, 3
            ),
            colors=colors,
        )
        return lines_handle

    def clear(self):
        if self._scene is None:
            return
        self._scene.clear()

    # ------------------------------------------------------------------
    # Playback controls (used by play scripts)
    # ------------------------------------------------------------------

    def consume_reset_request(self) -> bool:
        if not self._reset_requested:
            return False
        self._reset_requested = False
        return True

    def on_env_reset(self) -> None:
        self._step_count = 0
        if self._reward_bar_panel is not None:
            self._reward_bar_panel.clear_histories()
        if self._reward_plotter is not None:
            self._reward_plotter.clear_histories()
        self._update_status_display()

    def idle_while_paused(self) -> None:
        """Block while paused (unless a single step is pending)."""
        while self._paused and not self._single_step:
            self._refresh_gui_idle()
            time.sleep(1.0 / 60.0)

    def note_step_done(self) -> None:
        self._step_count += 1
        if self._single_step:
            self._single_step = False
            self._paused = True
            self._sync_pause_button()
        self._maybe_on_env_switch()
        self._update_reward_overlays(paused=False)
        self._update_status_display()

    def _sync_pause_button(self) -> None:
        if self._pause_button is None:
            return
        self._pause_button.label = "Play" if self._paused else "Pause"
        self._pause_button.icon = (
            viser.Icon.PLAYER_PLAY if self._paused else viser.Icon.PLAYER_PAUSE
        )

    def _refresh_gui_idle(self) -> None:
        if not self._is_setup:
            return
        self._maybe_on_env_switch()
        self._update_status_display()
        try:
            self._push_scene()
        except Exception:
            pass

    def _maybe_on_env_switch(self) -> None:
        if self._scene is None:
            return
        env_idx = int(self._scene.env_idx)
        if env_idx == self._prev_env_idx:
            return
        self._prev_env_idx = env_idx
        if self._reward_bar_panel is not None:
            self._reward_bar_panel.clear_histories()
        if self._reward_plotter is not None:
            self._reward_plotter.clear_histories()
            self._reward_plotter.update_env_idx(env_idx)

    def _update_reward_overlays(self, *, paused: bool) -> None:
        if paused or self._scene is None:
            return
        if self._reward_bar_panel is None and self._reward_plotter is None:
            return
        if not hasattr(self.env, "get_reward_terms_for_env"):
            return
        terms = self.env.get_reward_terms_for_env(int(self._scene.env_idx))
        if self._reward_plotter is not None:
            self._reward_plotter.update(terms)
        if self._reward_bar_panel is not None:
            self._reward_bar_panel.update(terms)

    def _update_status_display(self) -> None:
        if self._status_html is None:
            return
        env_idx = int(getattr(self._scene, "env_idx", 0)) if self._is_setup else 0
        lines = [
            f"<strong>Status:</strong> {'Paused' if self._paused else 'Running'}",
            f"<strong>Steps:</strong> {self._step_count}",
            f"<strong>Speed:</strong> {_format_speed(self.speed_scale)}",
            f"<strong>Env:</strong> {env_idx} / {self.env.num_envs}",
        ]

        episode_len = getattr(self.env, "episode_length_buf", None)
        if isinstance(episode_len, torch.Tensor) and episode_len.numel() > env_idx:
            lines.append(
                f"<strong>Episode Len:</strong> {int(episode_len[env_idx].item())}"
            )

        command = getattr(self.env, "command_manager", None)
        if command is not None:
            motion_len = getattr(command, "motion_len", None)
            t = getattr(command, "t", None)
            clip_id = None
            if hasattr(command, "current_motion_clip_id"):
                try:
                    clip_id = int(command.current_motion_clip_id(env_idx))
                except Exception:
                    clip_id = None
            if clip_id is not None and clip_id >= 0:
                if hasattr(command, "motion_clip_label"):
                    clip_label = command.motion_clip_label(clip_id)
                    lines.append(
                        "<strong>Clip:</strong><br/>"
                        '<span style="word-break:break-all;overflow-wrap:anywhere;">'
                        f"{clip_label}</span>"
                    )
                else:
                    lines.append(f"<strong>Motion ID:</strong> {clip_id}")
            else:
                motion_ids = getattr(command, "motion_ids", None)
                if isinstance(motion_ids, torch.Tensor) and motion_ids.numel() > env_idx:
                    lines.append(
                        f"<strong>Motion ID:</strong> {int(motion_ids[env_idx].item())}"
                    )
            pinned = getattr(command, "pinned_motion_clip_id", None)
            if pinned is not None:
                lines.append(f"<strong>Pinned:</strong> {pinned}")
            if (
                isinstance(t, torch.Tensor)
                and isinstance(motion_len, torch.Tensor)
                and t.numel() > env_idx
                and motion_len.numel() > env_idx
            ):
                lines.append(
                    "<strong>Motion t:</strong> "
                    f"{int(t[env_idx].item())} / {int(motion_len[env_idx].item())}"
                )
            viz = getattr(command, "viz", None)
            if viz is not None:
                lines.append(f"<strong>Ref Viz:</strong> {getattr(viz, 'mode', '?')}")

        if self._checkpoint_browser is not None:
            lines.append(
                f"<strong>Checkpoint:</strong> {self._checkpoint_browser.current_path.name}"
            )

        self._status_html.content = (
            f'<div style="font-size: 0.85em; line-height: 1.25; '
            f'padding: 0 1em 0.5em 1em; color: {_UI_TEXT};">'
            + "<br/>".join(lines)
            + "</div>"
        )

    # ------------------------------------------------------------------
    # MDP debug primitives (vectors / points), synced in update()
    # ------------------------------------------------------------------

    def clear_debug(self) -> None:
        self._debug_line_pts.clear()
        self._debug_line_cols.clear()
        self._debug_point_pts.clear()
        self._debug_point_cols.clear()
        scene = self.scene
        if scene is not None:
            scene.clear()

    def vector(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        size: float = 2.0,
        color: tuple[float, ...] = (0.0, 1.0, 1.0, 1.0),
    ) -> None:
        del size
        x_np = x.detach().cpu().reshape(-1, 3).numpy().astype(np.float32)
        v_np = v.detach().cpu().reshape(-1, 3).numpy().astype(np.float32)
        if x_np.shape != v_np.shape:
            raise ValueError(f"x and v must match, got {x_np.shape} and {v_np.shape}")
        seg = np.stack([x_np, x_np + v_np], axis=1)
        rgb = np.array(_rgba_to_rgb255(color), dtype=np.uint8)
        cols = np.broadcast_to(rgb, (seg.shape[0], 2, 3)).copy()
        self._debug_line_pts.append(seg)
        self._debug_line_cols.append(cols)

    def point(
        self,
        x: torch.Tensor,
        color: tuple[float, ...] = (1.0, 0.0, 0.0, 1.0),
        size: float = 10.0,
    ) -> None:
        pts = x.detach().cpu().reshape(-1, 3).numpy().astype(np.float32)
        rgb = np.array(_rgba_to_rgb255(color), dtype=np.uint8)
        cols = np.broadcast_to(rgb, (pts.shape[0], 3)).copy()
        self._debug_point_pts.append(pts)
        self._debug_point_cols.append(cols)
        self._debug_point_size = max(float(size) * 0.002, 0.005)

    def plot(
        self,
        x: torch.Tensor,
        size: float = 2.0,
        color: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        del size
        x_np = x.detach().cpu().reshape(-1, 3).numpy().astype(np.float32)
        if x_np.shape[0] < 2:
            return
        seg = np.stack([x_np[:-1], x_np[1:]], axis=1)
        rgb = np.array(_rgba_to_rgb255(color), dtype=np.uint8)
        cols = np.broadcast_to(rgb, (seg.shape[0], 2, 3)).copy()
        self._debug_line_pts.append(seg)
        self._debug_line_cols.append(cols)

    def _sync_debug_geometry(self) -> None:
        if self._debug_line_pts:
            points = np.concatenate(self._debug_line_pts, axis=0)
            colors = np.concatenate(self._debug_line_cols, axis=0)
        else:
            points = np.zeros((0, 2, 3), dtype=np.float32)
            colors = np.zeros((0, 2, 3), dtype=np.uint8)

        if self._line_handle is None:
            init_pts = (
                points if points.shape[0] > 0 else np.zeros((1, 2, 3), dtype=np.float32)
            )
            init_cols = (
                colors if colors.shape[0] > 0 else np.zeros((1, 2, 3), dtype=np.uint8)
            )
            self._line_handle = self._server.scene.add_line_segments(
                "/debug/mdp_lines",
                init_pts,
                init_cols,
                line_width=2.0,
                visible=points.shape[0] > 0,
            )
        elif points.shape[0] == 0:
            self._line_handle.visible = False
        else:
            self._line_handle.points = points
            self._line_handle.colors = colors
            self._line_handle.visible = True

        if self._debug_point_pts:
            pts = np.concatenate(self._debug_point_pts, axis=0)
            cols = np.concatenate(self._debug_point_cols, axis=0)
        else:
            pts = np.zeros((0, 3), dtype=np.float32)
            cols = np.zeros((0, 3), dtype=np.uint8)

        if self._point_handle is None:
            init_pts = pts if pts.shape[0] > 0 else np.zeros((1, 3), dtype=np.float32)
            init_cols = cols if cols.shape[0] > 0 else np.zeros((1, 3), dtype=np.uint8)
            self._point_handle = self._server.scene.add_point_cloud(
                "/debug/mdp_points",
                init_pts,
                init_cols,
                point_size=self._debug_point_size,
                visible=pts.shape[0] > 0,
            )
        elif pts.shape[0] == 0:
            self._point_handle.visible = False
        else:
            self._point_handle.points = pts
            self._point_handle.colors = cols
            self._point_handle.point_size = self._debug_point_size
            self._point_handle.visible = True

    # ------------------------------------------------------------------
    # Camera frustums
    # ------------------------------------------------------------------

    def register_camera(
        self,
        name: str,
        *,
        fov_y: float,
        aspect: float,
        scale: float = 0.15,
    ):
        """Create a Viser camera frustum (OpenCV +Z forward)."""
        if name in self._cameras:
            return self._cameras[name]
        handle = self._server.scene.add_camera_frustum(
            f"/cameras/{name}",
            fov=float(fov_y),
            aspect=float(aspect),
            scale=float(scale),
            color=(200, 200, 200),
            format="jpeg",
        )
        self._cameras[name] = handle
        return handle

    def _update_selected_env(self):
        scene = self._scene
        if scene is None:
            raise RuntimeError("MjLab viewer is not set up.")

        env_idx = int(scene.env_idx)
        body_xpos = self.sim.data.xpos[env_idx : env_idx + 1].cpu().numpy()
        body_xmat = self.sim.data.xmat[env_idx : env_idx + 1].cpu().numpy()
        if scene.mj_model.nmocap > 0:
            mocap_pos = self.sim.data.mocap_pos[env_idx : env_idx + 1].cpu().numpy()
            mocap_quat = self.sim.data.mocap_quat[env_idx : env_idx + 1].cpu().numpy()
        else:
            mocap_pos = np.zeros((1, 0, 3))
            mocap_quat = np.zeros((1, 0, 4))

        scene_offset = np.zeros(3)
        if scene.camera_tracking_enabled and scene._tracked_body_id is not None:
            tracked_pos = body_xpos[0, scene._tracked_body_id, :].copy()
            scene_offset = -tracked_pos

        contacts = None
        if scene.show_contact_points or scene.show_contact_forces:
            scene.mj_data.qpos[:] = self.sim.data.qpos[env_idx].cpu().numpy()
            scene.mj_data.qvel[:] = self.sim.data.qvel[env_idx].cpu().numpy()
            if scene.mj_model.nmocap > 0:
                scene.mj_data.mocap_pos[:] = mocap_pos[0]
                scene.mj_data.mocap_quat[:] = mocap_quat[0]
            import mujoco

            mujoco.mj_forward(scene.mj_model, scene.mj_data)
            contacts = scene._extract_contacts_from_mjdata(scene.mj_data)

        scene._update_visualization(
            body_xpos,
            body_xmat,
            mocap_pos,
            mocap_quat,
            0,
            scene_offset,
            contacts,
        )
        scene._sync_debug_visualizations(scene_offset)

    def _push_scene(self) -> None:
        if self._scene is None:
            raise RuntimeError("MjLab viewer is not set up.")
        if self._scene.show_only_selected and self.env.num_envs > 1:
            with ScopedTimer("viewer.update.selected_fast_path", sync=False):
                self._update_selected_env()
        else:
            with ScopedTimer("viewer.update.scene_update", sync=False):
                with self._server.atomic():
                    self._scene.update(self.sim.data)
                    self._server.flush()

    def update(self):
        if self._scene is None:
            raise RuntimeError("MjLab viewer is not set up.")
        self._sync_debug_geometry()
        self._maybe_on_env_switch()
        self._update_status_display()
        self._push_scene()

    def close(self):
        if self._reward_bar_panel is not None:
            self._reward_bar_panel.cleanup()
        if self._reward_plotter is not None:
            self._reward_plotter.cleanup()
        self._server.stop()
