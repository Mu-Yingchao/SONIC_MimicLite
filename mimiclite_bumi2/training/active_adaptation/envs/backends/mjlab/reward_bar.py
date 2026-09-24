"""Reward bar panel styled for light Viser sidebars (dark text)."""

from __future__ import annotations

import html
import warnings
from collections import deque

import numpy as np
import viser


class DarkRewardBarPanel:
    """HTML bar panel with dark labels for readability on light themes."""

    def __init__(
        self,
        server: viser.ViserServer,
        term_names: list[str],
        update_dt: float,
        max_terms: int = 20,
    ) -> None:
        self._server = server
        if len(term_names) > max_terms:
            dropped = term_names[max_terms:]
            warnings.warn(
                f"DarkRewardBarPanel: {len(term_names)} reward terms exceed "
                f"max_terms={max_terms}; showing first {max_terms}. "
                f"Hidden: {dropped}.",
                UserWarning,
                stacklevel=2,
            )
        self._term_names = term_names[:max_terms]
        self._window_steps = max(1, round(1.0 / max(update_dt, 1e-6)))
        self._histories: dict[str, deque[float]] = {
            name: deque(maxlen=self._window_steps) for name in self._term_names
        }
        self._html_handle = self._server.gui.add_html("")
        self._render_empty()

    def update(self, terms: list[tuple[str, np.ndarray]]) -> None:
        for name, arr in terms:
            if name not in self._histories:
                continue
            val = float(arr[0])
            if np.isfinite(val):
                self._histories[name].append(val)
        self._render()

    def clear_histories(self) -> None:
        for history in self._histories.values():
            history.clear()
        self._render_empty()

    def cleanup(self) -> None:
        self._html_handle.remove()

    def _render_empty(self) -> None:
        self._html_handle.content = (
            '<div style="padding:0.5em;color:#333;font-size:0.85em;">'
            "Waiting for data…</div>"
        )

    def _render(self) -> None:
        means: dict[str, float] = {}
        for name in self._term_names:
            buf = self._histories[name]
            means[name] = (sum(buf) / len(buf)) if buf else 0.0

        max_abs = max((abs(v) for v in means.values()), default=1e-8)
        if max_abs < 1e-12:
            max_abs = 1e-12

        rows: list[str] = []
        for name in self._term_names:
            val = means[name]
            pct = abs(val) / max_abs * 100.0
            color = "#2e7d32" if val >= 0 else "#c62828"
            text_color = "#fff" if pct > 30 else "#111"

            if abs(val) < 1e-6 and val != 0:
                val_str = f"{val:.2e}"
            else:
                val_str = f"{val:.4f}"

            safe_name = html.escape(name, quote=True)
            rows.append(
                f'<div style="display:flex;align-items:center;margin:2px 0;">'
                f'<span style="min-width:120px;font-size:0.78em;text-align:right;'
                f"padding-right:6px;color:#222;white-space:nowrap;overflow:hidden;"
                f'text-overflow:ellipsis;" title="{safe_name}">{safe_name}</span>'
                f'<div style="flex:1;background:#e6e6e6;border-radius:3px;height:18px;'
                f'position:relative;overflow:hidden;">'
                f'<div style="width:{pct:.1f}%;height:100%;background:{color};'
                f'border-radius:3px;transition:width 0.15s;"></div>'
                f'<span style="position:absolute;right:4px;top:0;line-height:18px;'
                f'font-size:0.72em;color:{text_color};">{val_str}</span>'
                f"</div></div>"
            )

        self._html_handle.content = (
            '<div style="padding:0.3em 0.5em;font-family:monospace;color:#111;">'
            + "".join(rows)
            + "</div>"
        )
