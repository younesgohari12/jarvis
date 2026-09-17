from __future__ import annotations

import math
import random
import time
import tkinter as tk

from jarvis.runtime.hardware import PerformanceProfile


class HolographicCore(tk.Canvas):
    """A lightweight 3D-looking energy core rendered entirely by Tk Canvas."""

    STATE_COLORS = {
        "IDLE": ("#1B7FA4", "#38D9FF"),
        "READY": ("#1B7FA4", "#38D9FF"),
        "LISTENING": ("#3C7CFF", "#70A7FF"),
        "THINKING": ("#7D57FF", "#C493FF"),
        "SEARCHING": ("#155AA8", "#52B9FF"),
        "READING": ("#007E91", "#5FE7F2"),
        "PLANNING": ("#425CC7", "#8FA8FF"),
        "WORKING": ("#00A87A", "#4FE0A0"),
        "VERIFYING": ("#168A76", "#62E6C8"),
        "LEARNING": ("#A62D89", "#FF9FEA"),
        "TRAINING": ("#934B00", "#FFB347"),
        "SPEAKING": ("#B07800", "#FFD166"),
        "ERROR": ("#A42B48", "#FF6B7A"),
    }

    def __init__(
        self,
        master: tk.Misc,
        profile: PerformanceProfile,
        animation_enabled: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(
            master,
            background="#090D14",
            highlightthickness=0,
            borderwidth=0,
            **kwargs,
        )
        self.profile = profile
        self.animation_enabled = animation_enabled
        self.state = "IDLE"
        self._phase = 0.0
        self._last_activity = time.monotonic()
        self._after_id: str | None = None
        self._size = (0, 0)
        self._arcs: list[int] = []
        self._particles: list[tuple[int, float, float, float]] = []
        self._core_layers: list[int] = []
        self._status_item = 0
        self._rng = random.Random(2026)
        self.bind("<Configure>", self._on_configure)
        self.bind("<Map>", lambda _event: self._schedule(10))
        self.bind("<Unmap>", lambda _event: self._schedule(1000))
        self._schedule(50)

    def set_state(self, state: str) -> None:
        normalized = state.upper()
        if normalized not in self.STATE_COLORS:
            normalized = "IDLE"
        self.state = normalized
        self._last_activity = time.monotonic()
        if self._status_item:
            self.itemconfigure(self._status_item, text=normalized)

    def set_profile(self, profile: PerformanceProfile) -> None:
        self.profile = profile
        self._draw_static(force=True)

    def set_animation(self, enabled: bool) -> None:
        self.animation_enabled = bool(enabled)
        if not enabled:
            self._draw_frame(0.0)

    def _on_configure(self, event: tk.Event[tk.Canvas]) -> None:
        size = (max(1, event.width), max(1, event.height))
        if abs(size[0] - self._size[0]) > 4 or abs(size[1] - self._size[1]) > 4:
            self._size = size
            self._draw_static(force=True)

    def _draw_static(self, force: bool = False) -> None:
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        if width < 30 or height < 30:
            return
        if force:
            self.delete("all")
        cx, cy = width / 2, height / 2 - 4
        radius = min(width, height) * 0.32

        for line in range(-5, 6):
            offset = line * max(18, width / 20)
            self.create_line(
                cx + offset, cy - radius * 1.35, cx + offset * 1.7, cy + radius * 1.35,
                fill="#0D2638", width=1, tags="grid"
            )
        for line in range(-3, 4):
            y = cy + line * max(18, height / 12)
            self.create_line(
                cx - radius * 1.8, y, cx + radius * 1.8, y,
                fill="#0C2232", width=1, tags="grid"
            )

        self._core_layers = []
        glow_colors = ["#0B1824", "#0C2332", "#10344A", "#15516B", "#1C7391"]
        glow_count = min(self.profile.glow_layers, len(glow_colors))
        for index in range(glow_count):
            layer_radius = radius * (0.92 - index * 0.11)
            item = self.create_oval(
                cx - layer_radius, cy - layer_radius,
                cx + layer_radius, cy + layer_radius,
                fill=glow_colors[index], outline="", tags="core"
            )
            self._core_layers.append(item)
        inner = radius * 0.34
        self._core_layers.append(
            self.create_oval(
                cx - inner, cy - inner, cx + inner, cy + inner,
                fill="#59E7FF", outline="#D5FAFF", width=2, tags="core"
            )
        )
        shine = radius * 0.13
        self.create_oval(
            cx - inner * 0.48, cy - inner * 0.58,
            cx - inner * 0.48 + shine, cy - inner * 0.58 + shine,
            fill="#E6FCFF", outline="", tags="core"
        )

        self._arcs = []
        for index, scale in enumerate((1.02, 1.18, 1.36, 1.52)):
            ring_radius = radius * scale
            extent = 215 - index * 24
            item = self.create_arc(
                cx - ring_radius, cy - ring_radius * 0.62,
                cx + ring_radius, cy + ring_radius * 0.62,
                start=index * 47, extent=extent, style="arc",
                outline="#38D9FF" if index < 2 else "#275C82",
                width=2 if index < 3 else 1, tags="rings"
            )
            self._arcs.append(item)

        self._particles = []
        for _ in range(self.profile.particles):
            angle = self._rng.random() * math.tau
            distance = radius * self._rng.uniform(1.0, 1.7)
            speed = self._rng.uniform(0.25, 0.9)
            size = self._rng.choice((1.5, 2.0, 2.5))
            x = cx + math.cos(angle) * distance
            y = cy + math.sin(angle) * distance * 0.56
            item = self.create_oval(
                x - size, y - size, x + size, y + size,
                fill="#54DBFF", outline="", tags="particles"
            )
            self._particles.append((item, angle, distance, speed))

        self.create_text(
            cx, cy + radius * 1.58, text="JARVIS CORE  •  FROM-SCRATCH INTELLIGENCE",
            fill="#4B6980", font=("Consolas", 8), tags="label"
        )
        self._status_item = self.create_text(
            cx, cy, text=self.state, fill="#07131C",
            font=("Segoe UI Semibold", 10), tags="label"
        )
        self._draw_frame(self._phase)

    def _draw_frame(self, phase: float) -> None:
        width, height = max(1, self.winfo_width()), max(1, self.winfo_height())
        cx, cy = width / 2, height / 2 - 4
        radius = min(width, height) * 0.32
        dark, bright = self.STATE_COLORS[self.state]
        speed = 2.0 if self.state in {
            "THINKING", "SEARCHING", "READING", "PLANNING", "WORKING",
            "VERIFYING", "LEARNING", "TRAINING"
        } else 1.0
        for index, item in enumerate(self._arcs):
            direction = -1 if index % 2 else 1
            self.itemconfigure(
                item,
                start=(index * 47 + direction * phase * (22 + index * 8) * speed) % 360,
                outline=bright if index < 2 else dark,
            )
        pulse_strength = 0.035 if self.state == "IDLE" else 0.075
        pulse = 1.0 + math.sin(phase * (2.0 if self.state != "IDLE" else 1.0)) * pulse_strength
        if self._core_layers:
            inner = radius * 0.34 * pulse
            self.coords(
                self._core_layers[-1], cx - inner, cy - inner, cx + inner, cy + inner
            )
            self.itemconfigure(self._core_layers[-1], fill=bright, outline="#E1FBFF")
        for item, base_angle, distance, particle_speed in self._particles:
            angle = base_angle + phase * particle_speed * speed
            x = cx + math.cos(angle) * distance
            y = cy + math.sin(angle) * distance * 0.56
            size = 2.0
            self.coords(item, x - size, y - size, x + size, y + size)
            self.itemconfigure(item, fill=bright)

    def _schedule(self, delay_ms: int) -> None:
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
        self._after_id = self.after(max(10, delay_ms), self._tick)

    def _tick(self) -> None:
        if not self.winfo_exists():
            return
        if not self.animation_enabled or not self.winfo_ismapped():
            self._schedule(1000)
            return
        idle_seconds = time.monotonic() - self._last_activity
        if idle_seconds > 60:
            fps = 1
        elif idle_seconds > 5 and self.state in {"IDLE", "READY"}:
            fps = self.profile.idle_fps
        else:
            fps = self.profile.active_fps
        self._phase += 1.0 / max(1, fps)
        self._draw_frame(self._phase)
        self._schedule(round(1000 / max(1, fps)))

    def destroy(self) -> None:
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
        super().destroy()
