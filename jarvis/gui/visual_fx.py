from __future__ import annotations

import math
import random
import tkinter as tk


class NeuralBackdrop(tk.Canvas):
    """Low-cost animated depth field used behind the desktop HUD."""

    def __init__(self, master: tk.Misc, *, bg: str = "#03060B", accent: str = "#31E6FF") -> None:
        super().__init__(master, bg=bg, highlightthickness=0, borderwidth=0)
        self.accent = accent
        self._phase = 0.0
        self._pointer = (0.5, 0.5)
        self._after_id: str | None = None
        self._rng = random.Random(18062026)
        self._stars = [
            (self._rng.random(), self._rng.random(), self._rng.uniform(0.15, 0.95), self._rng.uniform(0.4, 1.6))
            for _ in range(56)
        ]
        self.bind("<Configure>", lambda _e: self._draw())
        self.bind_all("<Motion>", self._motion, add="+")
        self._schedule()

    def _motion(self, event: tk.Event[tk.Misc]) -> None:
        w = max(1, self.winfo_toplevel().winfo_width())
        h = max(1, self.winfo_toplevel().winfo_height())
        self._pointer = (min(1.0, max(0.0, event.x_root / w)), min(1.0, max(0.0, event.y_root / h)))

    @staticmethod
    def _blend(hex_color: str, factor: float) -> str:
        factor = max(0.0, min(1.0, factor))
        raw = hex_color.lstrip("#")
        r, g, b = int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
        return f"#{int(r*factor):02X}{int(g*factor):02X}{int(b*factor):02X}"

    def _draw(self) -> None:
        if not self.winfo_exists():
            return
        w = max(80, self.winfo_width())
        h = max(80, self.winfo_height())
        self.delete("all")

        # Atmospheric radial bands.
        cx = w * (0.52 + (self._pointer[0] - 0.5) * 0.025)
        cy = h * (0.42 + (self._pointer[1] - 0.5) * 0.02)
        for i in range(8, 0, -1):
            rx = w * (0.12 + i * 0.055)
            ry = h * (0.08 + i * 0.045)
            factor = 0.035 + (9 - i) * 0.015
            self.create_oval(cx-rx, cy-ry, cx+rx, cy+ry, outline=self._blend(self.accent, factor), width=1)

        # Star/particle depth field with subtle parallax.
        px = (self._pointer[0] - 0.5) * 18
        py = (self._pointer[1] - 0.5) * 12
        for sx, sy, depth, size in self._stars:
            x = (sx * w + px * depth + math.sin(self._phase * 0.7 + sx * 9) * 2 * depth) % w
            y = (sy * h + py * depth + math.cos(self._phase * 0.45 + sy * 8) * 1.5 * depth) % h
            radius = max(0.7, size * depth)
            color = self._blend(self.accent, 0.16 + depth * 0.28)
            self.create_oval(x-radius, y-radius, x+radius, y+radius, fill=color, outline="")

        # Perspective floor grid.
        horizon = h * 0.66
        van_x = w * (0.5 + (self._pointer[0] - 0.5) * 0.03)
        base = h + 14
        grid_color = "#0B2634"
        glow_color = "#0D3444"
        for i in range(-14, 15):
            bx = w / 2 + i * (w / 13)
            self.create_line(van_x, horizon, bx, base, fill=grid_color, width=1)
        for i in range(1, 15):
            t = i / 15
            eased = t * t
            y = horizon + (base - horizon) * eased
            self.create_line(0, y, w, y, fill=glow_color if i % 3 == 0 else grid_color, width=1)

        # Horizon energy line.
        self.create_line(0, horizon, w, horizon, fill="#104B5E", width=1)
        self.create_line(w*0.18, horizon+1, w*0.82, horizon+1, fill="#1A6D82", width=1)

    def _tick(self) -> None:
        self._phase += 0.055
        if self.winfo_ismapped():
            self._draw()
        self._schedule()

    def _schedule(self) -> None:
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
        self._after_id = self.after(90, self._tick)

    def destroy(self) -> None:
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
        super().destroy()
