from __future__ import annotations

from collections import deque
import tkinter as tk

from jarvis.monitoring.live import SystemSnapshot


class MonitorPanel(tk.Frame):
    """Compact 60-sample HUD chart rendered with one Canvas update per second."""

    def __init__(self, master: tk.Misc, *, bg: str, panel: str, border: str) -> None:
        super().__init__(master, bg=panel, highlightbackground=border, highlightthickness=1)
        self._panel = panel
        self._history: deque[tuple[float, float, float]] = deque(maxlen=60)
        self._activity: deque[str] = deque(maxlen=5)
        tk.Label(
            self, text="LIVE SYSTEM MONITOR", bg=panel, fg="#38D9FF",
            font=("Consolas", 9, "bold"), anchor="w",
        ).pack(fill="x", padx=14, pady=(13, 3))
        tk.Label(
            self, text="60 SECOND TELEMETRY", bg=panel, fg="#587087",
            font=("Consolas", 7), anchor="w",
        ).pack(fill="x", padx=14, pady=(0, 8))
        self.metrics = tk.Label(
            self, bg=panel, fg="#B8CADB", justify="left", anchor="nw",
            font=("Consolas", 8), text="INITIALIZING …",
        )
        self.metrics.pack(fill="x", padx=14)
        self.chart = tk.Canvas(
            self, height=175, bg="#090F18", highlightbackground=border,
            highlightthickness=1, borderwidth=0,
        )
        self.chart.pack(fill="x", padx=12, pady=(13, 8))
        legend = tk.Frame(self, bg=panel)
        legend.pack(fill="x", padx=14)
        for color, text in (("#38D9FF", "CPU"), ("#C493FF", "RAM"), ("#4FE0A0", "GPU")):
            tk.Label(legend, text=f"● {text}", bg=panel, fg=color, font=("Consolas", 7)).pack(
                side="left", padx=(0, 8)
            )
        self.training = tk.Label(
            self, bg=panel, fg="#FFD166", justify="left", anchor="nw",
            font=("Consolas", 8), text="TRAINING  INACTIVE",
        )
        self.training.pack(fill="x", padx=14, pady=(10, 12))
        tk.Label(
            self, text="ACTIVITY / EXECUTION", bg=panel, fg="#38D9FF",
            font=("Consolas", 8, "bold"), anchor="w",
        ).pack(fill="x", padx=14, pady=(2, 3))
        self.activity = tk.Label(
            self, bg=panel, fg="#7F96AB", justify="left", anchor="nw",
            font=("Consolas", 7), text="• READY",
        )
        self.activity.pack(fill="x", padx=14, pady=(0, 10))
        self.inference = tk.Label(
            self, bg=panel, fg="#B8CADB", justify="left", anchor="nw",
            font=("Consolas", 8), text="INFERENCE  —\nTOKENS/S   —",
        )
        self.inference.pack(fill="x", padx=14, pady=(0, 12))

    def add_activity(self, value: str) -> None:
        clean = " ".join(str(value).split())[:80]
        if not clean or (self._activity and self._activity[-1] == clean):
            return
        self._activity.append(clean)
        self.activity.configure(text="\n".join(f"• {item}" for item in self._activity))

    def update_inference(
        self, latency_ms: float | None, tokens_per_second: float | None,
    ) -> None:
        latency = "—" if latency_ms is None else f"{latency_ms:.1f} ms"
        rate = "—" if tokens_per_second is None else f"{tokens_per_second:.1f}"
        self.inference.configure(text=f"INFERENCE  {latency}\nTOKENS/S   {rate}")

    def update_snapshot(self, value: SystemSnapshot) -> None:
        gpu = value.gpu_percent if value.gpu_percent is not None else 0.0
        self._history.append((value.cpu_percent, value.ram_percent, gpu))
        if value.gpu_available:
            gpu_line = f"GPU       {gpu:>5.1f}%"
            if value.vram_total_mb:
                gpu_line += f"  VRAM {value.vram_used_mb or 0:>5.0f}/{value.vram_total_mb:.0f} MB"
            if value.gpu_temperature_c is not None:
                gpu_line += f"  {value.gpu_temperature_c:.0f}°C"
        else:
            gpu_line = "GPU       NOT AVAILABLE"
        self.metrics.configure(
            text=(
                f"CPU       {value.cpu_percent:>5.1f}%\n"
                f"JARVIS    {value.app_cpu_percent:>5.1f}% CPU\n"
                f"RAM       {value.ram_percent:>5.1f}%  {value.ram_used_mb / 1024:>5.1f} GB\n"
                f"JARVIS    {value.app_ram_mb:>5.1f} MB RAM\n"
                f"{gpu_line}\n"
                f"DISK      {value.disk_percent:>5.1f}%  {value.disk_free_gb:>5.1f} GB FREE\n"
                f"NETWORK   {value.network_kbps:>7.1f} KB/s"
            )
        )
        self._draw_chart()

    def update_training(self, loss: float | None, tokens_per_second: float | None) -> None:
        if loss is None:
            self.training.configure(text="TRAINING  INACTIVE")
            return
        self.training.configure(
            text=f"TRAINING  ACTIVE\nLOSS      {loss:.4f}\nTOKENS/S  {tokens_per_second or 0:.1f}"
        )

    def _draw_chart(self) -> None:
        canvas = self.chart
        canvas.delete("all")
        width = max(40, canvas.winfo_width())
        height = max(40, canvas.winfo_height())
        for index in range(1, 4):
            y = height * index / 4
            canvas.create_line(0, y, width, y, fill="#152333")
        if len(self._history) < 2:
            return
        values = list(self._history)
        colors = ("#38D9FF", "#C493FF", "#4FE0A0")
        step = width / 59.0
        offset = 60 - len(values)
        for series, color in enumerate(colors):
            points: list[float] = []
            for index, row in enumerate(values):
                points.extend(((offset + index) * step, height - row[series] / 100.0 * height))
            canvas.create_line(*points, fill=color, width=2, smooth=True)
