from __future__ import annotations

import hashlib
import importlib.util
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

from jarvis.config import load_config
from jarvis.learning.failures import FailureCollector
from jarvis.learning.queue import LearningQueue
from jarvis.lab.catalog import DatasetCatalog, ModelCatalog, ModelRecord
from jarvis.monitoring.live import LiveMonitor
from jarvis.neural.config import TransformerConfig
from jarvis.neural.transformer import JarvisTransformer
from jarvis.runtime.hardware import HardwareManager
from training.numpy_trainer import CurriculumRunner, TrainingControl


class TrainingChart(tk.Canvas):
    COLORS = {
        "loss": "#38D9FF", "validation_loss": "#FF9FEA",
        "learning_rate": "#FFD166", "tokens_per_second": "#4FE0A0",
    }

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, bg="#080F18", highlightthickness=0, height=230)
        self.values: dict[str, list[float]] = {key: [] for key in self.COLORS}
        self.bind("<Configure>", lambda _event: self.redraw())

    def add(self, **values: float) -> None:
        for name, value in values.items():
            if name in self.values:
                self.values[name].append(float(value))
                self.values[name] = self.values[name][-180:]
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width, height = max(60, self.winfo_width()), max(60, self.winfo_height())
        for index in range(1, 5):
            y = height * index / 5
            self.create_line(0, y, width, y, fill="#172637")
        legend_x = 10
        for name, color in self.COLORS.items():
            self.create_text(legend_x, 10, text=name.upper(), fill=color, anchor="nw", font=("Consolas", 7))
            legend_x += max(80, len(name) * 7)
            series = self.values[name]
            if len(series) < 2:
                continue
            visible = series[-120:]
            low, high = min(visible), max(visible)
            spread = max(high - low, abs(high) * 0.05, 1e-9)
            points: list[float] = []
            for index, value in enumerate(visible):
                x = index / max(1, len(visible) - 1) * (width - 20) + 10
                y = height - 15 - (value - low) / spread * (height - 45)
                points.extend((x, y))
            self.create_line(*points, fill=color, width=2, smooth=True)


class JarvisLab:
    BG = "#070B12"
    PANEL = "#0D1420"
    PANEL_2 = "#111C2A"
    INPUT = "#121E2D"
    TEXT = "#E7F2FF"
    MUTED = "#6F849B"
    CYAN = "#38D9FF"
    BLUE = "#426BFF"
    GREEN = "#4FE0A0"
    AMBER = "#FFD166"
    RED = "#FF6B7A"

    def __init__(self, root: tk.Tk | tk.Toplevel, project_root: Path) -> None:
        self.root = root
        self.project_root = project_root.resolve()
        self.root.title("JARVIS LAB v3 — آزمایشگاه هوش از صفر")
        self.root.configure(bg=self.BG)
        self.root.geometry("1220x820")
        self.root.minsize(980, 680)
        self.hardware = HardwareManager()
        self.live_monitor = LiveMonitor(self.hardware)
        app_config = load_config(self.project_root)
        self.learning_queue = LearningQueue(
            app_config.paths.data_dir / "learning_queue.jsonl",
            app_config.learning.max_corrections,
        )
        self.failure_collector = FailureCollector(app_config.paths.failure_queue)
        self.dataset_catalog = DatasetCatalog(self.project_root)
        self.model_catalog = ModelCatalog(self.project_root)
        self.model_records: dict[str, ModelRecord] = {}
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.training_control: TrainingControl | None = None
        self.training_thread: threading.Thread | None = None
        self.training_process: subprocess.Popen[str] | None = None
        self.control_file = self.project_root / "runtime_data" / "lab_training_control.json"
        self.current_training_event: dict[str, Any] = {}
        self._closing = False
        self._configure_styles()
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self._poll_events)
        self.root.after(500, self._sample_hardware)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Lab.TNotebook", background=self.BG, borderwidth=0)
        style.configure(
            "Lab.TNotebook.Tab", background=self.PANEL, foreground=self.MUTED,
            padding=(13, 8), font=("Consolas", 8, "bold"),
        )
        style.map(
            "Lab.TNotebook.Tab", background=[("selected", self.PANEL_2)],
            foreground=[("selected", self.CYAN)],
        )
        style.configure(
            "Lab.TCombobox", fieldbackground=self.INPUT, background=self.INPUT,
            foreground=self.TEXT, arrowcolor=self.CYAN,
        )
        style.map("Lab.TCombobox", foreground=[("readonly", self.TEXT)], fieldbackground=[("readonly", self.INPUT)])
        style.configure(
            "Lab.Treeview", background=self.PANEL, fieldbackground=self.PANEL,
            foreground=self.TEXT, rowheight=27, borderwidth=0,
        )
        style.configure("Lab.Treeview.Heading", background=self.PANEL_2, foreground=self.CYAN)

    def _build(self) -> None:
        header = tk.Frame(self.root, bg=self.BG)
        header.pack(fill="x", padx=18, pady=(15, 8))
        tk.Label(header, text="JARVIS LAB v3", bg=self.BG, fg=self.TEXT, font=("Segoe UI Semibold", 22)).pack(side="right")
        tk.Label(
            header, text="  آموزش • ارزیابی • بهبود  //  بدون وزن Pretrained",
            bg=self.BG, fg=self.CYAN, font=("Consolas", 9),
        ).pack(side="right", pady=(8, 0))
        self.lab_status = tk.Label(
            header, text="● READY", bg=self.PANEL_2, fg=self.GREEN,
            font=("Consolas", 9, "bold"), padx=12, pady=7,
        )
        self.lab_status.pack(side="left")

        self.notebook = ttk.Notebook(self.root, style="Lab.TNotebook")
        self.notebook.pack(fill="both", expand=True, padx=16, pady=(0, 15))
        self.tabs: dict[str, tk.Frame] = {}
        tab_labels = {
            "DASHBOARD": "داشبورد", "DATASET": "دیتاست", "TRAIN": "آموزش",
            "EVALUATE": "ارزیابی", "TEST": "تست", "MODELS": "مدل‌ها",
            "CORRECTIONS": "اصلاح‌ها", "FAILURES": "خطاها", "LOGS": "لاگ‌ها",
        }
        for name in ("DASHBOARD", "DATASET", "TRAIN", "EVALUATE", "TEST", "MODELS", "CORRECTIONS", "FAILURES", "LOGS"):
            frame = tk.Frame(self.notebook, bg=self.BG)
            self.tabs[name] = frame
            self.notebook.add(frame, text=tab_labels[name])
        self._build_dashboard()
        self._build_dataset()
        self._build_train()
        self._build_evaluate()
        self._build_test()
        self._build_models()
        self._build_corrections()
        self._build_failures()
        self._build_logs()

    def _panel(self, parent: tk.Misc) -> tk.Frame:
        return tk.Frame(parent, bg=self.PANEL, highlightbackground="#20344A", highlightthickness=1)

    def _button(self, parent: tk.Misc, label: str, command: Any, *, color: str | None = None) -> tk.Button:
        return tk.Button(
            parent, text=label, command=command, bg=color or self.PANEL_2, fg=self.TEXT,
            activebackground=self.CYAN, activeforeground=self.BG, borderwidth=0,
            padx=11, pady=8, font=("Consolas", 8, "bold"), cursor="hand2",
        )

    def _text(self, parent: tk.Misc, height: int = 5) -> tk.Text:
        return tk.Text(
            parent, height=height, bg=self.INPUT, fg=self.TEXT, insertbackground=self.CYAN,
            borderwidth=0, wrap="word", padx=9, pady=7, font=("Segoe UI", 9),
        )

    def _build_dashboard(self) -> None:
        tab = self.tabs["DASHBOARD"]
        tab.columnconfigure(0, weight=3)
        tab.columnconfigure(1, weight=2)
        tab.rowconfigure(1, weight=1)
        metrics = self._panel(tab)
        metrics.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(9, 9))
        names = (
            "model", "parameters", "epoch", "step", "progress", "train_loss",
            "validation_loss", "learning_rate", "tokens_per_second", "remaining",
            "checkpoint",
        )
        self.metric_vars = {name: tk.StringVar(value="—") for name in names}
        labels = {
            "model": "CURRENT MODEL", "parameters": "PARAMETERS", "epoch": "STAGE / EPOCH",
            "step": "STEP", "progress": "PROGRESS", "train_loss": "TRAIN LOSS",
            "validation_loss": "VALIDATION LOSS", "learning_rate": "LEARNING RATE",
            "tokens_per_second": "TOKENS / SEC", "remaining": "REMAINING STEPS",
            "checkpoint": "CURRENT CHECKPOINT",
        }
        for index, name in enumerate(names):
            card = tk.Frame(metrics, bg=self.PANEL_2)
            card.grid(row=index // 6, column=index % 6, sticky="nsew", padx=4, pady=4)
            metrics.columnconfigure(index % 6, weight=1)
            tk.Label(card, text=labels[name], bg=self.PANEL_2, fg=self.MUTED, font=("Consolas", 7)).pack(anchor="w", padx=8, pady=(6, 1))
            tk.Label(card, textvariable=self.metric_vars[name], bg=self.PANEL_2, fg=self.CYAN, font=("Consolas", 10, "bold"), wraplength=165).pack(anchor="w", padx=8, pady=(0, 7))
        chart_panel = self._panel(tab)
        chart_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        tk.Label(chart_panel, text="LIVE TRAINING CHARTS", bg=self.PANEL, fg=self.CYAN, font=("Consolas", 9, "bold")).pack(anchor="w", padx=12, pady=10)
        self.chart = TrainingChart(chart_panel)
        self.chart.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        hardware = self._panel(tab)
        hardware.grid(row=1, column=1, sticky="nsew")
        tk.Label(hardware, text="HARDWARE AUTO CONFIGURATION", bg=self.PANEL, fg=self.CYAN, font=("Consolas", 9, "bold")).pack(anchor="w", padx=12, pady=10)
        self.hardware_label = tk.Label(hardware, bg=self.PANEL, fg=self.TEXT, justify="left", anchor="nw", font=("Consolas", 9))
        self.hardware_label.pack(fill="both", expand=True, padx=12)
        suggestion = self._suggest_configuration()
        self.suggestion_label = tk.Label(
            hardware, text="RECOMMENDED\n" + "\n".join(f"{key}: {value}" for key, value in suggestion.items()),
            bg=self.PANEL_2, fg=self.AMBER, justify="left", anchor="nw", font=("Consolas", 8), padx=10, pady=9,
        )
        self.suggestion_label.pack(fill="x", padx=12, pady=12)
        try:
            registry = json.loads(
                (self.project_root / "models" / "model_registry_v2.json").read_text(encoding="utf-8")
            )
            active_profile = registry.get("active_profile", "nano")
            active = next(value for value in registry.get("profiles", []) if value.get("profile") == active_profile)
            self.metric_vars["model"].set(str(active.get("name", active_profile)))
            auxiliary_parameters = sum(
                int(model.get("parameters", 0)) for model in registry.get("auxiliary_models", [])
                if model.get("status") in {"trained", "trained_quality_gate_passed", "active"}
            )
            total_parameters = int(active.get("parameters", 0)) + auxiliary_parameters
            self.metric_vars["parameters"].set(f"{total_parameters:,}")
            self.metric_vars["validation_loss"].set(str(active.get("validation_metrics", {}).get("loss", "—")))
            self.metric_vars["checkpoint"].set(str(active.get("checkpoint_parent", "—")))
        except (OSError, ValueError, StopIteration, json.JSONDecodeError):
            pass

    def _suggest_configuration(self) -> dict[str, object]:
        ram = self.hardware.info.ram_total_mb
        cores = self.hardware.info.logical_cores
        gpu = self.hardware.info.gpu_available
        return {
            "batch_size": 4 if gpu and ram >= 16384 else 1,
            "gradient_accumulation": 8 if ram < 8192 else 4,
            "precision": "bf16/fp16" if gpu else "fp32",
            "sequence_length": 64 if ram >= 8192 else 32,
            "workers": min(4, max(1, cores // 2)),
            "checkpointing": "every curriculum stage",
        }

    def _build_dataset(self) -> None:
        tab = self.tabs["DATASET"]
        form = self._panel(tab)
        form.pack(fill="both", expand=True, pady=9)
        tk.Label(form, text="DATASET EDITOR  //  USER-AUTHORED ENTRIES", bg=self.PANEL, fg=self.CYAN, font=("Consolas", 10, "bold")).pack(anchor="w", padx=15, pady=(14, 8))
        row = tk.Frame(form, bg=self.PANEL)
        row.pack(fill="x", padx=15)
        self.dataset_type = tk.StringVar(value="Conversation")
        self.dataset_language = tk.StringVar(value="fa")
        for label, variable, values in (
            ("TYPE", self.dataset_type, ("Conversation", "Tool", "Instruction", "Reasoning", "Context")),
            ("LANGUAGE", self.dataset_language, ("fa", "en", "mixed")),
        ):
            tk.Label(row, text=label, bg=self.PANEL, fg=self.MUTED, font=("Consolas", 8)).pack(side="left", padx=(0, 6))
            ttk.Combobox(row, textvariable=variable, values=values, state="readonly", width=15, style="Lab.TCombobox").pack(side="left", padx=(0, 18))
        self.dataset_fields: dict[str, tk.Text] = {}
        for label, height in (("USER / INPUT", 4), ("JARVIS / OUTPUT", 5), ("TOOL", 2), ("ARGUMENTS (JSON)", 3)):
            tk.Label(form, text=label, bg=self.PANEL, fg=self.MUTED, font=("Consolas", 8)).pack(anchor="w", padx=15, pady=(10, 2))
            widget = self._text(form, height)
            widget.pack(fill="x", padx=15)
            self.dataset_fields[label] = widget
        self.dataset_status = tk.Label(form, text="", bg=self.PANEL, fg=self.GREEN, font=("Consolas", 8))
        self.dataset_status.pack(anchor="w", padx=15, pady=(9, 0))
        actions = tk.Frame(form, bg=self.PANEL)
        actions.pack(fill="x", padx=15, pady=9)
        for label, command, color in (
            ("ذخیره نمونه", self._save_dataset_entry, self.BLUE),
            ("بازخوانی", self.refresh_datasets, None),
            ("افزودن JSONL", self.import_dataset, None),
            ("فعال/غیرفعال", self.toggle_dataset, None),
            ("حذف Import", self.delete_dataset, self.RED),
        ):
            self._button(actions, label, command, color=color).pack(side="right", padx=(6, 0))
        self.dataset_tree = ttk.Treeview(
            form,
            columns=("path", "samples", "category", "language", "duplicates", "malformed", "enabled"),
            show="headings", height=6, style="Lab.Treeview",
        )
        columns = (
            ("path", "DATASET", 330), ("samples", "SAMPLES", 75),
            ("category", "CATEGORY", 150), ("language", "LANG", 75),
            ("duplicates", "DUP", 55), ("malformed", "BAD", 55),
            ("enabled", "ACTIVE", 65),
        )
        for key, title, width in columns:
            self.dataset_tree.heading(key, text=title)
            self.dataset_tree.column(key, width=width)
        self.dataset_tree.pack(fill="both", expand=True, padx=15, pady=(0, 12))
        self.refresh_datasets()

    def _save_dataset_entry(self) -> None:
        input_text = self.dataset_fields["USER / INPUT"].get("1.0", "end-1c").strip()
        output_text = self.dataset_fields["JARVIS / OUTPUT"].get("1.0", "end-1c").strip()
        tool = self.dataset_fields["TOOL"].get("1.0", "end-1c").strip()
        arguments_text = self.dataset_fields["ARGUMENTS (JSON)"].get("1.0", "end-1c").strip()
        if not input_text or not (output_text or tool):
            messagebox.showerror("Dataset", "INPUT and OUTPUT or TOOL are required.", parent=self.root)
            return
        try:
            arguments = json.loads(arguments_text) if arguments_text else {}
        except json.JSONDecodeError as exc:
            messagebox.showerror("Dataset", f"Arguments must be valid JSON: {exc}", parent=self.root)
            return
        path = self.project_root / "datasets" / "raw" / "user_entries_v009.jsonl"
        row = {
            "id": f"user-{int(time.time() * 1000)}",
            "dataset_version": "dataset_v009",
            "parent_dataset": "dataset_v009",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "origin": "user-authored",
            "pretrained_source": None,
            "category": self.dataset_type.get().casefold(),
            "language": self.dataset_language.get(),
            "input": input_text, "output": output_text,
            "action": tool, "arguments": arguments,
            "status": "reviewed",
            "metadata": {
                "source": "user-authored", "quality": "reviewed",
                "task_type": self.dataset_type.get().casefold(),
                "risk_level": "L0" if not tool else "L2",
                "requires_tools": bool(tool), "expected_tool": tool,
                "permission_required": bool(tool),
            },
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        content = path.read_bytes()
        manifest = {
            "format": "jarvis-user-dataset-v2", "version": "dataset_v009",
            "parent": "dataset_v009", "entries": len(path.read_text(encoding="utf-8").splitlines()),
            "sha256": hashlib.sha256(content).hexdigest(), "pretrained_source": None,
        }
        (self.project_root / "datasets" / "manifest_user_v009.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        self.dataset_status.configure(text=f"{row['id']} در dataset_v009 ذخیره شد؛ پیش از آموزش دوباره Tokenize کنید.")
        self.refresh_datasets()

    def refresh_datasets(self) -> None:
        if not hasattr(self, "dataset_tree"):
            return
        for item in self.dataset_tree.get_children():
            self.dataset_tree.delete(item)
        try:
            records = self.dataset_catalog.scan()
        except (OSError, ValueError, UnicodeError) as exc:
            self.dataset_status.configure(text=f"خطای بررسی دیتاست: {exc}", fg=self.RED)
            return
        for index, record in enumerate(records):
            self.dataset_tree.insert(
                "", "end", iid=f"dataset-{index}", values=(
                    record.path, record.samples, ", ".join(record.categories[:3]),
                    ", ".join(record.languages), record.duplicates, record.malformed,
                    "YES" if record.enabled else "NO",
                ),
            )
        bad = sum(record.malformed for record in records)
        duplicates = sum(record.duplicates for record in records)
        samples = sum(record.samples for record in records if record.enabled)
        self.dataset_status.configure(
            text=f"{len(records)} فایل | {samples:,} نمونه فعال | duplicate={duplicates} | malformed={bad}",
            fg=self.GREEN if bad == 0 else self.AMBER,
        )

    def _selected_dataset(self) -> str:
        selected = self.dataset_tree.selection()
        return str(self.dataset_tree.item(selected[0], "values")[0]) if selected else ""

    def import_dataset(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root, title="افزودن دیتاست JSONL", filetypes=(("JSON Lines", "*.jsonl"),)
        )
        if not selected:
            return
        try:
            record = self.dataset_catalog.import_jsonl(Path(selected))
            self.dataset_status.configure(text=f"افزوده شد: {record.path}", fg=self.GREEN)
            self.refresh_datasets()
        except (OSError, ValueError, UnicodeError) as exc:
            messagebox.showerror("Dataset", str(exc), parent=self.root)

    def toggle_dataset(self) -> None:
        relative = self._selected_dataset()
        if not relative:
            return
        values = self.dataset_tree.item(self.dataset_tree.selection()[0], "values")
        try:
            self.dataset_catalog.set_enabled(relative, str(values[-1]) != "YES")
            self.refresh_datasets()
        except (OSError, ValueError) as exc:
            messagebox.showerror("Dataset", str(exc), parent=self.root)

    def delete_dataset(self) -> None:
        relative = self._selected_dataset()
        if not relative or not messagebox.askyesno(
            "حذف دیتاست", f"فقط فایل Import شده حذف می‌شود:\n{relative}", parent=self.root,
        ):
            return
        try:
            self.dataset_catalog.delete_imported(relative)
            self.refresh_datasets()
        except (OSError, ValueError, PermissionError) as exc:
            messagebox.showerror("Dataset", str(exc), parent=self.root)

    def _build_train(self) -> None:
        tab = self.tabs["TRAIN"]
        panel = self._panel(tab)
        panel.pack(fill="both", expand=True, pady=9)
        tk.Label(panel, text="TRAINING CONTROL", bg=self.PANEL, fg=self.CYAN, font=("Consolas", 11, "bold")).pack(anchor="w", padx=15, pady=(14, 10))
        controls = tk.Frame(panel, bg=self.PANEL)
        controls.pack(fill="x", padx=15)
        self.profile_var = tk.StringVar(value="nano")
        self.steps_var = tk.StringVar(value="8")
        self.sequence_var = tk.StringVar(value=str(self._suggest_configuration()["sequence_length"]))
        self.resume_var = tk.BooleanVar(value=False)
        for label, variable, values in (
            ("PROFILE", self.profile_var, ("nano", "core", "pro")),
            ("STEPS / STAGE", self.steps_var, ("1", "4", "8", "32", "128", "256")),
            ("SEQUENCE", self.sequence_var, ("16", "32", "64", "128")),
        ):
            tk.Label(controls, text=label, bg=self.PANEL, fg=self.MUTED, font=("Consolas", 8)).pack(side="left", padx=(0, 5))
            ttk.Combobox(controls, textvariable=variable, values=values, width=10, style="Lab.TCombobox").pack(side="left", padx=(0, 15))
        tk.Checkbutton(
            controls, text="TRAIN FROM LATEST CHECKPOINT", variable=self.resume_var,
            bg=self.PANEL, fg=self.TEXT, selectcolor=self.PANEL_2, activebackground=self.PANEL,
            activeforeground=self.TEXT, font=("Consolas", 8),
        ).pack(side="left")
        buttons = tk.Frame(panel, bg=self.PANEL)
        buttons.pack(fill="x", padx=15, pady=14)
        for label, command, color in (
            ("START", self.start_training, self.BLUE), ("PAUSE", self.pause_training, None),
            ("RESUME", self.resume_training, None), ("STOP", self.stop_training, self.RED),
            ("SAVE CHECKPOINT", self.save_checkpoint, None), ("VALIDATE", self.validate_model, None),
        ):
            self._button(buttons, label, command, color=color).pack(side="left", padx=(0, 7))
        self.training_output = self._text(panel, 22)
        self.training_output.configure(font=("Consolas", 8))
        self.training_output.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    def _build_evaluate(self) -> None:
        tab = self.tabs["EVALUATE"]
        self.evaluate_output = self._text(tab, 28)
        self.evaluate_output.pack(fill="both", expand=True, pady=(50, 9))
        self._button(tab, "ارزیابی Next-token", self.validate_model, color=self.BLUE).place(x=0, y=9)
        self._button(tab, "۱۰۰۰ تست Unseen Agent", self.validate_agent).place(x=170, y=9)

    def _build_test(self) -> None:
        tab = self.tabs["TEST"]
        self.test_output = self._text(tab, 28)
        self.test_output.pack(fill="both", expand=True, pady=(50, 9))
        self._button(tab, "RUN UNIT + INTEGRATION + SCENARIOS", self.run_tests, color=self.BLUE).place(x=0, y=9)

    def _build_models(self) -> None:
        tab = self.tabs["MODELS"]
        self.models_output = self._text(tab, 28)
        self.models_output.pack(fill="both", expand=True, pady=(92, 9))
        self.model_var = tk.StringVar(value="")
        self.model_combo = ttk.Combobox(
            tab, textvariable=self.model_var, state="readonly", width=50, style="Lab.TCombobox",
        )
        self.model_combo.place(x=0, y=10)
        model_actions = (
            ("ایجاد", self.prepare_new_model, self.BLUE),
            ("بازخوانی", self.refresh_models, None),
            ("ادامه آموزش", self.resume_selected_model, None),
            ("ارزیابی", self.evaluate_selected_model, self.BLUE),
            ("فعال‌سازی", self.activate_selected_model, self.GREEN),
            ("پیش‌فرض", self.reset_active_model, None),
            ("Import", self.import_model, None),
            ("حذف", self.delete_selected_model, self.RED),
        )
        x = 0
        for label, command, color in model_actions:
            button = self._button(tab, label, command, color=color)
            button.place(x=x, y=49)
            x += 105
        self._button(tab, "BENCHMARK", self.benchmark_model).place(x=x, y=49)
        self._button(tab, "EXPORT", lambda: self.export_model("torch")).place(x=x + 120, y=49)
        self.refresh_models()

    def _build_corrections(self) -> None:
        tab = self.tabs["CORRECTIONS"]
        self.correction_tree = ttk.Treeview(
            tab, columns=("input", "correction", "status"), show="headings", style="Lab.Treeview"
        )
        for key, title, width in (("input", "ORIGINAL INPUT", 300), ("correction", "CORRECT RESPONSE / ACTION", 480), ("status", "STATUS", 100)):
            self.correction_tree.heading(key, text=title)
            self.correction_tree.column(key, width=width)
        self.correction_tree.pack(fill="both", expand=True, pady=(50, 9))
        self._button(tab, "REFRESH", self.refresh_corrections).place(x=0, y=9)
        self._button(tab, "APPROVE", lambda: self._review_correction("approved"), color=self.BLUE).place(x=90, y=9)
        self._button(tab, "REJECT", lambda: self._review_correction("rejected"), color=self.RED).place(x=190, y=9)
        self.refresh_corrections()

    def _build_failures(self) -> None:
        tab = self.tabs["FAILURES"]
        self.failure_tree = ttk.Treeview(
            tab, columns=("input", "prediction", "code", "status"),
            show="headings", style="Lab.Treeview",
        )
        for key, title, width in (
            ("input", "INPUT", 300),
            ("prediction", "PREDICTED TOOL / ACTION", 280),
            ("code", "FAILURE", 190),
            ("status", "STATUS", 90),
        ):
            self.failure_tree.heading(key, text=title)
            self.failure_tree.column(key, width=width)
        self.failure_tree.pack(fill="both", expand=True, pady=(50, 9))
        self._button(tab, "REFRESH", self.refresh_failures).place(x=0, y=9)
        self._button(
            tab, "APPROVE + CORRECT", lambda: self._review_failure("approved"),
            color=self.BLUE,
        ).place(x=90, y=9)
        self._button(
            tab, "REJECT", lambda: self._review_failure("rejected"), color=self.RED,
        ).place(x=255, y=9)
        self._button(
            tab, "RESOLVED", lambda: self._review_failure("resolved"),
        ).place(x=350, y=9)
        self.refresh_failures()

    def _build_logs(self) -> None:
        tab = self.tabs["LOGS"]
        self.log_name = tk.StringVar(value="training")
        names = ("training", "app.log", "errors.log", "chat.jsonl", "agent.jsonl", "tools.jsonl", "search.jsonl", "performance.jsonl")
        ttk.Combobox(tab, textvariable=self.log_name, values=names, state="readonly", style="Lab.TCombobox", width=24).place(x=0, y=12)
        self._button(tab, "LOAD", self.load_log).place(x=210, y=9)
        self.log_output = self._text(tab, 28)
        self.log_output.configure(font=("Consolas", 8))
        self.log_output.pack(fill="both", expand=True, pady=(50, 9))

    def _append_output(self, widget: tk.Text, value: str, *, replace: bool = False) -> None:
        widget.configure(state="normal")
        if replace:
            widget.delete("1.0", "end")
        widget.insert("end", value.rstrip() + "\n")
        widget.see("end")

    def start_training(self) -> None:
        if self.training_thread and self.training_thread.is_alive() or self.training_process and self.training_process.poll() is None:
            messagebox.showinfo("Training", "A training run is already active.", parent=self.root)
            return
        try:
            steps = max(1, int(self.steps_var.get()))
            sequence = max(8, int(self.sequence_var.get()))
        except ValueError:
            messagebox.showerror("Training", "Steps and sequence must be integers.", parent=self.root)
            return
        profile = self.profile_var.get()
        try:
            from training.tokenize_dataset_v18 import tokenize as tokenize_active_dataset
            token_report = tokenize_active_dataset()
            self._append_output(
                self.training_output,
                f"Prepared {token_report['dataset_version']} with {token_report['training_examples_after_lab_merge']} tokenized training rows.",
            )
        except Exception as exc:
            messagebox.showerror("Training dataset", f"Could not prepare dataset_v009: {exc}", parent=self.root)
            self.lab_status.configure(text="● READY", fg=self.GREEN)
            return
        self.metric_vars["model"].set(f"Jarvis {profile.title()}")
        self.metric_vars["remaining"].set(str(steps * 14))
        self.lab_status.configure(text="● TRAINING", fg=self.AMBER)
        self._append_output(self.training_output, f"Starting real {profile} training …", replace=True)
        if profile == "nano":
            self.training_control = TrainingControl()
            self.training_thread = threading.Thread(
                target=self._run_numpy_training,
                args=(steps, sequence, self.resume_var.get()), daemon=True,
                name="jarvis-lab-numpy-training",
            )
            self.training_thread.start()
            return
        if importlib.util.find_spec("torch") is None:
            messagebox.showerror(
                "Training dependency",
                "Core/Pro needs PyTorch. Run setup.ps1 -Training. No model will be downloaded.",
                parent=self.root,
            )
            self.lab_status.configure(text="● READY", fg=self.GREEN)
            return
        self.control_file.parent.mkdir(parents=True, exist_ok=True)
        self._write_torch_control(pause=False, stop=False, save=False)
        command = [
            sys.executable, str(self.project_root / "training" / "torch_train.py"),
            "--profile", profile, "--steps-per-stage", str(steps),
            "--sequence-length", str(sequence), "--control-file", str(self.control_file),
            "--dataset-version", "dataset_v009",
        ]
        if self.resume_var.get():
            candidates = sorted((self.project_root / "models" / "checkpoints").glob("torch_stage*/checkpoint.pt"))
            if candidates:
                command.extend(("--resume", str(candidates[-1])))
        self.training_thread = threading.Thread(target=self._run_torch_process, args=(command,), daemon=True)
        self.training_thread.start()

    def _run_numpy_training(self, steps: int, sequence: int, resume: bool) -> None:
        try:
            config = TransformerConfig.load(self.project_root / "configs" / "nano_v7.json")
            latest = self.project_root / "models" / "jarvis_nano_v18.npz"
            model = JarvisTransformer.load(latest, dequantize=True) if resume and latest.is_file() else JarvisTransformer.initialize(config)
            runner = CurriculumRunner(
                self.project_root, model, steps_per_stage=steps,
                sequence_length=sequence, control=self.training_control,
                log_callback=self.events.put, retain_full_checkpoints=1,
                dataset_version="dataset_v009",
                tokenized_file=self.project_root / "datasets" / "tokenized" / "dataset_v009_train.jsonl",
                checkpoint_root=self.project_root / "models" / "checkpoints_v7_lab",
                metrics_filename="jarvis_lab_training_metrics_v7.json",
                save_stage_weights=False,
            )
            metrics = runner.run()
            if not metrics.get("stopped"):
                source = self.project_root / "models" / "checkpoints_v7_lab" / "stage14_jarvis_personality" / "model.npz"
                runs = self.project_root / "models" / "lab_runs"
                runs.mkdir(parents=True, exist_ok=True)
                destination = runs / f"jarvis_nano_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.npz"
                shutil.copy2(source, destination)
                self.events.put({"event": "lab_model_saved", "path": str(destination.relative_to(self.project_root))})
        except Exception as exc:
            self.events.put({"event": "error", "message": str(exc)})

    def _run_torch_process(self, command: list[str]) -> None:
        try:
            self.training_process = subprocess.Popen(
                command, cwd=self.project_root, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            assert self.training_process.stdout is not None
            for line in self.training_process.stdout:
                try:
                    self.events.put(json.loads(line))
                except json.JSONDecodeError:
                    self.events.put({"event": "text", "message": line.rstrip()})
            code = self.training_process.wait()
            self.events.put({"event": "process_complete", "returncode": code})
        except Exception as exc:
            self.events.put({"event": "error", "message": str(exc)})

    def _write_torch_control(self, **updates: bool) -> None:
        current = {"pause": False, "stop": False, "save": False}
        if self.control_file.is_file():
            try:
                current.update(json.loads(self.control_file.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
        current.update(updates)
        temporary = self.control_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(current), encoding="utf-8")
        temporary.replace(self.control_file)

    def pause_training(self) -> None:
        if self.training_control:
            self.training_control.pause.set()
        self._write_torch_control(pause=True)
        self.lab_status.configure(text="● PAUSED", fg=self.AMBER)

    def resume_training(self) -> None:
        if self.training_control:
            self.training_control.pause.clear()
        self._write_torch_control(pause=False)
        self.lab_status.configure(text="● TRAINING", fg=self.AMBER)

    def stop_training(self) -> None:
        if self.training_control:
            self.training_control.stop.set()
        self._write_torch_control(stop=True, pause=False)
        self._append_output(self.training_output, "Stop requested; the current optimizer step will finish safely.")

    def save_checkpoint(self) -> None:
        if self.training_control:
            self.training_control.save_checkpoint.set()
        self._write_torch_control(save=True)
        self._append_output(self.training_output, "Checkpoint requested.")

    def validate_model(self) -> None:
        command = [sys.executable, str(self.project_root / "training" / "evaluate_model.py"), "--maximum-cases", "128"]
        threading.Thread(target=self._run_capture, args=(command, self.evaluate_output), daemon=True).start()

    def validate_agent(self) -> None:
        command = [sys.executable, str(self.project_root / "training" / "evaluate_agent_v7.py")]
        threading.Thread(target=self._run_capture, args=(command, self.evaluate_output), daemon=True).start()

    def run_tests(self) -> None:
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
        threading.Thread(target=self._run_capture, args=(command, self.test_output), daemon=True).start()

    def benchmark_model(self) -> None:
        command = [sys.executable, str(self.project_root / "training" / "benchmark_model.py")]
        selected = self.model_records.get(self.model_var.get())
        if selected and selected.path:
            command.extend(("--model", str(self.project_root / selected.path)))
        threading.Thread(target=self._run_capture, args=(command, self.models_output), daemon=True).start()

    def export_model(self, kind: str) -> None:
        command = [
            sys.executable, str(self.project_root / "training" / "export_model.py"),
            "--format", kind, "--precision", "fp32",
        ]
        selected = self.model_records.get(self.model_var.get())
        if selected and selected.path:
            command.extend(("--model", str(self.project_root / selected.path)))
        threading.Thread(target=self._run_capture, args=(command, self.models_output), daemon=True).start()

    def evaluate_selected_model(self) -> None:
        selected = self.model_records.get(self.model_var.get())
        if selected is None or not selected.path:
            messagebox.showerror("Model", "یک مدل آموزش‌دیده را انتخاب کنید.", parent=self.root)
            return
        command = [
            sys.executable, str(self.project_root / "training" / "evaluate_model.py"),
            "--model", str(self.project_root / selected.path), "--maximum-cases", "411",
        ]
        threading.Thread(target=self._run_capture, args=(command, self.models_output), daemon=True).start()

    def resume_selected_model(self) -> None:
        selected = self.model_records.get(self.model_var.get())
        if selected is None:
            return
        if selected.profile not in {"nano", "core", "pro"}:
            messagebox.showinfo(
                "Resume", "برای مدل Import شده ابتدا آن را فعال کنید؛ Resume فقط از checkpointهای LAB انجام می‌شود.",
                parent=self.root,
            )
            return
        self.profile_var.set(selected.profile)
        self.resume_var.set(True)
        self.notebook.select(self.tabs["TRAIN"])

    def prepare_new_model(self) -> None:
        selected = self.model_records.get(self.model_var.get())
        profile = selected.profile if selected and selected.profile in {"nano", "core", "pro"} else "nano"
        self.profile_var.set(profile)
        self.resume_var.set(False)
        self._append_output(
            self.training_output,
            f"NEW {profile.upper()} RUN: وزن‌ها هنگام START از مقداردهی تصادفی ساخته می‌شوند.",
            replace=True,
        )
        self.notebook.select(self.tabs["TRAIN"])

    def activate_selected_model(self) -> None:
        selected = self.model_records.get(self.model_var.get())
        if selected is None or not selected.activatable or not selected.path:
            messagebox.showerror("Model", "این پروفایل وزن معتبر و قابل‌فعال‌سازی ندارد.", parent=self.root)
            return
        if not messagebox.askyesno(
            "فعال‌سازی مدل", "مدل پس از راه‌اندازی دوباره JARVIS فعال می‌شود. ادامه؟", parent=self.root,
        ):
            return
        try:
            self.model_catalog.activate(selected.path)
            self._append_output(self.models_output, f"ACTIVE AFTER RESTART: {selected.path}")
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Model", str(exc), parent=self.root)

    def reset_active_model(self) -> None:
        self.model_catalog.reset_activation()
        self._append_output(self.models_output, "ACTIVE AFTER RESTART: bundled Jarvis Nano v0.8")

    def import_model(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root, title="Import JARVIS model", filetypes=(("JARVIS NPZ", "*.npz"),)
        )
        if not path:
            return
        try:
            record = self.model_catalog.import_model(
                Path(path), self.project_root / "models" / "jarvis_tokenizer_v003.json"
            )
            self.refresh_models()
            self.model_var.set(record.key)
        except (OSError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Model", str(exc), parent=self.root)

    def delete_selected_model(self) -> None:
        selected = self.model_records.get(self.model_var.get())
        if selected is None or not selected.path:
            return
        if not messagebox.askyesno(
            "حذف مدل", "فقط مدل‌های ساخته/Import شده در LAB قابل حذف‌اند. ادامه؟", parent=self.root,
        ):
            return
        try:
            self.model_catalog.delete_lab_model(selected.path)
            self.refresh_models()
        except (OSError, ValueError, PermissionError) as exc:
            messagebox.showerror("Model", str(exc), parent=self.root)

    def _run_capture(self, command: list[str], widget: tk.Text) -> None:
        self.events.put({"event": "capture_start", "widget": widget, "command": command})
        completed = subprocess.run(
            command, cwd=self.project_root, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.events.put({
            "event": "capture_complete", "widget": widget,
            "text": (completed.stdout + "\n" + completed.stderr).strip(),
            "returncode": completed.returncode,
        })

    def refresh_models(self) -> None:
        records = self.model_catalog.scan()
        self.model_records = {record.key: record for record in records}
        keys = tuple(self.model_records)
        self.model_combo.configure(values=keys)
        if self.model_var.get() not in self.model_records:
            self.model_var.set(keys[0] if keys else "")
        active = self.project_root / "models" / "active_model_v7.json"
        payload = {
            "models": [
                {
                    "id": record.key, "profile": record.profile, "path": record.path,
                    "parameters": record.parameters, "status": record.status,
                    "activatable": record.activatable,
                }
                for record in records
            ],
            "active_override": (
                json.loads(active.read_text(encoding="utf-8")) if active.is_file() else None
            ),
        }
        self._append_output(
            self.models_output, json.dumps(payload, ensure_ascii=False, indent=2), replace=True
        )

    def refresh_corrections(self) -> None:
        for item in self.correction_tree.get_children():
            self.correction_tree.delete(item)
        for row in self.learning_queue.rows():
            corrected = str(row.get("correct_response") or row.get("correct_action") or "")
            self.correction_tree.insert(
                "", "end", iid=str(row["id"]),
                values=(str(row.get("original_input", ""))[:100], corrected[:160], row.get("status", "pending")),
            )

    def _review_correction(self, status: str) -> None:
        selected = self.correction_tree.selection()
        if not selected:
            return
        for row_id in selected:
            self.learning_queue.set_status(row_id, status)
        self.refresh_corrections()

    def refresh_failures(self) -> None:
        for item in self.failure_tree.get_children():
            self.failure_tree.delete(item)
        for row in self.failure_collector.rows():
            prediction = row.get("prediction", {})
            label = f"{prediction.get('tool', '')} / {prediction.get('action', '')}".strip(" / ")
            self.failure_tree.insert(
                "", "end", iid=str(row["id"]),
                values=(
                    str(row.get("input", ""))[:120], label[:120],
                    str(row.get("failure_code", ""))[:100],
                    row.get("status", "pending"),
                ),
            )

    def _review_failure(self, status: str) -> None:
        selected = self.failure_tree.selection()
        if not selected:
            return
        correct_action = ""
        correct_response = ""
        if status == "approved":
            correct_action = simpledialog.askstring(
                "Correct action",
                "Correct tool/action (leave empty for a response correction):",
                parent=self.root,
            ) or ""
            if not correct_action:
                correct_response = simpledialog.askstring(
                    "Correct response", "Reviewed correct response:", parent=self.root,
                ) or ""
            if not (correct_action or correct_response):
                return
        for row_id in selected:
            self.failure_collector.review(
                row_id, status, correct_action=correct_action,
                correct_response=correct_response,
            )
        self.refresh_failures()

    def load_log(self) -> None:
        name = self.log_name.get()
        if name == "training":
            paths = sorted((self.project_root / "logs" / "training").glob("*"))
            path = paths[-1] if paths else None
        else:
            path = self.project_root / "logs" / name
        if path is None or not path.is_file():
            content = "No log is available yet."
        else:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            content = "\n".join(lines[-1000:])
        self._append_output(self.log_output, content, replace=True)

    def _poll_events(self) -> None:
        if self._closing:
            return
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = str(event.get("event", ""))
            if kind == "step":
                self.current_training_event = event
                step = int(event.get("step", 0))
                stage = int(event.get("stage", 0))
                total = max(1, int(self.steps_var.get() or 1) * 14)
                self.metric_vars["epoch"].set(f"{stage} / 14")
                self.metric_vars["step"].set(str(step))
                self.metric_vars["progress"].set(f"{min(100.0, step / total * 100):.1f}%")
                self.metric_vars["train_loss"].set(str(event.get("loss", "—")))
                self.metric_vars["learning_rate"].set(f"{float(event.get('learning_rate', 0)):.2e}")
                self.metric_vars["tokens_per_second"].set(str(event.get("tokens_per_second", "—")))
                self.metric_vars["remaining"].set(str(max(0, total - step)))
                self.chart.add(
                    loss=float(event.get("loss", 0)),
                    learning_rate=float(event.get("learning_rate", 0)),
                    tokens_per_second=float(event.get("tokens_per_second", 0)),
                )
                self._append_output(self.training_output, json.dumps(event, ensure_ascii=False))
            elif kind == "training_start":
                self.metric_vars["parameters"].set(f"{int(event.get('parameters', 0)):,}")
                self._append_output(self.training_output, json.dumps(event, ensure_ascii=False))
            elif kind == "stage_complete":
                validation = event.get("validation_loss", event.get("mean_loss", "—"))
                self.metric_vars["validation_loss"].set(str(validation))
                self.metric_vars["checkpoint"].set(str(event.get("checkpoint", "—")))
                if isinstance(validation, (int, float)):
                    self.chart.add(validation_loss=float(validation))
                self._append_output(self.training_output, json.dumps(event, ensure_ascii=False))
            elif kind in {"training_complete", "process_complete", "training_stopped", "lab_model_saved"}:
                self._append_output(self.training_output, json.dumps(event, ensure_ascii=False))
                if kind != "lab_model_saved":
                    self.lab_status.configure(text="● READY", fg=self.GREEN)
            elif kind == "capture_start":
                widget = event["widget"]
                self._append_output(widget, "RUNNING: " + " ".join(event["command"]), replace=True)
            elif kind == "capture_complete":
                self._append_output(event["widget"], str(event.get("text", "")))
            elif kind == "error":
                self.lab_status.configure(text="● ERROR", fg=self.RED)
                self._append_output(self.training_output, "ERROR: " + str(event.get("message", "")))
            elif kind == "text":
                self._append_output(self.training_output, str(event.get("message", "")))
            self.events.task_done()
        self.root.after(100, self._poll_events)

    def _sample_hardware(self) -> None:
        if self._closing:
            return
        try:
            value = self.live_monitor.sample()
            gpu = (
                f"{value.gpu_percent or 0:.1f}% | VRAM {value.vram_used_mb or 0:.0f}/{value.vram_total_mb or 0:.0f} MB | {value.gpu_temperature_c or 0:.0f}°C"
                if value.gpu_available else "NOT AVAILABLE — CPU FALLBACK"
            )
            self.hardware_label.configure(
                text=(
                    f"CPU       {self.hardware.info.cpu}\n"
                    f"CORES     {self.hardware.info.physical_cores} physical / {self.hardware.info.logical_cores} logical\n"
                    f"RAM       {value.ram_used_mb / 1024:.1f}/{value.ram_total_mb / 1024:.1f} GB ({value.ram_percent:.1f}%)\n"
                    f"GPU       {gpu}\n"
                    f"CUDA      {'AVAILABLE' if value.gpu_available and importlib.util.find_spec('torch') else 'NOT DETECTED'}\n"
                    f"DISK      {value.disk_free_gb:.1f} GB free\n"
                    f"JARVIS    CPU {value.app_cpu_percent:.1f}% | RAM {value.app_ram_mb:.1f} MB"
                )
            )
        except Exception as exc:
            self.hardware_label.configure(text=f"Telemetry unavailable: {exc}")
        self.root.after(1000, self._sample_hardware)

    def close(self) -> None:
        active = bool(
            (self.training_thread and self.training_thread.is_alive())
            or (self.training_process and self.training_process.poll() is None)
        )
        if active and not messagebox.askyesno(
            "Close JARVIS LAB", "Training is active. Request a safe stop and close the Lab?",
            parent=self.root,
        ):
            return
        if active:
            self.stop_training()
        self._closing = True
        self.root.destroy()
