from __future__ import annotations

import logging
import queue
import re
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import json
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from jarvis.agent.core import AgentReply
from jarvis.gui.chat_surface import ModernChatSurface
from jarvis.gui.orb import HolographicCore
from jarvis.gui.monitor import MonitorPanel
from jarvis.gui.palette import CommandPalette
from jarvis.gui.settings import SettingsWindow
from jarvis.gui.visual_fx import NeuralBackdrop
from jarvis.learning.queue import LearningQueue
from jarvis.monitoring.live import LiveMonitor
from jarvis.runtime.bootstrap import RuntimeContext
from jarvis.utils.text import detect_language


class JarvisGUI:
    """Single-window, lightweight desktop shell for the Jarvis agent."""

    BG = "#070B12"
    PANEL = "#0D1420"
    PANEL_2 = "#111C2A"
    INPUT = "#121E2D"
    BORDER = "#20344A"
    TEXT = "#E7F2FF"
    MUTED = "#6F849B"
    CYAN = "#38D9FF"
    BLUE = "#426BFF"
    GREEN = "#4FE0A0"
    AMBER = "#FFD166"
    RED = "#FF6B7A"
    USER_BG = "#17345B"
    ASSISTANT_BG = "#101D2B"

    def __init__(self, runtime: RuntimeContext, auto_close_ms: int | None = None) -> None:
        self.runtime = runtime
        self.agent = runtime.agent
        self.logger = logging.getLogger("jarvis.gui")
        self.root = tk.Tk()
        self.root.title(f"JARVIS v{runtime.config.version} — Smart Brain & Dynamic Agent")
        self.root.configure(bg=self.BG)
        self.root.minsize(runtime.config.window.min_width, runtime.config.window.min_height)
        self._center_window(runtime.config.window.width, runtime.config.window.height)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.colors = {
            "bg": self.BG,
            "panel": self.PANEL,
            "input": self.INPUT,
            "border": self.BORDER,
            "text": self.TEXT,
            "muted": self.MUTED,
            "cyan": self.CYAN,
            "blue": self.BLUE,
        }
        self._tasks: queue.Queue[tuple[str, Any] | None] = queue.Queue()
        self._results: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.agent.set_status_callback(
            lambda status: self._results.put(("status", status))
        )
        self._closing = False
        self._settings_window: SettingsWindow | None = None
        self._context_menu: tk.Menu | None = None
        self._input_context_menu: tk.Menu | None = None
        self.chat_surface: ModernChatSurface | None = None
        self._lab_window: tk.Toplevel | None = None
        self._last_user_text = ""
        self._last_reply_text = ""
        self._last_plan: dict[str, Any] | None = None
        self._stop_stream_requested = False
        self._stream_after_id: str | None = None
        self._chat_busy = False
        self._chat_messages: list[tuple[str, str]] = []
        self.copy_chat_button: Any = None
        self.live_monitor = LiveMonitor(runtime.hardware)
        self.learning_queue = LearningQueue(
            runtime.config.paths.data_dir / "learning_queue.jsonl",
            runtime.config.learning.max_corrections,
        )

        saved_performance = self.agent.get_setting("performance", "AUTO").upper()
        self.performance_var = tk.StringVar(value=saved_performance)
        self.personality_var = tk.StringVar(value=self.agent.current_personality)
        self.dry_run_var = tk.BooleanVar(value=False)
        self.copy_count_var = tk.StringVar(value="همه پیام‌ها")
        self._active_performance = self._resolve_performance(saved_performance)
        self.ui_font, self.mono_font = self._resolve_fonts()

        self._configure_styles()
        self._build_layout()
        self._bind_shortcuts()
        self._restore_history()

        self._worker = threading.Thread(
            target=self._worker_loop, name="jarvis-agent-worker", daemon=True
        )
        self._worker.start()
        self.root.after(65, self._poll_results)
        self.root.after(800, self._update_metrics)
        self.root.after(100, self.message_input.focus_set)

        for warning in runtime.startup_warnings:
            self._append_system(f"⚠ {warning}", error=True)
        if not self.agent.recent_history():
            self._append_assistant(
                f"سلام! من JARVIS v{runtime.config.version} هستم؛ دستیار هوشمند محلی، توسعه‌یافته توسط یونس گوهری. چه کاری انجام بدیم؟",
                "fa",
            )
        self._set_status("READY")
        if auto_close_ms:
            self.root.after(max(250, auto_close_ms), self.close)

    def _center_window(self, width: int, height: int) -> None:
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _resolve_fonts(self) -> tuple[str, str]:
        available = {name.casefold(): name for name in tkfont.families(self.root)}
        ui = next(
            (available[name.casefold()] for name in ("Vazirmatn", "Segoe UI", "Tahoma", "Arial") if name.casefold() in available),
            "TkDefaultFont",
        )
        mono = next(
            (available[name.casefold()] for name in ("Cascadia Mono", "Consolas", "Courier New") if name.casefold() in available),
            "TkFixedFont",
        )
        return ui, mono

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Jarvis.TCombobox",
            fieldbackground=self.PANEL_2,
            background=self.PANEL_2,
            foreground=self.TEXT,
            arrowcolor=self.CYAN,
            bordercolor=self.BORDER,
            lightcolor=self.BORDER,
            darkcolor=self.BORDER,
            padding=6,
        )
        style.map(
            "Jarvis.TCombobox",
            fieldbackground=[("readonly", self.PANEL_2)],
            foreground=[("readonly", self.TEXT)],
            selectbackground=[("readonly", self.PANEL_2)],
            selectforeground=[("readonly", self.TEXT)],
        )
        style.configure(
            "Jarvis.Vertical.TScrollbar",
            background=self.PANEL_2,
            troughcolor=self.PANEL,
            bordercolor=self.PANEL,
            arrowcolor=self.MUTED,
        )

    def _build_layout(self) -> None:
        """Build the JARVIS v18 command deck: layered HUD + holographic conversation bay."""
        self.backdrop = NeuralBackdrop(self.root, bg="#02050A", accent=self.CYAN)
        self.backdrop.place(x=0, y=0, relwidth=1, relheight=1)

        shell = tk.Frame(self.root, bg="#030810")
        shell.pack(fill="both", expand=True, padx=13, pady=11)
        shell.rowconfigure(1, weight=1)
        shell.columnconfigure(0, weight=1)

        self._build_header(shell)
        body = tk.Frame(shell, bg="#030810")
        body.grid(row=1, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        self._build_chat_panel(body)
        self._build_monitor_panel(body)
        self._build_core_panel(body)
        self._build_footer(shell)

    def _build_header(self, parent: tk.Frame) -> None:
        import customtkinter as ctk

        header = ctk.CTkFrame(
            parent, fg_color="#071019", corner_radius=20,
            border_width=1, border_color="#143240", height=64,
        )
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.grid_columnconfigure(1, weight=1)
        header.grid_propagate(False)

        sigil = ctk.CTkFrame(
            header, width=45, height=45, corner_radius=16,
            fg_color="#092933", border_width=1, border_color="#1A6675",
        )
        sigil.grid(row=0, column=0, padx=(10, 10), pady=9)
        sigil.grid_propagate(False)
        ctk.CTkLabel(
            sigil, text="J", text_color="#75F6FF",
            font=(self.ui_font, 22, "bold"),
        ).place(relx=.5, rely=.5, anchor="center")

        brand = ctk.CTkFrame(header, fg_color="transparent")
        brand.grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(
            brand, text="JARVIS // NEURAL COMMAND DECK",
            text_color="#E9FCFF", font=(self.mono_font, 12, "bold"), anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            brand,
            text=f"LOCAL INTELLIGENCE v{self.runtime.config.version}  •  developed by Younes Gohari",
            text_color="#527485", font=(self.mono_font, 7), anchor="w",
        ).pack(anchor="w")

        controls = ctk.CTkFrame(header, fg_color="transparent")
        controls.grid(row=0, column=2, sticky="e", padx=10)

        self.personality_box = ctk.CTkOptionMenu(
            controls, variable=self.personality_var,
            values=self.runtime.personalities.available_names(),
            command=lambda _value: self._personality_changed(),
            width=118, height=32, corner_radius=12,
            fg_color="#0C1C29", button_color="#123343", button_hover_color="#17495B",
            text_color=self.TEXT, dropdown_fg_color="#091722",
            dropdown_hover_color="#123343", font=(self.ui_font, 8, "bold"),
        )
        self.personality_box.pack(side="left", padx=(0, 6))

        self.performance_box = ctk.CTkOptionMenu(
            controls, variable=self.performance_var,
            values=("AUTO", "LOW", "BALANCED", "ULTRA"),
            command=lambda _value: self._performance_changed(),
            width=108, height=32, corner_radius=12,
            fg_color="#0C1C29", button_color="#123343", button_hover_color="#17495B",
            text_color=self.TEXT, dropdown_fg_color="#091722",
            dropdown_hover_color="#123343", font=(self.mono_font, 8, "bold"),
        )
        self.performance_box.pack(side="left", padx=(0, 6))

        self.settings_button = ctk.CTkButton(
            controls, text="⚙", command=self.open_settings,
            width=34, height=32, corner_radius=12,
            fg_color="#0C1C29", hover_color="#17495B",
            border_width=1, border_color="#173A49", text_color="#BEEFF5",
            font=(self.ui_font, 12),
        )
        self.settings_button.pack(side="left", padx=(0, 6))

        # tk.Label is intentional: _set_status changes the foreground dynamically.
        self.status_label = tk.Label(
            controls, text="● READY", bg="#081D22", fg=self.GREEN,
            padx=12, pady=7, font=(self.mono_font, 8, "bold"),
            highlightbackground="#15505A", highlightthickness=1,
        )
        self.status_label.pack(side="left")

    def _build_core_panel(self, parent: tk.Frame) -> None:
        import customtkinter as ctk

        panel = ctk.CTkFrame(
            parent, width=238, fg_color="#061019", corner_radius=23,
            border_width=1, border_color="#173542",
        )
        panel.grid(row=0, column=2, sticky="nsew", padx=(10, 0))
        panel.grid_propagate(False)
        panel.rowconfigure(0, minsize=255)
        panel.rowconfigure(3, weight=1)
        panel.columnconfigure(0, weight=1)

        top = ctk.CTkFrame(panel, fg_color="transparent")
        top.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 0))
        top.grid_rowconfigure(0, weight=1)
        top.grid_columnconfigure(0, weight=1)
        animation = self.agent.get_setting("animation", "true").casefold() == "true"
        profile = self.runtime.hardware.performance_profile(self._active_performance)
        self.orb = HolographicCore(top, profile, animation_enabled=animation)
        self.orb.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(
            panel, text="COGNITIVE CORE", text_color="#6CF7FF",
            font=(self.mono_font, 9, "bold"),
        ).grid(row=1, column=0, sticky="w", padx=15, pady=(5, 0))

        diagnostics = ctk.CTkFrame(
            panel, fg_color="#081822", corner_radius=15,
            border_width=1, border_color="#12313F",
        )
        diagnostics.grid(row=2, column=0, sticky="ew", padx=10, pady=9)
        self.mode_label = ctk.CTkLabel(
            diagnostics, text=f"PROFILE  {self._active_performance}",
            text_color="#49E8F6", font=(self.mono_font, 8, "bold"), anchor="w",
        )
        self.mode_label.pack(fill="x", padx=11, pady=(9, 2))
        brain_state = "ONLINE" if self.runtime.brain is not None else "FALLBACK"
        self.brain_label = ctk.CTkLabel(
            diagnostics,
            text=(
                f"SMART    {'LOADED' if self.runtime.neural and self.runtime.neural.loaded else 'LAZY'}\n"
                f"AGENT    {brain_state}\n"
                f"NETWORK  {'ENABLED' if self.agent.internet_enabled else 'LOCAL ONLY'}"
            ),
            text_color="#64818F", justify="left", anchor="w", font=(self.mono_font, 7),
        )
        self.brain_label.pack(fill="x", padx=11, pady=(0, 9))

        navigation = ctk.CTkScrollableFrame(
            panel, fg_color="transparent", corner_radius=0,
            scrollbar_button_color="#153848", scrollbar_button_hover_color="#2ADDF6",
        )
        navigation.grid(row=3, column=0, sticky="nsew", padx=5, pady=(0, 8))
        items = (
            ("＋  گفتگوی جدید", self.clear_conversation),
            ("◷  تاریخچه", self.show_history),
            ("▤  دانش محلی", self.show_knowledge),
            ("⚡  آزمایشگاه", self.open_lab),
            ("◉  عملیات", self.show_action_viewer),
            ("≡  گزارش فعالیت", self.show_logs),
            ("⚙  تنظیمات", self.open_settings),
        )
        for label, command in items:
            ctk.CTkButton(
                navigation, text=label, command=command,
                height=34, corner_radius=11, anchor="e",
                fg_color="#0A1924", hover_color="#0F3040",
                border_width=1, border_color="#112B38",
                text_color="#A6C4D0", font=(self.ui_font, 8, "bold"),
            ).pack(fill="x", pady=3, padx=2)

    def _build_monitor_panel(self, parent: tk.Frame) -> None:
        holder = tk.Frame(parent, bg="#030810", width=245)
        holder.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        holder.grid_propagate(False)
        holder.rowconfigure(0, weight=1)
        holder.columnconfigure(0, weight=1)
        self.monitor_panel = MonitorPanel(
            holder, bg="#030810", panel="#061019", border="#173542"
        )
        self.monitor_panel.grid(row=0, column=0, sticky="nsew")

    def _build_chat_panel(self, parent: tk.Frame) -> None:
        if not ModernChatSurface.available():
            raise RuntimeError("CustomTkinter 6.0.0 is required. Run setup.ps1 to install JARVIS runtime dependencies.")
        self._build_modern_chat_panel(parent)

    def _build_modern_chat_panel(self, parent: tk.Frame) -> None:
        import customtkinter as ctk

        panel = ctk.CTkFrame(parent, fg_color="transparent", corner_radius=0)
        panel.grid(row=0, column=0, sticky="nsew")
        panel.grid_rowconfigure(0, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        self.chat_surface = ModernChatSurface(
            panel,
            ui_font=self.ui_font,
            mono_font=self.mono_font,
            colors={
                "panel": "#071019", "border": "#173A4A",
                "text": self.TEXT, "cyan": self.CYAN,
            },
            clear_command=self.clear_conversation,
        )
        self.chat_surface.grid(row=0, column=0, sticky="nsew")

        toolstrip = ctk.CTkFrame(
            panel, fg_color="#061019", corner_radius=16,
            border_width=1, border_color="#143240",
        )
        toolstrip.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        toolstrip.grid_columnconfigure(0, weight=1)

        copy_controls = ctk.CTkFrame(toolstrip, fg_color="transparent")
        copy_controls.grid(row=0, column=0, sticky="w", padx=8, pady=7)
        ctk.CTkOptionMenu(
            copy_controls, variable=self.copy_count_var,
            values=("۱ پیام", "۵ پیام", "۱۰ پیام", "۲۰ پیام", "۵۰ پیام", "همه پیام‌ها"),
            width=104, height=27, corner_radius=10,
            fg_color="#0B1C28", button_color="#123343", button_hover_color="#17495B",
            text_color="#9EC3CE", dropdown_fg_color="#091722",
            dropdown_hover_color="#123343", font=(self.ui_font, 7),
        ).pack(side="left", padx=(0, 5))
        self.copy_chat_button = ctk.CTkButton(
            copy_controls, text="COPY LINK", command=self._copy_conversation,
            width=86, height=27, corner_radius=10,
            fg_color="#0B1C28", hover_color="#123343",
            border_width=1, border_color="#173A49",
            text_color="#7EDDE8", font=(self.mono_font, 7, "bold"),
        )
        self.copy_chat_button.pack(side="left")

        feedback = ctk.CTkFrame(toolstrip, fg_color="transparent")
        feedback.grid(row=0, column=1, sticky="e", padx=8, pady=7)
        for label, command in (
            ("STOP", self.stop_response),
            ("EDIT", self.edit_last),
            ("REGEN", self.regenerate),
            ("TEACH", self.open_teach),
            ("👎", lambda: self._feedback(False)),
            ("👍", lambda: self._feedback(True)),
        ):
            ctk.CTkButton(
                feedback, text=label, command=command,
                width=42 if len(label) <= 2 else 58, height=27, corner_radius=10,
                fg_color="#0B1C28", hover_color="#123343",
                border_width=1, border_color="#112F3D",
                text_color="#9FC0CC", font=(self.mono_font, 7, "bold"),
            ).pack(side="right", padx=2)

        composer = ctk.CTkFrame(
            panel, fg_color="#07111B", corner_radius=22,
            border_width=1, border_color="#1B4657",
        )
        composer.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        composer.grid_columnconfigure(1, weight=1)

        left = ctk.CTkFrame(composer, fg_color="transparent")
        left.grid(row=0, column=0, sticky="ns", padx=(10, 7), pady=9)
        self.attach_button = ctk.CTkButton(
            left, text="＋", command=self.attach_file,
            width=38, height=38, corner_radius=13,
            fg_color="#0C2230", hover_color="#123D4E",
            border_width=1, border_color="#1A5363",
            text_color="#5CEEFF", font=(self.ui_font, 17, "bold"),
        )
        self.attach_button.pack(pady=(0, 5))
        ctk.CTkButton(
            left, text="↶", command=lambda: self._submit_command("undo"),
            width=38, height=30, corner_radius=11,
            fg_color="#0A1823", hover_color="#102C3A",
            text_color="#6F94A1", font=(self.ui_font, 12),
        ).pack()

        input_card = ctk.CTkFrame(
            composer, fg_color="#08151F", corner_radius=17,
            border_width=1, border_color="#123542",
        )
        input_card.grid(row=0, column=1, sticky="nsew", padx=0, pady=9)
        input_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            input_card, text="COMMAND / MESSAGE", text_color="#3DBFCC",
            font=(self.mono_font, 7, "bold"), anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=13, pady=(7, 0))
        self.message_input = tk.Text(
            input_card, height=4, bg="#08151F", fg="#EAFBFF",
            insertbackground="#5CEEFF", selectbackground="#315DD1",
            wrap="word", borderwidth=0, highlightthickness=0,
            padx=10, pady=7, undo=True, maxundo=40, font=(self.ui_font, 11),
        )
        self.message_input.grid(row=1, column=0, sticky="ew", padx=4)
        self.message_input.bind("<Return>", self._enter_pressed)
        self.message_input.bind("<Shift-Return>", self._shift_enter)
        self.message_input.bind("<KeyRelease>", self._input_direction_changed)
        self._bind_input_clipboard_shortcuts()
        ctk.CTkCheckBox(
            input_card, text="DRY RUN  •  فقط نمایش برنامه، بدون اجرا",
            variable=self.dry_run_var, width=220, height=23,
            checkbox_width=16, checkbox_height=16, corner_radius=5,
            border_color="#1B4657", fg_color="#2D79E8", hover_color="#3988F5",
            text_color="#617F8C", font=(self.ui_font, 7),
        ).grid(row=2, column=0, sticky="e", padx=12, pady=(0, 7))

        right = ctk.CTkFrame(composer, fg_color="transparent")
        right.grid(row=0, column=2, sticky="ns", padx=(7, 10), pady=9)
        self.send_button = ctk.CTkButton(
            right, text="SEND\n◈", command=self.send_message,
            width=74, height=76, corner_radius=18,
            fg_color="#1F63E9", hover_color="#347AF4",
            border_width=1, border_color="#64A1FF",
            text_color="#FFFFFF", font=(self.mono_font, 9, "bold"),
        )
        self.send_button.pack(fill="both", expand=True)

    def _build_footer(self, parent: tk.Frame) -> None:
        footer = tk.Frame(parent, bg="#030810", height=27)
        footer.grid(row=2, column=0, sticky="ew", pady=(7, 0))
        tk.Label(
            footer,
            text="ENTER SEND  •  SHIFT+ENTER NEW LINE  •  CTRL+C/V  •  CTRL+K COMMAND PALETTE",
            bg="#030810", fg="#365563", font=(self.mono_font, 7),
        ).pack(side="left")
        self.metrics_label = tk.Label(
            footer, text="CPU 0%  •  RAM 0 MB", bg="#030810", fg="#4E7887",
            font=(self.mono_font, 7, "bold"),
        )
        self.metrics_label.pack(side="right")

    def _configure_chat_tags(self) -> None:
        self.chat.tag_configure(
            "assistant_label", foreground=self.CYAN, font=(self.ui_font, 8, "bold"),
            justify="left"
        )
        self.chat.tag_configure(
            "user_label", foreground="#9CB8FF", font=(self.ui_font, 8, "bold"),
            justify="right"
        )
        self.chat.tag_configure(
            "assistant", foreground=self.TEXT, background=self.ASSISTANT_BG,
            lmargin1=12, lmargin2=12, rmargin=70, spacing1=5, spacing3=10
        )
        self.chat.tag_configure(
            "assistant_fa", foreground=self.TEXT, background=self.ASSISTANT_BG,
            justify="left", lmargin1=12, lmargin2=12, rmargin=70,
            spacing1=5, spacing3=10
        )
        self.chat.tag_configure(
            "user", foreground=self.TEXT, background=self.USER_BG, justify="right",
            lmargin1=70, lmargin2=70, rmargin=12, spacing1=5, spacing3=10
        )
        self.chat.tag_configure(
            "system", foreground=self.MUTED, justify="center", font=(self.ui_font, 8)
        )
        self.chat.tag_configure(
            "attachment", foreground="#B9D8EA", background="#0C2735",
            lmargin1=22, lmargin2=22, rmargin=22, spacing1=7, spacing3=7,
            font=(self.mono_font, 9)
        )
        self.chat.tag_configure(
            "error", foreground=self.RED, justify="center",
            font=(self.ui_font, 9, "bold")
        )
        self.chat.tag_configure(
            "code", foreground="#C9F3FF", background="#07111C",
            font=(self.mono_font, 9), lmargin1=20, lmargin2=20, rmargin=20,
            spacing1=5, spacing3=5,
        )
        self.chat.tag_configure("url", foreground="#52B9FF", underline=True)
        self.chat.tag_bind("url", "<Button-1>", self._open_clicked_url)
        self.chat.tag_bind("url", "<Enter>", lambda _event: self.chat.configure(cursor="hand2"))
        self.chat.tag_bind("url", "<Leave>", lambda _event: self.chat.configure(cursor=""))

    def _build_chat_context_menu(self) -> None:
        menu = tk.Menu(self.chat, tearoff=False, bg=self.PANEL_2, fg=self.TEXT)
        menu.add_command(label="کپی", command=self._copy_chat_selection)
        menu.add_separator()
        menu.add_command(label="پاک‌کردن گفتگو", command=self.clear_conversation)
        self._context_menu = menu
        for sequence in ("<Button-3>", "<Button-2>"):
            self.chat.bind(sequence, self._show_chat_context_menu)
        for sequence in ("<Control-c>", "<Control-C>"):
            self.chat.bind(sequence, self._copy_chat_selection_event)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-k>", self._open_palette_event)
        self.root.bind("<Control-K>", self._open_palette_event)
        self.root.bind("<Control-l>", lambda _event: self.clear_conversation())
        try:
            self.root.bind("<<Drop>>", self._drop_received)
        except tk.TclError:
            pass

    def _bind_input_clipboard_shortcuts(self) -> None:
        for sequence in ("<Control-c>", "<Control-C>"):
            self.message_input.bind(sequence, self._copy_input_selection)
        for sequence in ("<<Paste>>", "<Control-v>", "<Control-V>", "<Shift-Insert>"):
            self.message_input.bind(sequence, self._paste_input_clipboard)
        # On Windows a Persian keyboard layout may report the physical V key
        # with a non-Latin keysym, so the literal <Control-v> binding is missed.
        self.message_input.bind("<Control-KeyPress>", self._clipboard_layout_keypress)
        for sequence in ("<Button-3>", "<Button-2>"):
            self.message_input.bind(sequence, self._show_input_context_menu)

    def _clipboard_layout_keypress(self, event: tk.Event[tk.Text]) -> str | None:
        keysym = str(getattr(event, "keysym", "")).casefold()
        keycode = int(getattr(event, "keycode", 0) or 0)
        if keysym == "v" or keycode in {55, 86}:
            return self._paste_input_clipboard(event)
        if keysym == "c" or keycode in {54, 67}:
            return self._copy_input_selection(event)
        return None

    def _show_input_context_menu(self, event: tk.Event[tk.Text]) -> str:
        if self._input_context_menu is not None:
            try:
                self._input_context_menu.destroy()
            except tk.TclError:
                pass
        menu = tk.Menu(
            self.message_input,
            tearoff=False,
            bg=self.PANEL_2,
            fg=self.TEXT,
            activebackground=self.BLUE,
            activeforeground="#FFFFFF",
        )
        menu.add_command(label="کپی", command=self._copy_input_selection)
        menu.add_command(label="چسباندن", command=self._paste_input_clipboard)
        menu.add_separator()
        menu.add_command(
            label="انتخاب همه",
            command=lambda: (
                self.message_input.tag_add("sel", "1.0", "end-1c"),
                self.message_input.mark_set("insert", "end-1c"),
            ),
        )
        self._input_context_menu = menu
        menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _copy_input_selection(self, _event: tk.Event[tk.Text] | None = None) -> str:
        try:
            selected = self.message_input.get("sel.first", "sel.last")
        except tk.TclError:
            return "break"
        self.root.clipboard_clear()
        self.root.clipboard_append(ModernChatSurface._clean_copy(selected))
        self.root.update_idletasks()
        return "break"

    def _paste_input_clipboard(self, _event: tk.Event[tk.Text] | None = None) -> str:
        try:
            value = ModernChatSurface._clean_display(self.root.clipboard_get())
        except tk.TclError:
            return "break"
        try:
            self.message_input.delete("sel.first", "sel.last")
        except tk.TclError:
            pass
        self.message_input.insert("insert", value)
        self.message_input.see("insert")
        self._input_direction_changed()
        return "break"

    @staticmethod
    def _copy_message_limit(selection: str) -> int | None:
        normalized = str(selection).translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))
        if "همه" in normalized or normalized.casefold() == "all":
            return None
        match = re.search(r"\d+", normalized)
        return max(1, int(match.group(0))) if match else None

    @classmethod
    def _format_conversation_copy(
        cls, messages: list[tuple[str, str]], selection: str
    ) -> str:
        limit = cls._copy_message_limit(selection)
        selected = messages[-limit:] if limit is not None else messages
        blocks: list[str] = []
        for role, text in selected:
            label = "شما" if role == "user" else "JARVIS"
            clean = ModernChatSurface._clean_copy(str(text)).strip()
            if clean:
                blocks.append(f"{label}:\n{clean}")
        return "\n\n".join(blocks)

    def _copy_conversation(self) -> None:
        transcript = self._format_conversation_copy(
            self._chat_messages, self.copy_count_var.get()
        )
        if not transcript:
            self.root.bell()
            self._set_copy_button_feedback("پیامی نیست")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(transcript)
        self.root.update_idletasks()
        self._set_copy_button_feedback("کپی شد ✓")

    def _set_copy_button_feedback(self, text: str) -> None:
        if self.copy_chat_button is None:
            return
        try:
            self.copy_chat_button.configure(text=text)
            self.root.after(1400, lambda: self.copy_chat_button.configure(text="کپی گفتگو"))
        except tk.TclError:
            pass

    def _button(
        self, parent: tk.Widget, text: str, command: Any,
        background: str, width: int
    ) -> tk.Button:
        return tk.Button(
            parent, text=text, command=command, width=width, bg=background,
            fg=self.TEXT,
            activebackground=self.CYAN if background == self.BLUE else self.BORDER,
            activeforeground=self.BG if background == self.BLUE else self.TEXT,
            borderwidth=0, relief="flat", cursor="hand2", padx=8, pady=8,
            font=(self.ui_font, 8, "bold")
        )

    def _restore_history(self) -> None:
        for message in self.agent.recent_history():
            language = str(message.metadata.get("language") or detect_language(message.content))
            if message.role == "user":
                self._append_user(message.content, language, message.created_at)
            elif message.role == "assistant":
                self._append_assistant(message.content, language, message.created_at)
            else:
                self._append_system(message.content)

    def _timestamp(self, explicit: str = "") -> str:
        if self.agent.get_setting("timestamps", "false").casefold() != "true":
            return ""
        if explicit:
            return explicit[11:16]
        return time.strftime("%H:%M")

    def _insert_rich_text(self, text: str, body_tag: str) -> None:
        """Render fenced code and clickable URLs without a browser engine."""
        def plain(value: str) -> str:
            return ModernChatSurface._clean_display(value)

        for block in re.split(r"(```[\s\S]*?```)", plain(text)):
            if not block:
                continue
            if block.startswith("```") and block.endswith("```"):
                payload = block[3:-3]
                if "\n" in payload:
                    first, rest = payload.split("\n", 1)
                    if re.fullmatch(r"[\w.+-]{1,20}", first.strip()):
                        payload = rest
                self.chat.insert("end", f"{payload.rstrip()}\n", (body_tag, "code"))
                continue
            cursor = 0
            for match in re.finditer(r"https?://[^\s<>]+", block):
                if match.start() > cursor:
                    self.chat.insert("end", plain(block[cursor:match.start()]), body_tag)
                self.chat.insert("end", match.group(0).rstrip(".,)"), (body_tag, "url"))
                cursor = match.start() + len(match.group(0).rstrip(".,)"))
            if cursor < len(block):
                self.chat.insert("end", plain(block[cursor:]), body_tag)

    def _open_clicked_url(self, event: tk.Event[tk.Text]) -> str:
        index = self.chat.index(f"@{event.x},{event.y}")
        ranges = self.chat.tag_prevrange("url", f"{index}+1c")
        if ranges:
            value = self.chat.get(ranges[0], ranges[1]).strip("\u2066\u2067\u2068\u2069")
            if value.startswith(("http://", "https://")):
                webbrowser.open(value)
        return "break"

    def _append(
        self, label: str, text: str, label_tag: str, body_tag: str,
        timestamp: str = ""
    ) -> None:
        if self.chat_surface is not None:
            role = "user" if label_tag == "user_label" else "assistant"
            language = "fa" if role == "user" or body_tag == "assistant_fa" else "en"
            self.chat_surface.add_message(
                role, text, language, self._timestamp(timestamp)
            )
            return
        self.chat.configure(state="normal")
        if self.chat.index("end-1c") != "1.0":
            self.chat.insert("end", "\n")
        shown_time = self._timestamp(timestamp)
        heading = f"{label}  {shown_time}" if shown_time else label
        self.chat.insert("end", f"{heading}\n", label_tag)
        self._insert_rich_text(f"{text.strip()}\n", body_tag)
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _append_user(self, text: str, _language: str, timestamp: str = "") -> None:
        self._chat_messages.append(("user", str(text)))
        if self.chat_surface is not None:
            self.chat_surface.add_message(
                "user", text, _language, self._timestamp(timestamp)
            )
            return
        self._append("شما", text, "user_label", "user", timestamp)

    def _append_assistant(self, text: str, language: str, timestamp: str = "") -> None:
        self._chat_messages.append(("assistant", str(text)))
        if self.chat_surface is not None:
            self.chat_surface.add_message(
                "assistant", text, language, self._timestamp(timestamp)
            )
            return
        body_tag = "assistant_fa" if language == "fa" else "assistant"
        self._append("JARVIS", text, "assistant_label", body_tag, timestamp)

    def _stream_assistant(self, text: str, language: str) -> None:
        """Incrementally render a completed local response without blocking Tk."""
        if self.chat_surface is not None:
            self._append_assistant(text, language)
            self._set_chat_busy(False)
            self._set_status("READY")
            return
        if language == "fa":
            self._append_assistant(text, language)
            self._set_chat_busy(False)
            self._set_status("READY")
            return
        self._chat_messages.append(("assistant", str(text)))
        body_tag = "assistant_fa" if language == "fa" else "assistant"
        chunks = re.findall(r"\S+\s*|\s+", text.strip()) or [text]
        self.chat.configure(state="normal")
        if self.chat.index("end-1c") != "1.0":
            self.chat.insert("end", "\n")
        self.chat.insert("end", "JARVIS\n", "assistant_label")
        self.chat.configure(state="disabled")
        self._set_status("SPEAKING")
        self.send_button.configure(state="disabled")

        def emit(index: int = 0) -> None:
            self._stream_after_id = None
            if self._closing or self._stop_stream_requested:
                self._set_chat_busy(False)
                return
            if index >= len(chunks):
                self.chat.configure(state="normal")
                self.chat.insert("end", "\n", body_tag)
                self.chat.configure(state="disabled")
                self.chat.see("end")
                self._set_chat_busy(False)
                self._set_status("READY")
                return
            self.chat.configure(state="normal")
            self.chat.insert("end", chunks[index], body_tag)
            self.chat.configure(state="disabled")
            self.chat.see("end")
            delay = 12 if self._active_performance != "LOW" else 22
            self._stream_after_id = self.root.after(delay, lambda: emit(index + 1))

        emit()

    def _append_system(self, text: str, error: bool = False) -> None:
        if self.chat_surface is not None:
            self.chat_surface.add_system(text, error)
            return
        self.chat.configure(state="normal")
        self.chat.insert("end", f"\n{text.strip()}\n", "error" if error else "system")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _append_attachment_card(self, data: dict[str, Any]) -> None:
        name = str(data.get("name", "file"))
        size = int(data.get("size", 0) or 0)
        kind = str(data.get("kind", "unknown")).upper()
        if size < 1024:
            size_label = f"{size} B"
        elif size < 1024 * 1024:
            size_label = f"{size / 1024:.1f} KB"
        else:
            size_label = f"{size / (1024 * 1024):.2f} MB"
        if self.chat_surface is not None:
            self.chat_surface.add_attachment(name, kind, size_label)
            return
        self.chat.configure(state="normal")
        self.chat.insert("end", f"\n  📎 {name}\n  {kind}  •  {size_label}\n", "attachment")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _set_status(self, status: str) -> None:
        internal = status.upper()
        normalized = {
            "PLANNING": "THINKING", "READING": "WORKING", "OPENING": "WORKING",
            "FINDING": "WORKING", "LEARNING": "WORKING", "TRAINING": "WORKING",
            "SPEAKING": "WORKING", "LISTENING": "READY", "IDLE": "READY",
        }.get(internal, internal)
        colors = {
            "READY": self.GREEN, "IDLE": self.CYAN, "LISTENING": self.BLUE,
            "THINKING": "#C493FF", "SEARCHING": "#52B9FF", "READING": "#5FE7F2",
            "PLANNING": "#8FA8FF", "WORKING": self.AMBER,
            "OPENING": "#52B9FF", "FINDING": "#8FA8FF",
            "VERIFYING": "#62E6C8", "LEARNING": "#FF9FEA",
            "TRAINING": "#FFB347", "SPEAKING": self.AMBER, "ERROR": self.RED,
        }
        self.status_label.configure(
            text=f"● {normalized}", fg=colors.get(normalized, self.MUTED)
        )
        self.orb.set_state(normalized)
        if hasattr(self, "monitor_panel"):
            self.monitor_panel.add_activity(normalized)

    def _enter_pressed(self, event: tk.Event[tk.Text]) -> str | None:
        if event.state & 0x0001:
            return None
        self.send_message()
        return "break"

    def _input_direction_changed(self, _event: tk.Event[tk.Text] | None = None) -> None:
        value = self.message_input.get("1.0", "end-1c")
        self.message_input.tag_remove("input_direction", "1.0", "end")
        self.message_input.tag_configure(
            "input_direction", justify="right" if detect_language(value) == "fa" else "left"
        )
        self.message_input.tag_add("input_direction", "1.0", "end")

    @staticmethod
    def _shift_enter(_event: tk.Event[tk.Text]) -> None:
        return None

    def _set_chat_busy(self, busy: bool) -> None:
        self._chat_busy = bool(busy)
        state = "disabled" if busy else "normal"
        self.send_button.configure(state=state)
        self.message_input.configure(state=state)
        if not busy:
            self.message_input.focus_set()

    def send_message(self) -> None:
        if self._chat_busy:
            self.root.bell()
            return
        text = self.message_input.get("1.0", "end-1c").strip()
        if not text:
            return
        self._last_user_text = text
        request_text = (
            f"بدون اجرا، فقط برنامه را نشان بده: {text}"
            if self.dry_run_var.get() else text
        )
        self._stop_stream_requested = False
        self.message_input.delete("1.0", "end")
        self._append_user(text, detect_language(text))
        self._set_status("THINKING")
        self._set_chat_busy(True)
        self._tasks.put(("chat", request_text))

    def attach_file(self) -> None:
        extensions = " ".join(
            f"*{extension}" for extension in (
                ".txt", ".py", ".js", ".ts", ".json", ".md", ".csv", ".html",
                ".css", ".xml", ".log", ".pdf", ".docx", ".xlsx", ".zip"
            )
        )
        path = filedialog.askopenfilename(
            parent=self.root, title="Attach a file to Jarvis",
            filetypes=(("Supported files", extensions), ("All files", "*.*"))
        )
        if path:
            self._queue_attachment(path)

    def _queue_attachment(self, path: str) -> None:
        self._append_system(f"Reading {Path(path).name} …")
        self._set_status("READING")
        self._tasks.put(("attach", path))

    def _drop_received(self, event: tk.Event[tk.Misc]) -> str:
        data = str(getattr(event, "data", "")).strip().strip("{}")
        if data and Path(data).is_file():
            self._queue_attachment(data)
        return "break"

    def _personality_changed(self, _event: tk.Event[ttk.Combobox] | None = None) -> None:
        self._set_status("WORKING")
        self._tasks.put(("personality", self.personality_var.get()))

    def _resolve_performance(self, value: str) -> str:
        normalized = value.upper()
        if normalized == "AUTO":
            return self.runtime.hardware.recommended_performance()
        return normalized if normalized in {"LOW", "BALANCED", "ULTRA"} else "BALANCED"

    def _performance_changed(self, _event: tk.Event[ttk.Combobox] | None = None) -> None:
        saved = self.performance_var.get().upper()
        self.agent.update_setting("performance", saved)
        self._active_performance = self._resolve_performance(saved)
        self.orb.set_profile(
            self.runtime.hardware.performance_profile(self._active_performance)
        )
        self.mode_label.configure(text=f"PROFILE  {self._active_performance}")

    def _worker_loop(self) -> None:
        while True:
            task = self._tasks.get()
            if task is None:
                self._tasks.task_done()
                return
            kind, payload = task
            try:
                if kind == "chat":
                    result = self.agent.respond(str(payload))
                elif kind == "attach":
                    result = self.agent.attach_file(str(payload))
                elif kind == "personality":
                    result = self.agent.set_personality(str(payload))
                elif kind == "new_session":
                    result = self.agent.new_session()
                elif kind == "clear_memory":
                    self.agent.clear_long_term_memory()
                    result = True
                else:
                    raise ValueError(f"Unknown UI task: {kind}")
                self._results.put((kind, result))
            except Exception as exc:
                self.logger.exception("Background task failed")
                self._results.put(("error", str(exc)))
            finally:
                self._tasks.task_done()

    def _poll_results(self) -> None:
        if self._closing:
            return
        processed = False
        while True:
            try:
                kind, result = self._results.get_nowait()
            except queue.Empty:
                break
            processed = True
            if kind == "status":
                self._set_status(str(result))
            elif kind in {"chat", "attach"} and isinstance(result, AgentReply):
                if kind == "chat" and self._stop_stream_requested:
                    self._stop_stream_requested = False
                    self._set_chat_busy(False)
                    self._results.task_done()
                    continue
                if result.event == "clear_conversation":
                    self._clear_chat_display()
                if result.event == "attachment":
                    self._append_attachment_card(result.data)
                self._last_reply_text = result.text
                latency = result.data.get("latency_ms")
                rate = result.data.get("tokens_per_second")
                if latency is not None or rate is not None:
                    self.monitor_panel.update_inference(
                        float(latency) if latency is not None else None,
                        float(rate) if rate is not None else None,
                    )
                plan = result.data.get("plan")
                if isinstance(plan, dict):
                    self._last_plan = plan
                    steps = plan.get("steps", [])
                    if isinstance(steps, list):
                        calls = [
                            str(step.get("tool_call", {}).get("tool", ""))
                            for step in steps if isinstance(step, dict) and step.get("tool_call")
                        ]
                        if calls:
                            self.monitor_panel.add_activity(" → ".join(calls))
                if kind == "chat":
                    self._stream_assistant(result.text, result.language)
                else:
                    self._append_assistant(result.text, result.language)
                if result.event == "open_settings":
                    self.root.after_idle(self.open_settings)
                if result.status == "ERROR":
                    self._set_status("ERROR")
            elif kind == "personality":
                self._append_system(str(result))
            elif kind == "new_session":
                self._clear_chat_display()
                self._append_system("New local conversation started.")
            elif kind == "clear_memory":
                self._append_system("Saved conversation context and facts were cleared.")
            else:
                self._append_system(f"Error: {result}", error=True)
                self._set_status("ERROR")
                if kind == "error":
                    self._set_chat_busy(False)
            self._results.task_done()
        if processed and self._tasks.unfinished_tasks == 0 and self._stream_after_id is None:
            self.root.after(350, lambda: self._set_status("READY"))
        self.root.after(65, self._poll_results)

    def _update_metrics(self) -> None:
        if self._closing:
            return
        try:
            sample = self.live_monitor.sample()
            self.metrics_label.configure(
                text=(
                    f"CPU {sample.cpu_percent:>3.0f}%  •  JARVIS RAM "
                    f"{sample.app_ram_mb:>5.1f} MB  •  DISK {sample.disk_percent:>3.0f}%"
                )
            )
            self.monitor_panel.update_snapshot(sample)
            if self.agent.debug:
                self.agent.events.performance.write(
                    "sample", cpu=round(sample.cpu_percent, 3),
                    jarvis_cpu=round(sample.app_cpu_percent, 3),
                    ram=round(sample.ram_percent, 3), jarvis_ram_mb=round(sample.app_ram_mb, 3),
                    gpu=sample.gpu_percent, vram_mb=sample.vram_used_mb,
                )
        except Exception:
            self.metrics_label.configure(text="CPU —  •  RAM —")
        self.root.after(1000, self._update_metrics)

    def _open_palette_event(self, _event: tk.Event[tk.Misc]) -> str:
        self.open_palette()
        return "break"

    def open_palette(self) -> None:
        commands: list[tuple[str, Any]] = [
            ("Open settings", self.open_settings),
            ("Attach file", self.attach_file),
            ("New conversation", self.clear_conversation),
            ("Clear saved memory", self._confirm_clear_memory),
            ("Show system information", lambda: self._submit_command("system info")),
        ]
        for name in self.runtime.personalities.available_names():
            commands.append((f"Personality: {name}", lambda value=name: self._select_personality(value)))
        for profile in ("AUTO", "LOW", "BALANCED", "ULTRA"):
            commands.append((f"Performance: {profile}", lambda value=profile: self._select_performance(value)))
        CommandPalette(self.root, commands, self.colors)

    def _submit_command(self, message: str) -> None:
        self.message_input.delete("1.0", "end")
        self.message_input.insert("1.0", message)
        self.send_message()

    def _select_personality(self, value: str) -> None:
        self.personality_var.set(value)
        self._personality_changed()

    def _select_performance(self, value: str) -> None:
        self.performance_var.set(value)
        self._performance_changed()

    def _show_text_window(self, title: str, content: str) -> None:
        window = tk.Toplevel(self.root)
        window.title(title)
        window.configure(bg=self.BG)
        window.geometry("780x560")
        text = tk.Text(
            window, bg=self.PANEL, fg=self.TEXT, insertbackground=self.TEXT,
            wrap="word", borderwidth=0, padx=16, pady=14,
            font=(self.mono_font, 9),
        )
        text.insert("1.0", content)
        text.configure(state="disabled")
        text.pack(fill="both", expand=True, padx=12, pady=12)

    def show_history(self) -> None:
        lines: list[str] = []
        for message in self.agent.recent_history():
            lines.append(f"[{message.created_at}] {message.role.upper()}\n{message.content}\n")
        self._show_text_window("JARVIS — تاریخچهٔ گفتگو", "\n".join(lines) or "تاریخچه‌ای وجود ندارد.")

    def show_action_viewer(self) -> None:
        plan = self._last_plan
        if not plan:
            self._show_text_window("JARVIS — نمایش عملیات", "هنوز برنامهٔ عملیاتی ثبت نشده است.")
            return
        lines = [f"درخواست: {plan.get('request', '')}", f"Dry Run: {bool(plan.get('dry_run'))}", ""]
        for step in plan.get("steps", []):
            if not isinstance(step, dict):
                continue
            lines.append(f"{step.get('index', '•')}. {step.get('description', '')}")
            call = step.get("tool_call")
            if not isinstance(call, dict):
                continue
            tool_name = str(call.get("tool", ""))
            specification = self.runtime.tools.spec(tool_name)
            schema = specification.schema() if specification else {}
            lines.append(
                f"   ابزار: {tool_name}  |  دسترسی: {schema.get('permission_level', 'BLOCKED')}"
                f"  |  تأیید: {'لازم' if call.get('requires_confirmation') else 'لازم نیست'}"
            )
            if self.agent.debug:
                lines.append(f"   آرگومان‌ها: {json.dumps(call.get('arguments', {}), ensure_ascii=False, default=str)}")
        self._show_text_window("JARVIS — نمایش عملیات", "\n".join(lines))

    def show_knowledge(self) -> None:
        stats = self.runtime.rag.stats()
        attachment = self.agent.current_attachment
        window = tk.Toplevel(self.root)
        window.title("JARVIS — دانش محلی")
        window.configure(bg=self.BG)
        window.geometry("520x310")
        tk.Label(
            window, text="دانش محلی Hybrid RAG", bg=self.BG, fg=self.CYAN,
            font=(self.ui_font, 12, "bold"),
        ).pack(anchor="e", padx=20, pady=(20, 7))
        tk.Label(
            window,
            text=(
                f"سندهای ایندکس‌شده: {stats['documents']}\n"
                f"قطعه‌های ایندکس‌شده: {stats['chunks']}\n\n"
                "رتبه‌بندی: Keyword + BM25 + بردار هش‌شدهٔ محلی + Rerank\n"
                "بدون embedding ابری و بدون مدل pretrained."
            ),
            bg=self.BG, fg=self.TEXT, justify="right", font=(self.ui_font, 10),
        ).pack(anchor="e", padx=20)
        button = self._button(
            window, "ایندکس فایل پیوست‌شده", self._index_attachment,
            self.BLUE, width=28,
        )
        button.configure(state="normal" if attachment else "disabled")
        button.pack(anchor="e", padx=20, pady=20)

    def _index_attachment(self) -> None:
        try:
            result = self.agent.index_attachment()
            self._append_system(
                f"Knowledge index updated: {result.get('chunks', 0)} chunks."
            )
        except Exception as exc:
            messagebox.showerror("Knowledge", str(exc), parent=self.root)

    def show_logs(self) -> None:
        parts: list[str] = []
        for name in (
            "app.log", "errors.log", "chat.jsonl", "agent.jsonl", "tools.jsonl",
            "search.jsonl", "performance.jsonl",
        ):
            path = self.runtime.config.paths.logs_dir / name
            parts.append(f"===== {name} =====")
            if not path.is_file():
                parts.append("(no events yet)")
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                parts.extend(lines[-100:])
            except OSError as exc:
                parts.append(str(exc))
        self._show_text_window("JARVIS — Logs / Activity", "\n".join(parts))

    def _feedback(self, positive: bool) -> None:
        if not self._last_reply_text:
            return
        self.agent.add_feedback(positive)
        self._append_system("Feedback saved locally. Thank you." if positive else "Feedback saved to the review queue.")

    def open_teach(self) -> None:
        if not self._last_user_text:
            messagebox.showinfo("Teach JARVIS", "Send a message first.", parent=self.root)
            return
        window = tk.Toplevel(self.root)
        window.title("TEACH JARVIS — Learning Queue")
        window.configure(bg=self.BG)
        window.geometry("700x680")
        fields: dict[str, tk.Text] = {}
        initial = {
            "Original Input": self._last_user_text,
            "Jarvis Response": self._last_reply_text,
            "Correct Response": "",
            "Correct Action": "",
            "Correct Arguments (JSON)": "",
            "Notes": "",
        }
        for label, value in initial.items():
            tk.Label(
                window, text=label.upper(), bg=self.BG, fg=self.CYAN,
                font=("Consolas", 8), anchor="w",
            ).pack(fill="x", padx=18, pady=(9, 2))
            widget = tk.Text(
                window, height=2 if label not in {"Jarvis Response", "Correct Response"} else 4,
                bg=self.INPUT, fg=self.TEXT, insertbackground=self.CYAN,
                borderwidth=0, wrap="word", padx=8, pady=6,
            )
            widget.insert("1.0", value)
            if label in {"Original Input", "Jarvis Response"}:
                widget.configure(state="disabled")
            widget.pack(fill="x", padx=18)
            fields[label] = widget

        def save() -> None:
            try:
                self.learning_queue.add(
                    original_input=self._last_user_text,
                    jarvis_response=self._last_reply_text,
                    correct_response=fields["Correct Response"].get("1.0", "end-1c"),
                    correct_action=fields["Correct Action"].get("1.0", "end-1c"),
                    correct_arguments=fields["Correct Arguments (JSON)"].get("1.0", "end-1c"),
                    notes=fields["Notes"].get("1.0", "end-1c"),
                )
            except ValueError as exc:
                messagebox.showerror("Teach JARVIS", str(exc), parent=window)
                return
            self.agent.add_feedback(False, "Teach entry queued")
            self._append_system("Correction added to Learning Queue; weights were not changed.")
            window.destroy()

        self._button(window, "ADD TO LEARNING QUEUE", save, self.BLUE, width=28).pack(
            pady=15
        )

    def regenerate(self) -> None:
        if self._last_user_text and self._tasks.unfinished_tasks == 0:
            value = self._last_user_text
            self.message_input.delete("1.0", "end")
            self.message_input.insert("1.0", value)
            self.send_message()

    def edit_last(self) -> None:
        if self._last_user_text:
            self.message_input.delete("1.0", "end")
            self.message_input.insert("1.0", self._last_user_text)
            self.message_input.focus_set()

    def stop_response(self) -> None:
        self._stop_stream_requested = True
        self.agent.cancel_current()
        if self._stream_after_id is not None:
            try:
                self.root.after_cancel(self._stream_after_id)
            except tk.TclError:
                pass
            self._stream_after_id = None
        self._set_chat_busy(False)
        self._set_status("READY")
        self._append_system("درخواست لغو شد؛ تولید متن و فرمان‌های قابل‌لغو متوقف می‌شوند.")

    def open_lab(self) -> None:
        if self._lab_window is not None and self._lab_window.winfo_exists():
            self._lab_window.lift()
            return
        from jarvis.lab.app import JarvisLab

        self._lab_window = tk.Toplevel(self.root)
        JarvisLab(self._lab_window, self.runtime.config.paths.root)

    def open_settings(self) -> None:
        if self._settings_window is not None and self._settings_window.winfo_exists():
            self._settings_window.lift()
            return
        self._settings_window = SettingsWindow(
            self.root, self.agent, self._apply_settings,
            lambda: self._tasks.put(("clear_memory", None)), self.colors
        )

    def _apply_settings(self, values: dict[str, str | bool]) -> None:
        requested_personality = str(values.pop("personality", self.agent.current_personality))
        if requested_personality != self.agent.current_personality:
            self.personality_var.set(requested_personality)
            self._tasks.put(("personality", requested_personality))
        for key, value in values.items():
            self.agent.update_setting(key, value)
        self.performance_var.set(str(values.get("performance", "AUTO")))
        self._performance_changed()
        self.orb.set_animation(bool(values.get("animation", True)))
        self.brain_label.configure(
            text=(
                f"SMART    {'LOADED' if self.runtime.neural and self.runtime.neural.loaded else 'LAZY' if self.runtime.neural else 'FALLBACK'}\n"
                f"AGENT    {'ONLINE' if self.runtime.brain is not None else 'FALLBACK'}\n"
                f"NETWORK  {'ENABLED' if self.agent.internet_enabled else 'LOCAL ONLY'}"
            )
        )
        self._append_system("Settings applied locally.")

    def clear_conversation(self) -> None:
        self._set_status("WORKING")
        self._tasks.put(("new_session", None))

    def _confirm_clear_memory(self) -> None:
        from tkinter import messagebox

        if messagebox.askyesno(
            "Clear memory",
            "Delete saved facts and conversation context? UI settings are preserved.",
            parent=self.root,
        ):
            self._set_status("WORKING")
            self._tasks.put(("clear_memory", None))

    def _clear_chat_display(self) -> None:
        self._chat_messages.clear()
        if self.chat_surface is not None:
            self.chat_surface.clear()
            return
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")

    def _show_chat_context_menu(self, event: tk.Event[tk.Text]) -> str:
        if self._context_menu is not None:
            self._context_menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _copy_chat_selection(self) -> None:
        if self.chat_surface is not None:
            selected = self.chat_surface.selected_text()
            if not selected:
                return
            self.root.clipboard_clear()
            self.root.clipboard_append(selected)
            return
        try:
            selected = self.chat.get("sel.first", "sel.last")
        except tk.TclError:
            return
        selected = selected.translate(
            {ord(mark): None for mark in "\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"}
        )
        self.root.clipboard_clear()
        self.root.clipboard_append(selected)

    def _copy_chat_selection_event(self, _event: tk.Event[tk.Text] | None = None) -> str:
        self._copy_chat_selection()
        return "break"

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self._stream_after_id is not None:
            try:
                self.root.after_cancel(self._stream_after_id)
            except tk.TclError:
                pass
        self.agent.set_status_callback(None)
        self.agent.cancel_current()
        self._tasks.put(None)
        self._worker.join(timeout=5.0)
        if self._worker.is_alive():
            self.logger.warning("Agent worker is still active; runtime resources were left open safely")
        else:
            try:
                self.runtime.close()
            except Exception:
                self.logger.exception("Runtime close failed")
        self.root.destroy()
