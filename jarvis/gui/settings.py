from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk

from jarvis.agent.core import JarvisAgent


class SettingsWindow(tk.Toplevel):
    def __init__(
        self,
        master: tk.Misc,
        agent: JarvisAgent,
        on_apply: Callable[[dict[str, str | bool]], None],
        on_clear_memory: Callable[[], None],
        colors: dict[str, str],
    ) -> None:
        super().__init__(master)
        self.agent = agent
        self.on_apply = on_apply
        self.on_clear_memory = on_clear_memory
        self.colors = colors
        self.ui_font = getattr(master, "ui_font", "Segoe UI")
        self.title("تنظیمات JARVIS")
        self.geometry("620x650")
        self.minsize(540, 600)
        self.configure(bg=colors["bg"])
        self.transient(master)
        self.grab_set()

        self.personality = tk.StringVar(value=agent.current_personality)
        self.performance = tk.StringVar(value=agent.get_setting("performance", "AUTO"))
        self.theme = tk.StringVar(value=agent.get_setting("theme", "CYBER"))
        self.language = tk.StringVar(value=agent.get_setting("language", "AUTO"))
        self.sampling = tk.StringVar(value=agent.get_setting("sampling_profile", "Balanced"))
        self.internet = tk.BooleanVar(value=agent.internet_enabled)
        self.memory = tk.BooleanVar(value=agent.memory_enabled)
        self.animation = tk.BooleanVar(
            value=agent.get_setting("animation", "true").casefold() == "true"
        )
        self.timestamps = tk.BooleanVar(
            value=agent.get_setting("timestamps", "false").casefold() == "true"
        )
        self._build()

    def _build(self) -> None:
        tk.Label(
            self, text="تنظیمات JARVIS", bg=self.colors["bg"], fg=self.colors["cyan"],
            font=(self.ui_font, 17, "bold")
        ).pack(anchor="e", padx=24, pady=(22, 4))
        tk.Label(
            self, text="تنظیمات محلی • تغییرها در SQLite ذخیره می‌شوند",
            bg=self.colors["bg"], fg=self.colors["muted"], font=(self.ui_font, 9)
        ).pack(anchor="e", padx=24, pady=(0, 18))

        form = tk.Frame(
            self, bg=self.colors["panel"], highlightbackground=self.colors["border"],
            highlightthickness=1
        )
        form.pack(fill="both", expand=True, padx=24, pady=(0, 16))
        form.columnconfigure(0, weight=1)
        row = 0
        row = self._choice(form, row, "شخصیت", self.personality, self.agent.personalities.available_names())
        row = self._choice(form, row, "کارایی", self.performance, ("AUTO", "LOW", "BALANCED", "ULTRA"))
        row = self._choice(form, row, "پروفایل تولید", self.sampling, ("Precise", "Balanced", "Creative"))
        row = self._choice(form, row, "پوسته", self.theme, ("CYBER", "MIDNIGHT"))
        row = self._choice(form, row, "زبان", self.language, ("AUTO", "PERSIAN", "ENGLISH"))
        row = self._toggle(form, row, "ابزار اینترنت", self.internet, "فقط برای جستجو/دریافت وب؛ چت اصلی محلی است")
        row = self._toggle(form, row, "حافظه", self.memory, "Context جلسه و Factهای مفید را نگه دارد")
        row = self._toggle(form, row, "انیمیشن", self.animation, "انیمیشن هستهٔ هولوگرافیک")
        self._toggle(form, row, "زمان پیام", self.timestamps, "زمان کوچکی کنار پیام‌ها نشان داده شود")

        actions = tk.Frame(self, bg=self.colors["bg"])
        actions.pack(fill="x", padx=24, pady=(0, 20))
        tk.Button(
            actions, text="پاک‌کردن حافظهٔ بلندمدت", command=self._confirm_clear,
            bg="#2A1720", fg="#FF8190", activebackground="#47202C",
            activeforeground="#FFFFFF", relief="flat", padx=14, pady=9,
            font=(self.ui_font, 9, "bold"), cursor="hand2"
        ).pack(side="left")
        tk.Button(
            actions, text="اعمال", command=self._apply,
            bg=self.colors["blue"], fg="#FFFFFF", activebackground=self.colors["cyan"],
            activeforeground="#081018", relief="flat", padx=24, pady=9,
            font=(self.ui_font, 9, "bold"), cursor="hand2"
        ).pack(side="right")
        tk.Button(
            actions, text="انصراف", command=self.destroy,
            bg=self.colors["panel"], fg=self.colors["text"], relief="flat",
            padx=18, pady=9, font=(self.ui_font, 9), cursor="hand2"
        ).pack(side="right", padx=(0, 8))

    def _choice(
        self, parent: tk.Frame, row: int, label: str, variable: tk.StringVar,
        values: tuple[str, ...]
    ) -> int:
        tk.Label(
            parent, text=label, bg=self.colors["panel"], fg=self.colors["text"],
            font=(self.ui_font, 10, "bold")
        ).grid(row=row, column=1, sticky="e", padx=18, pady=11)
        box = ttk.Combobox(
            parent, textvariable=variable, values=values,
            state="readonly", style="Jarvis.TCombobox"
        )
        box.grid(row=row, column=0, sticky="ew", padx=(18, 10), pady=11)
        return row + 1

    def _toggle(
        self, parent: tk.Frame, row: int, label: str, variable: tk.BooleanVar, hint: str
    ) -> int:
        block = tk.Frame(parent, bg=self.colors["panel"])
        block.grid(row=row, column=0, columnspan=2, sticky="ew", padx=14, pady=6)
        tk.Checkbutton(
            block, text=label, variable=variable, bg=self.colors["panel"],
            fg=self.colors["text"], activebackground=self.colors["panel"],
            activeforeground=self.colors["cyan"], selectcolor=self.colors["input"],
            font=(self.ui_font, 10, "bold"), anchor="e", justify="right"
        ).pack(anchor="e")
        tk.Label(
            block, text=hint, bg=self.colors["panel"], fg=self.colors["muted"],
            font=(self.ui_font, 8), justify="right"
        ).pack(anchor="e", padx=23)
        return row + 1

    def _apply(self) -> None:
        values: dict[str, str | bool] = {
            "performance": self.performance.get(),
            "theme": self.theme.get(),
            "language": self.language.get(),
            "sampling_profile": self.sampling.get(),
            "internet_enabled": self.internet.get(),
            "memory_enabled": self.memory.get(),
            "animation": self.animation.get(),
            "timestamps": self.timestamps.get(),
            "personality": self.personality.get(),
        }
        self.on_apply(values)
        self.destroy()

    def _confirm_clear(self) -> None:
        if messagebox.askyesno(
            "پاک‌کردن حافظه",
            "Factها و Context گفتگو پاک شوند؟ تنظیمات رابط حفظ می‌شوند.",
            parent=self,
        ):
            self.on_clear_memory()
