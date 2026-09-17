from __future__ import annotations

import tkinter as tk
from collections.abc import Callable


class CommandPalette(tk.Toplevel):
    def __init__(
        self,
        master: tk.Misc,
        commands: list[tuple[str, Callable[[], None]]],
        colors: dict[str, str],
    ) -> None:
        super().__init__(master)
        self.commands = commands
        self.filtered = list(commands)
        self.colors = colors
        self.title("Jarvis Command Palette")
        self.geometry("520x370")
        self.configure(bg=colors["panel"])
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        self.bind("<Escape>", lambda _event: self.destroy())

        tk.Label(
            self, text="COMMAND PALETTE", bg=colors["panel"], fg=colors["cyan"],
            font=("Segoe UI Semibold", 11)
        ).pack(anchor="w", padx=18, pady=(16, 8))
        self.query = tk.StringVar()
        entry = tk.Entry(
            self, textvariable=self.query, bg=colors["input"], fg=colors["text"],
            insertbackground=colors["cyan"], relief="flat", font=("Segoe UI", 11)
        )
        entry.pack(fill="x", padx=18, ipady=9)
        self.listbox = tk.Listbox(
            self, bg=colors["panel"], fg=colors["text"],
            selectbackground=colors["blue"], selectforeground="#FFFFFF",
            relief="flat", highlightthickness=0, font=("Segoe UI", 10),
            activestyle="none"
        )
        self.listbox.pack(fill="both", expand=True, padx=18, pady=12)
        self.query.trace_add("write", lambda *_: self._filter())
        entry.bind("<Down>", lambda _event: self.listbox.focus_set())
        entry.bind("<Return>", lambda _event: self._run_selected())
        self.listbox.bind("<Double-Button-1>", lambda _event: self._run_selected())
        self.listbox.bind("<Return>", lambda _event: self._run_selected())
        self._render()
        entry.focus_set()

    def _filter(self) -> None:
        query = self.query.get().casefold().strip()
        self.filtered = [item for item in self.commands if query in item[0].casefold()]
        self._render()

    def _render(self) -> None:
        self.listbox.delete(0, "end")
        for label, _callback in self.filtered:
            self.listbox.insert("end", f"  {label}")
        if self.filtered:
            self.listbox.selection_set(0)

    def _run_selected(self) -> None:
        selection = self.listbox.curselection()
        if not selection or not self.filtered:
            return
        callback = self.filtered[selection[0]][1]
        master = self.master
        self.destroy()
        master.after_idle(callback)
