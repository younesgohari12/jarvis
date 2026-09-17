from __future__ import annotations

import math
import re
import tkinter as tk
import webbrowser
from typing import Any, Callable

try:
    import customtkinter as ctk
except ImportError:  # import-safe self tests
    ctk = None  # type: ignore[assignment]


_BIDI_CONTROLS = "\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
_BIDI_TRANSLATION = str.maketrans("", "", _BIDI_CONTROLS)


class ModernChatSurface:
    """Holographic conversation surface rebuilt from scratch for JARVIS v18."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        ui_font: str,
        mono_font: str,
        colors: dict[str, str],
        clear_command: Callable[[], None],
    ) -> None:
        if ctk is None:
            raise RuntimeError("CustomTkinter is required for the JARVIS command deck")
        ctk.set_appearance_mode("dark")
        self.master = master
        self.ui_font = ui_font
        self.mono_font = mono_font
        self.colors = colors
        self.clear_command = clear_command
        self._row = 0
        self._text_widgets: list[tk.Text] = []

        self.frame = ctk.CTkFrame(
            master,
            fg_color="#071019",
            corner_radius=26,
            border_width=1,
            border_color="#173A4A",
        )
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        self._build_header()
        self.messages = ctk.CTkScrollableFrame(
            self.frame,
            fg_color="#071019",
            corner_radius=0,
            border_width=0,
            scrollbar_button_color="#153848",
            scrollbar_button_hover_color="#2ADDF6",
        )
        self.messages.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.messages.grid_columnconfigure(0, weight=1)
        self._build_empty_marker()

    @staticmethod
    def available() -> bool:
        return ctk is not None

    def grid(self, **kwargs: Any) -> None:
        self.frame.grid(**kwargs)

    def _build_header(self) -> None:
        head = ctk.CTkFrame(self.frame, fg_color="#09141F", corner_radius=22, height=54)
        head.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        head.grid_columnconfigure(1, weight=1)
        head.grid_propagate(False)

        badge = ctk.CTkFrame(head, fg_color="#0A2731", corner_radius=14, border_width=1, border_color="#1C6373")
        badge.grid(row=0, column=0, padx=(10, 8), pady=9)
        ctk.CTkLabel(
            badge, text="◈", text_color="#57F3FF", width=30,
            font=(self.ui_font, 16, "bold"),
        ).pack(padx=4, pady=2)

        title_box = ctk.CTkFrame(head, fg_color="transparent")
        title_box.grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(
            title_box, text="NEURAL CONVERSATION LINK", text_color="#DDFBFF",
            font=(self.mono_font, 10, "bold"), anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_box, text="secure local channel  •  bilingual cognition", text_color="#51798B",
            font=(self.mono_font, 7), anchor="w",
        ).pack(anchor="w")

        ctk.CTkButton(
            head, text="NEW LINK  +", command=self.clear_command,
            width=92, height=30, corner_radius=12,
            fg_color="#0D2330", hover_color="#123B4A",
            border_width=1, border_color="#1D5261",
            text_color="#82EDFF", font=(self.mono_font, 8, "bold"),
        ).grid(row=0, column=2, padx=10, pady=10)

    def _build_empty_marker(self) -> None:
        marker = ctk.CTkFrame(self.messages, fg_color="#08131D", corner_radius=18, border_width=1, border_color="#102B38")
        marker.grid(row=self._row, column=0, sticky="ew", padx=54, pady=(16, 10))
        marker.grid_columnconfigure(0, weight=1)
        self._row += 1
        ctk.CTkLabel(
            marker, text="JARVIS // COGNITIVE CHANNEL ONLINE", text_color="#2ABFD2",
            font=(self.mono_font, 8, "bold"),
        ).grid(row=0, column=0, pady=(10, 1))
        ctk.CTkLabel(
            marker, text="پیام بده؛ مغز محلی آمادهٔ پردازش است.", text_color="#678795",
            font=(self.ui_font, 9),
        ).grid(row=1, column=0, pady=(0, 10))

    @staticmethod
    def _visual_lines(text: str) -> int:
        total = 0
        in_code = False
        for line in str(text).splitlines() or [""]:
            if line.strip().startswith("```"):
                in_code = not in_code
                continue
            width = 92 if in_code else 74
            total += max(1, math.ceil(max(1, len(line)) / width))
        return max(2, min(34, total + 1))

    @staticmethod
    def _safe_url(value: str) -> str:
        return value.translate(_BIDI_TRANSLATION).strip(".,،؛)")

    @staticmethod
    def _clean_display(value: str) -> str:
        return str(value).translate(_BIDI_TRANSLATION)

    @staticmethod
    def _clean_copy(value: str) -> str:
        return str(value).translate(_BIDI_TRANSLATION)

    def _insert_rich(self, widget: tk.Text, text: str, language: str) -> None:
        body_tag = "rtl" if language == "fa" else "ltr"
        widget.tag_configure("rtl", justify="right", rmargin=10, lmargin1=10, lmargin2=10, spacing1=2, spacing3=5)
        widget.tag_configure("ltr", justify="left", rmargin=10, lmargin1=10, lmargin2=10, spacing1=2, spacing3=5)
        widget.tag_configure(
            "code", justify="left", font=(self.mono_font, 9), foreground="#C8FBFF", background="#050D14",
            lmargin1=14, lmargin2=14, rmargin=14, spacing1=5, spacing3=5,
        )
        widget.tag_configure("url", foreground="#5CEEFF", underline=True)

        def open_url(event: tk.Event[tk.Text]) -> str:
            index = widget.index(f"@{event.x},{event.y}")
            ranges = widget.tag_prevrange("url", f"{index}+1c")
            if ranges:
                url = self._safe_url(widget.get(ranges[0], ranges[1]))
                if url.startswith(("http://", "https://")):
                    webbrowser.open(url)
            return "break"

        widget.tag_bind("url", "<Button-1>", open_url)
        widget.tag_bind("url", "<Enter>", lambda _event: widget.configure(cursor="hand2"))
        widget.tag_bind("url", "<Leave>", lambda _event: widget.configure(cursor="xterm"))

        display_text = self._clean_display(text)
        for block in re.split(r"(```[\s\S]*?```)", display_text):
            if not block:
                continue
            if block.startswith("```") and block.endswith("```"):
                payload = block[3:-3]
                if "\n" in payload:
                    first, rest = payload.split("\n", 1)
                    if re.fullmatch(r"[\w.+-]{1,20}", first.strip()):
                        payload = rest
                widget.insert("end", f"{payload.rstrip()}\n", "code")
                continue
            cursor = 0
            for match in re.finditer(r"https?://[^\s<>]+", block):
                if match.start() > cursor:
                    widget.insert("end", block[cursor:match.start()], body_tag)
                url = match.group(0).rstrip(".,،؛)")
                widget.insert("end", url, (body_tag, "url"))
                cursor = match.start() + len(url)
            if cursor < len(block):
                widget.insert("end", block[cursor:], body_tag)

    def _context_menu(self, widget: tk.Text, event: tk.Event[tk.Text]) -> str:
        menu = tk.Menu(
            widget, tearoff=False, bg="#0B1B27", fg="#DDFBFF",
            activebackground="#12394A", activeforeground="#FFFFFF",
            relief="flat", borderwidth=0,
        )

        def copy_selection() -> None:
            try:
                selected = widget.get("sel.first", "sel.last")
            except tk.TclError:
                return
            self.master.clipboard_clear()
            self.master.clipboard_append(self._clean_copy(selected))

        def copy_all() -> None:
            self.master.clipboard_clear()
            self.master.clipboard_append(self._clean_copy(widget.get("1.0", "end-1c")))

        menu.add_command(label="کپی انتخاب", command=copy_selection)
        menu.add_command(label="کپی کل پیام", command=copy_all)
        menu.add_separator()
        menu.add_command(label="گفتگوی جدید", command=self.clear_command)
        menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def add_message(self, role: str, text: str, language: str, timestamp: str = "") -> None:
        is_user = role == "user"
        # Three-layer card: shadow -> rim -> content, creating depth without GPU cost.
        shell = ctk.CTkFrame(self.messages, fg_color="transparent")
        shell.grid(row=self._row, column=0, sticky="ew", padx=(100, 18) if is_user else (18, 100), pady=(8, 6))
        shell.grid_columnconfigure(0, weight=1)
        self._row += 1

        shadow = ctk.CTkFrame(shell, fg_color="#03080D", corner_radius=22, height=8)
        shadow.grid(row=1, column=0, sticky="ew", padx=(8, 0) if is_user else (0, 8), pady=(0, 0))

        rim = ctk.CTkFrame(
            shell,
            fg_color="#0B2231" if is_user else "#08231F",
            corner_radius=22,
            border_width=1,
            border_color="#245980" if is_user else "#1D665B",
        )
        rim.grid(row=0, column=0, sticky="ew", padx=(0, 8) if is_user else (8, 0))
        rim.grid_columnconfigure(0, weight=1)

        top = ctk.CTkFrame(rim, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 0))
        top.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            top,
            text="YOU" if is_user else "JARVIS",
            text_color="#8FB7FF" if is_user else "#68FFE0",
            font=(self.mono_font, 9, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            top,
            text=(timestamp or ("UPLINK" if is_user else "NEURAL RESPONSE")),
            text_color="#466477",
            font=(self.mono_font, 7),
        ).grid(row=0, column=1, sticky="e")

        body_bg = "#0A1C2A" if is_user else "#081B19"
        body = tk.Text(
            rim,
            height=self._visual_lines(text), wrap="word",
            bg=body_bg, fg="#EAFBFF",
            selectbackground="#2F69E8", selectforeground="#FFFFFF",
            insertbackground="#5CEEFF", relief="flat", borderwidth=0,
            highlightthickness=0, padx=11, pady=8,
            font=(self.ui_font, 11), cursor="xterm",
        )
        body.grid(row=1, column=0, sticky="ew", padx=8, pady=(7, 9))
        self._insert_rich(body, str(text).strip(), language)
        body.configure(state="disabled")

        def copy_selection(_event: tk.Event[tk.Text]) -> str:
            try:
                selected = body.get("sel.first", "sel.last")
            except tk.TclError:
                return "break"
            self.master.clipboard_clear()
            self.master.clipboard_append(self._clean_copy(selected))
            return "break"

        for sequence in ("<<Copy>>", "<Control-c>", "<Control-C>"):
            body.bind(sequence, copy_selection)
        for sequence in ("<Button-3>", "<Button-2>"):
            body.bind(sequence, lambda event, item=body: self._context_menu(item, event))
        self._text_widgets.append(body)
        self.scroll_to_bottom()

    def add_system(self, text: str, error: bool = False) -> None:
        card = ctk.CTkFrame(
            self.messages, fg_color="#180D15" if error else "#091722",
            corner_radius=14, border_width=1,
            border_color="#7C2743" if error else "#153746",
        )
        card.grid(row=self._row, column=0, sticky="ew", padx=110, pady=6)
        self._row += 1
        ctk.CTkLabel(
            card, text=str(text).strip(),
            text_color="#FF829B" if error else "#7896A5",
            font=(self.ui_font, 9, "bold" if error else "normal"),
            justify="right", anchor="e", wraplength=760,
        ).pack(fill="x", padx=14, pady=9)
        self.scroll_to_bottom()

    def add_attachment(self, name: str, kind: str, size_label: str) -> None:
        card = ctk.CTkFrame(
            self.messages, fg_color="#071E27", corner_radius=17,
            border_width=1, border_color="#1A6878",
        )
        card.grid(row=self._row, column=0, sticky="ew", padx=(160, 28), pady=7)
        self._row += 1
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=9)
        ctk.CTkLabel(row, text="▱", text_color="#5CEEFF", font=(self.ui_font, 17, "bold"), width=28).pack(side="left")
        labels = ctk.CTkFrame(row, fg_color="transparent")
        labels.pack(side="right", fill="x", expand=True)
        ctk.CTkLabel(labels, text=name, text_color="#DDFBFF", font=(self.ui_font, 10, "bold"), anchor="e").pack(fill="x")
        ctk.CTkLabel(labels, text=f"{kind}  •  {size_label}  •  ATTACHED", text_color="#5D8290", font=(self.mono_font, 7), anchor="e").pack(fill="x")
        self.scroll_to_bottom()

    def clear(self) -> None:
        for child in self.messages.winfo_children():
            child.destroy()
        self._text_widgets.clear()
        self._row = 0
        self._build_empty_marker()

    def selected_text(self) -> str:
        for widget in reversed(self._text_widgets):
            try:
                return self._clean_copy(widget.get("sel.first", "sel.last"))
            except tk.TclError:
                continue
        return ""

    def scroll_to_bottom(self) -> None:
        def move() -> None:
            canvas = getattr(self.messages, "_parent_canvas", None)
            if canvas is not None:
                canvas.yview_moveto(1.0)
        self.master.after_idle(move)


__all__ = ["ModernChatSurface"]
