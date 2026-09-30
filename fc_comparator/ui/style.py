"""Colours, fonts and ttk styles (plain tkinter/ttk, 'clam' theme)."""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

BG = "#f3f4f6"
SURFACE = "#ffffff"
BORDER = "#d1d5db"
TEXT = "#111827"
MUTED = "#6b7280"
HEADER = "#111827"
HEADER_TEXT = "#f9fafb"
ACCENT = "#1d4ed8"
ACCENT_DARK = "#1e40af"
OK = "#15803d"
NG = "#b91c1c"
WARN = "#b45309"
IDLE = "#4b5563"
CANVAS_BG = "#1f2937"

# Box colours per clip type on images (BGR is not used here: Tk wants #rrggbb).
CLASS_COLORS = ["#2563eb", "#16a34a", "#ea580c", "#9333ea", "#0891b2", "#db2777", "#65a30d", "#ca8a04", "#4f46e5"]


def class_color(classes: list[str], label: str) -> str:
    return CLASS_COLORS[classes.index(label) % len(CLASS_COLORS)] if label in classes else "#6b7280"


def font(size: int = 10, weight: str = "normal") -> tuple:
    return ("Segoe UI", size, weight)


MONO = ("Consolas", 10)


def apply(root: tk.Tk) -> None:
    root.configure(bg=BG)
    base = tkfont.nametofont("TkDefaultFont")
    base.configure(family="Segoe UI", size=10)
    for name in ("TkTextFont", "TkMenuFont", "TkHeadingFont"):
        tkfont.nametofont(name).configure(family="Segoe UI", size=10)
    root.option_add("*TCombobox*Listbox.font", font(11))

    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".", background=BG, foreground=TEXT, font=font(10))
    s.configure("TFrame", background=BG)
    s.configure("Card.TFrame", background=SURFACE, relief="solid", borderwidth=1, bordercolor=BORDER)
    s.configure("Header.TFrame", background=HEADER)
    s.configure("TLabel", background=BG, foreground=TEXT)
    s.configure("Card.TLabel", background=SURFACE)
    s.configure("Muted.TLabel", foreground=MUTED)
    s.configure("CardMuted.TLabel", background=SURFACE, foreground=MUTED)
    s.configure("Title.TLabel", font=font(16, "bold"))
    s.configure("CardTitle.TLabel", background=SURFACE, font=font(12, "bold"))
    s.configure("Header.TLabel", background=HEADER, foreground=HEADER_TEXT, font=font(14, "bold"))
    s.configure("HeaderMuted.TLabel", background=HEADER, foreground="#9ca3af", font=font(10))
    s.configure("TLabelframe", background=BG, bordercolor=BORDER)
    s.configure("TLabelframe.Label", background=BG, foreground=TEXT, font=font(10, "bold"))

    s.configure("TButton", padding=(12, 6), font=font(10), background="#e5e7eb", bordercolor=BORDER)
    s.map("TButton", background=[("active", "#d1d5db"), ("disabled", "#f3f4f6")])
    s.configure("Accent.TButton", background=ACCENT, foreground="#ffffff", bordercolor=ACCENT_DARK, font=font(10, "bold"))
    s.map("Accent.TButton", background=[("active", ACCENT_DARK), ("disabled", "#93c5fd")],
          foreground=[("disabled", "#eff6ff")])
    s.configure("Big.TButton", background=ACCENT, foreground="#ffffff", bordercolor=ACCENT_DARK,
                font=font(16, "bold"), padding=(16, 14))
    s.map("Big.TButton", background=[("active", ACCENT_DARK), ("disabled", "#93c5fd")],
          foreground=[("disabled", "#eff6ff")])
    s.configure("Danger.TButton", background=NG, foreground="#ffffff", bordercolor="#7f1d1d", font=font(10, "bold"))
    s.map("Danger.TButton", background=[("active", "#991b1b")])
    s.configure("Header.TButton", background="#374151", foreground=HEADER_TEXT, bordercolor="#4b5563", padding=(10, 4))
    s.map("Header.TButton", background=[("active", "#4b5563")])

    s.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(8, 6, 8, 0))
    s.configure("TNotebook.Tab", padding=(18, 8), font=font(11), background="#e5e7eb")
    s.map("TNotebook.Tab", background=[("selected", SURFACE)], foreground=[("selected", ACCENT)],
          font=[("selected", font(11, "bold"))])

    s.configure("Treeview", rowheight=26, font=font(10), background=SURFACE, fieldbackground=SURFACE)
    s.configure("Treeview.Heading", font=font(10, "bold"), background="#e5e7eb")
    s.map("Treeview", background=[("selected", "#bfdbfe")], foreground=[("selected", TEXT)])
    s.configure("TEntry", padding=4)
    s.configure("TCombobox", padding=4)
    s.configure("Horizontal.TProgressbar", background=ACCENT)
