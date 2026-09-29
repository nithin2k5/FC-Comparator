"""Colours, fonts and ttk styling shared by every screen (light and dark mode)."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import customtkinter as ctk

# (light, dark) pairs, CustomTkinter picks the right one for the appearance mode.
BG = ("#eef2f7", "#0f1623")
SURFACE = ("#ffffff", "#1a2332")
SURFACE_2 = ("#f6f8fb", "#222d3f")
BORDER = ("#dde3ec", "#2f3b50")
TEXT = ("#0f172a", "#e6ebf2")
MUTED = ("#64748b", "#94a3b8")
SIDEBAR = ("#101828", "#0a0f1a")
SIDEBAR_TEXT = "#cbd5e1"
SIDEBAR_ACTIVE = "#2563eb"
ACCENT = ("#2563eb", "#3b82f6")
ACCENT_HOVER = ("#1d4ed8", "#2563eb")
OK = "#16a34a"
OK_HOVER = "#15803d"
NG = "#dc2626"
NG_HOVER = "#b91c1c"
WARN = "#d97706"
IDLE = ("#475569", "#334155")
CANVAS_BG = "#0b1220"

CLASS_PALETTE = ["#2563eb", "#16a34a", "#ea580c", "#9333ea", "#0891b2", "#db2777", "#65a30d", "#ca8a04", "#4f46e5"]


def pick(color: str | tuple[str, str]) -> str:
    """Resolve a (light, dark) pair for plain-tk widgets."""
    if isinstance(color, tuple):
        return color[1] if ctk.get_appearance_mode() == "Dark" else color[0]
    return color


def font(size: int = 14, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(size=size, weight=weight)


def class_color(classes: list[str], label: str) -> str:
    if label in classes:
        return CLASS_PALETTE[classes.index(label) % len(CLASS_PALETTE)]
    return "#64748b"


def hex_to_bgr(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)


def style_ttk(root: tk.Misc) -> None:
    """Make ttk.Treeview (used for long lists/tables) match the CustomTkinter look."""
    style = ttk.Style(root)
    style.theme_use("clam")
    bg, fg, head = pick(SURFACE), pick(TEXT), pick(SURFACE_2)
    sel = pick(ACCENT)
    style.configure("Treeview", background=bg, fieldbackground=bg, foreground=fg, rowheight=34,
                    borderwidth=0, font=("Segoe UI", 11))
    style.map("Treeview", background=[("selected", sel)], foreground=[("selected", "#ffffff")])
    style.configure("Treeview.Heading", background=head, foreground=pick(MUTED), relief="flat",
                    font=("Segoe UI", 10, "bold"), padding=(8, 6))
    style.map("Treeview.Heading", background=[("active", head)])
    style.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
