"""Login screen shown before the station UI (and again after logging out)."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from ..config import AppConfig
from ..station.security import verify_secret
from . import style


class LoginPage(ttk.Frame):
    def __init__(self, root: tk.Misc, cfg: AppConfig, on_login: Callable[[str], None], log_event=None):
        super().__init__(root)
        self.cfg = cfg
        self.on_login = on_login
        self.log_event = log_event or (lambda *_a: None)

        box = ttk.Frame(self, style="Card.TFrame", padding=(40, 32))
        box.place(relx=0.5, rely=0.45, anchor="center")
        ttk.Label(box, text="FC-Comparator", style="CardTitle.TLabel", font=style.font(22, "bold")).pack()
        ttk.Label(box, text=f"Clip inspection  ·  {cfg.station.name}", style="CardMuted.TLabel").pack(pady=(0, 20))

        ttk.Label(box, text="Username", style="Card.TLabel").pack(anchor="w")
        self.user = ttk.Entry(box, width=32, font=style.font(12))
        self.user.pack(fill="x", pady=(2, 10))
        ttk.Label(box, text="Password", style="Card.TLabel").pack(anchor="w")
        self.password = ttk.Entry(box, width=32, font=style.font(12), show="•")
        self.password.pack(fill="x", pady=(2, 4))
        self.error = ttk.Label(box, text="", style="Card.TLabel", foreground=style.NG)
        self.error.pack(anchor="w", pady=(0, 6))
        ttk.Button(box, text="Sign in", style="Accent.TButton", command=self.submit).pack(fill="x", ipady=4)

        self.user.bind("<Return>", lambda _e: self.password.focus_set())
        self.password.bind("<Return>", lambda _e: self.submit())

    def reset(self) -> None:
        self.user.delete(0, "end")
        self.password.delete(0, "end")
        self.error.configure(text="")
        self.after(50, self.user.focus_set)

    def check(self, user: str, password: str) -> bool:
        sec = self.cfg.security
        return user == sec.login_user and verify_secret(password, sec.login_password)

    def submit(self) -> bool:
        user = self.user.get().strip()
        if not self.check(user, self.password.get()):
            self.log_event("login_denied", user, "wrong username or password")
            self.password.delete(0, "end")
            self.error.configure(text="Wrong username or password.")
            self.password.focus_set()
            return False
        self.log_event("login", user, "")
        self.on_login(user)
        return True
