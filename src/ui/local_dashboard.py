from __future__ import annotations

import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from src.vision_pipeline import process_image


ROUTE_OPTIONS = ("A", "B", "C", "D")
STATUS_LABELS = {
    "idle": "\ub300\uae30 \uc911",
    "moving": "\ub85c\ubd07 \uc8fc\ud589 \uc911",
    "image_received": "\uc774\ubbf8\uc9c0 \ub3c4\ucc29",
    "analyzing": "\uc774\ubbf8\uc9c0 \ubd84\uc11d \uc911",
    "done": "\ubd84\uc11d \uc644\ub8cc",
    "error": "\uc624\ub958 \ubc1c\uc0dd",
}


@dataclass
class AnalysisResult:
    image_path: str = "-"
    predicted_status: str = "-"
    book_count: str = "-"
    lots: str = "-"


class LocalVisionDashboard:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Box Lot Vision Dashboard")
        self.root.geometry("1320x860")
        self.root.minsize(1180, 760)
        self.root.configure(bg="#f2efe8")

        self.selected_route = tk.StringVar(value="")
        self.progress_text = tk.StringVar(value="\uacbd\ub85c\ub97c \uc120\ud0dd\ud558\uc138\uc694.")
        self.stage_text = tk.StringVar(value=STATUS_LABELS["idle"])
        self.count_text = tk.StringVar(value="-")
        self.status_text = tk.StringVar(value="-")
        self.route_text = tk.StringVar(value="-")
        self.lot_text = tk.StringVar(value="-")
        self.result = AnalysisResult()
        self.preview_image: ImageTk.PhotoImage | None = None

        self._configure_styles()
        self._build_layout()
        self._render_result(AnalysisResult())

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Panel.TFrame", background="#fbf8f3")
        style.configure("Shell.TFrame", background="#f2efe8")
        style.configure("Title.TLabel", background="#f2efe8", foreground="#172121", font=("Malgun Gothic", 24, "bold"))
        style.configure("Subtitle.TLabel", background="#f2efe8", foreground="#516060", font=("Malgun Gothic", 11))
        style.configure("Section.TLabel", background="#fbf8f3", foreground="#172121", font=("Malgun Gothic", 15, "bold"))
        style.configure("Body.TLabel", background="#fbf8f3", foreground="#2b3a3a", font=("Malgun Gothic", 11))
        style.configure("MetricLabel.TLabel", background="#fbf8f3", foreground="#6a7474", font=("Malgun Gothic", 10))
        style.configure("MetricValue.TLabel", background="#fbf8f3", foreground="#172121", font=("Malgun Gothic", 20, "bold"))
        style.configure("Stage.TLabel", background="#d7efe2", foreground="#14532d", font=("Malgun Gothic", 11, "bold"), padding=(12, 8))
        style.configure("Primary.TButton", background="#c56b32", foreground="#fffaf2", font=("Malgun Gothic", 11, "bold"), padding=(12, 10), borderwidth=0)
        style.map("Primary.TButton", background=[("active", "#aa5927")])
        style.configure("Secondary.TButton", background="#1f4b4b", foreground="#f6f4ef", font=("Malgun Gothic", 10, "bold"), padding=(12, 10), borderwidth=0)
        style.map("Secondary.TButton", background=[("active", "#173838")])
        style.configure("Route.TButton", background="#ece4d8", foreground="#172121", font=("Malgun Gothic", 16, "bold"), padding=(20, 16), borderwidth=0)
        style.map("Route.TButton", background=[("active", "#ddcfbb")])
        style.configure("SelectedRoute.TButton", background="#1f4b4b", foreground="#f7f4ed", font=("Malgun Gothic", 16, "bold"), padding=(20, 16), borderwidth=0)
        style.map("SelectedRoute.TButton", background=[("active", "#173838")])

    def _build_layout(self) -> None:
        shell = ttk.Frame(self.root, style="Shell.TFrame", padding=24)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=5)
        shell.columnconfigure(1, weight=4)
        shell.rowconfigure(1, weight=1)

        header = ttk.Frame(shell, style="Shell.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 18))
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Box Lot Vision Dashboard", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="\uacbd\ub85c \uc120\ud0dd\ubd80\ud130 \uc774\ubbf8\uc9c0 \ub3c4\ucc29, LOT \ubd84\uc11d \uacb0\uacfc\uae4c\uc9c0 \ud55c \ud654\uba74\uc5d0\uc11c \ud655\uc778\ud569\ub2c8\ub2e4.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

        left_panel = ttk.Frame(shell, style="Panel.TFrame", padding=22)
        left_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 16))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(3, weight=1)

        right_panel = ttk.Frame(shell, style="Panel.TFrame", padding=22)
        right_panel.grid(row=1, column=1, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(2, weight=1)

        self._build_route_section(left_panel)
        self._build_progress_section(left_panel)
        self._build_result_section(left_panel)
        self._build_preview_section(right_panel)

    def _build_route_section(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="1. \uacbd\ub85c \uc120\ud0dd", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            parent,
            text="\ud55c \ubc88\uc5d0 \ud558\ub098\uc758 \uc704\uce58\ub97c \uc120\ud0dd\ud558\uace0, \ub85c\uceec \uc774\ubbf8\uc9c0\ub85c \ub3c4\ucc29 \uc0c1\ud669\uc744 \uc2dc\ubbac\ub808\uc774\uc158\ud569\ub2c8\ub2e4.",
            style="Body.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(6, 14))

        button_row = ttk.Frame(parent, style="Panel.TFrame")
        button_row.grid(row=2, column=0, sticky="ew")
        for column, route in enumerate(ROUTE_OPTIONS):
            button_row.columnconfigure(column, weight=1)
            button = ttk.Button(button_row, text=route, style="Route.TButton", command=lambda value=route: self._select_route(value))
            button.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 10, 0))
            setattr(self, f"route_button_{route}", button)

    def _build_progress_section(self, parent: ttk.Frame) -> None:
        progress_panel = ttk.Frame(parent, style="Panel.TFrame")
        progress_panel.grid(row=3, column=0, sticky="ew", pady=(24, 18))
        progress_panel.columnconfigure(0, weight=1)

        ttk.Label(progress_panel, text="2. \uc9c4\ud589 \uc0c1\ud0dc", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(progress_panel, textvariable=self.stage_text, style="Stage.TLabel").grid(row=1, column=0, sticky="w", pady=(12, 10))
        ttk.Label(progress_panel, textvariable=self.progress_text, style="Body.TLabel", wraplength=560, justify="left").grid(
            row=2, column=0, sticky="w", pady=(0, 16)
        )

        action_row = ttk.Frame(progress_panel, style="Panel.TFrame")
        action_row.grid(row=3, column=0, sticky="w")

        self.arrival_button = ttk.Button(
            action_row,
            text="\ub3c4\ucc29 \uc774\ubbf8\uc9c0 \ubd88\ub7ec\uc624\uae30",
            style="Primary.TButton",
            command=self._choose_arrived_image,
        )
        self.arrival_button.grid(row=0, column=0, sticky="w")

        ttk.Button(
            action_row,
            text="\ucd08\uae30\ud654",
            style="Secondary.TButton",
            command=self._reset_dashboard,
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))

    def _build_result_section(self, parent: ttk.Frame) -> None:
        result_panel = ttk.Frame(parent, style="Panel.TFrame")
        result_panel.grid(row=4, column=0, sticky="nsew")
        for column in range(3):
            result_panel.columnconfigure(column, weight=1)

        ttk.Label(result_panel, text="3. \ubd84\uc11d \uacb0\uacfc", style="Section.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")

        self._metric_card(result_panel, 1, 0, "\uc7ac\uace0 \uac1c\uc218", self.count_text)
        self._metric_card(result_panel, 1, 1, "\uc7ac\uace0 \uc0c1\ud0dc", self.status_text)
        self._metric_card(result_panel, 1, 2, "\uc120\ud0dd \uacbd\ub85c", self.route_text)

        lot_panel = ttk.Frame(result_panel, style="Panel.TFrame", padding=(0, 18, 0, 0))
        lot_panel.grid(row=2, column=0, columnspan=3, sticky="ew")
        lot_panel.columnconfigure(0, weight=1)

        ttk.Label(lot_panel, text="\uc778\uc2dd\ub41c LOT \ubc88\ud638", style="MetricLabel.TLabel").grid(row=0, column=0, sticky="w")
        self.lot_value = ttk.Label(lot_panel, style="MetricValue.TLabel", textvariable=self.lot_text, wraplength=760, justify="left")
        self.lot_value.grid(row=1, column=0, sticky="w", pady=(8, 0))

    def _build_preview_section(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="\ub3c4\ucc29 \uc774\ubbf8\uc9c0", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(parent, text="\ub85c\ubd07\uc774 \ub3c4\ucc29 \ud6c4 \ubcf4\ub0b8 \uc774\ubbf8\uc9c0\ub97c \uc774 \uc601\uc5ed\uc5d0 \ud45c\uc2dc\ud569\ub2c8\ub2e4.", style="Body.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 14))

        self.preview_card = tk.Frame(parent, bg="#e5ded2", highlightthickness=0)
        self.preview_card.grid(row=2, column=0, sticky="nsew")
        self.preview_card.grid_rowconfigure(0, weight=1)
        self.preview_card.grid_columnconfigure(0, weight=1)

        self.image_label = tk.Label(
            self.preview_card,
            text="\uc544\uc9c1 \ub3c4\ucc29\ud55c \uc774\ubbf8\uc9c0\uac00 \uc5c6\uc2b5\ub2c8\ub2e4.",
            bg="#e5ded2",
            fg="#4f5b5b",
            font=("Malgun Gothic", 13),
            justify="center",
        )
        self.image_label.grid(row=0, column=0, sticky="nsew")

        self.image_meta = ttk.Label(parent, style="Body.TLabel", wraplength=460, justify="left")
        self.image_meta.grid(row=3, column=0, sticky="w", pady=(12, 0))

    def _metric_card(self, parent: ttk.Frame, row: int, column: int, title: str, value_var: tk.StringVar) -> None:
        card = tk.Frame(parent, bg="#f6f2ea", padx=16, pady=16)
        card.grid(row=row, column=column, sticky="nsew", padx=(0 if column == 0 else 10, 0), pady=(14, 0))
        card.grid_columnconfigure(0, weight=1)

        ttk.Label(card, text=title, style="MetricLabel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(card, textvariable=value_var, style="MetricValue.TLabel").grid(row=1, column=0, sticky="w", pady=(8, 0))

    def _select_route(self, route: str) -> None:
        self.selected_route.set(route)
        self.stage_text.set(STATUS_LABELS["moving"])
        self.progress_text.set(f"{route} \uacbd\ub85c\uac00 \uc120\ud0dd\ub418\uc5c8\uc2b5\ub2c8\ub2e4. \uc2e4\uc81c \uc5f0\ub3d9 \uc2dc \uc774 \uc9c0\uc810\uc5d0\uc11c ROS2\ub85c \uc8fc\ud589 \uba85\ub839\uc744 \ubcf4\ub0c5\ub2c8\ub2e4.")
        for option in ROUTE_OPTIONS:
            button: ttk.Button = getattr(self, f"route_button_{option}")
            button.configure(style="SelectedRoute.TButton" if option == route else "Route.TButton")
        self.route_text.set(route)

    def _choose_arrived_image(self) -> None:
        if not self.selected_route.get():
            messagebox.showwarning("\uacbd\ub85c \uc120\ud0dd \ud544\uc694", "\uba3c\uc800 A, B, C, D \uc911 \ud558\ub098\uc758 \uacbd\ub85c\ub97c \uc120\ud0dd\ud558\uc138\uc694.")
            return

        image_path = filedialog.askopenfilename(
            title="\ub3c4\ucc29 \uc774\ubbf8\uc9c0\ub97c \uc120\ud0dd\ud558\uc138\uc694",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")],
        )
        if not image_path:
            return

        resolved_path = Path(image_path)
        self._show_preview(resolved_path)
        self.stage_text.set(STATUS_LABELS["image_received"])
        self.progress_text.set(f"{self.selected_route.get()} \uc704\uce58\uc5d0\uc11c \uc774\ubbf8\uc9c0\uac00 \ub3c4\ucc29\ud588\uc2b5\ub2c8\ub2e4. \ubd84\uc11d\uc744 \uc2dc\uc791\ud569\ub2c8\ub2e4.")
        self.image_meta.configure(text=f"\uc218\uc2e0 \ud30c\uc77c: {resolved_path.name}")
        threading.Thread(target=self._run_analysis, args=(resolved_path,), daemon=True).start()

    def _run_analysis(self, image_path: Path) -> None:
        self.root.after(0, lambda: self.stage_text.set(STATUS_LABELS["analyzing"]))
        self.root.after(0, lambda: self.progress_text.set("\ube44\uc804 \ud30c\uc774\ud504\ub77c\uc778\uc744 \uc2e4\ud589\ud574 \uc7ac\uace0 \uc0c1\ud0dc\uc640 LOT \ubc88\ud638\ub97c \uc77d\ub294 \uc911\uc785\ub2c8\ub2e4."))

        try:
            analysis = process_image(image_path=image_path)
            lots = str(analysis.get("pred_lots", "UNKNOWN")).replace("|", ", ")
            result = AnalysisResult(
                image_path=str(image_path),
                predicted_status=str(analysis.get("pred_status", "unknown")),
                book_count=str(analysis.get("pred_count", "-")),
                lots=lots or "UNKNOWN",
            )
        except Exception as exc:
            self.root.after(0, lambda: self._handle_analysis_error(exc))
            return

        self.root.after(0, lambda: self._apply_analysis_result(result))

    def _handle_analysis_error(self, exc: Exception) -> None:
        self.stage_text.set(STATUS_LABELS["error"])
        self.progress_text.set("\uc774\ubbf8\uc9c0\ub97c \ubd84\uc11d\ud558\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4. \ubaa8\ub378 \uacbd\ub85c \ub610\ub294 \uc758\uc874\uc131\uc744 \ud655\uc778\ud558\uc138\uc694.")
        self.status_text.set("error")
        self.count_text.set("-")
        self.lot_text.set("-")
        self.image_meta.configure(text=f"\ubd84\uc11d \uc2e4\ud328: {exc}")

    def _apply_analysis_result(self, result: AnalysisResult) -> None:
        self.result = result
        self.stage_text.set(STATUS_LABELS["done"])
        self.progress_text.set(f"{self.selected_route.get()} \uc704\uce58 \ubd84\uc11d\uc774 \uc644\ub8cc\ub418\uc5c8\uc2b5\ub2c8\ub2e4. \uacb0\uacfc\ub97c \ud655\uc778\ud558\uc138\uc694.")
        self._render_result(result)
        self.image_meta.configure(text=f"\uc218\uc2e0 \ud30c\uc77c: {Path(result.image_path).name}")

    def _render_result(self, result: AnalysisResult) -> None:
        self.count_text.set(result.book_count)
        self.status_text.set(result.predicted_status)
        self.route_text.set(self.selected_route.get() or "-")
        self.lot_text.set(result.lots)

    def _show_preview(self, image_path: Path) -> None:
        with Image.open(image_path) as image:
            corrected = ImageOps.exif_transpose(image).convert("RGB")
            preview = corrected.copy()

        preview.thumbnail((560, 560))
        self.preview_image = ImageTk.PhotoImage(preview)
        self.image_label.configure(image=self.preview_image, text="")

    def _reset_dashboard(self) -> None:
        self.selected_route.set("")
        self.stage_text.set(STATUS_LABELS["idle"])
        self.progress_text.set("\uacbd\ub85c\ub97c \uc120\ud0dd\ud558\uc138\uc694.")
        self.result = AnalysisResult()
        self._render_result(self.result)
        self.image_label.configure(image="", text="\uc544\uc9c1 \ub3c4\ucc29\ud55c \uc774\ubbf8\uc9c0\uac00 \uc5c6\uc2b5\ub2c8\ub2e4.")
        self.preview_image = None
        self.image_meta.configure(text="")
        for option in ROUTE_OPTIONS:
            button: ttk.Button = getattr(self, f"route_button_{option}")
            button.configure(style="Route.TButton")


def run_local_dashboard() -> None:
    root = tk.Tk()
    LocalVisionDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    run_local_dashboard()
