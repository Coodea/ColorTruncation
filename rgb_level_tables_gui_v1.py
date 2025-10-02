#!/usr/bin/env python3
# rgb_level_tables_gui.py
#
# Read BEFORE/AFTER "pixels CSV" and "binary TXT" files (from color_tool_gui_cvd_v7),
# compute a single RGB level (0..255) per file, and display two tables:
#   Pixels table  : BEFORE, AFTER, Δ(after−before)
#   Binary table  : BEFORE, AFTER, Δ(after−before)
#
# Notes:
# - "Pixels CSV" rows: R,G,B  (header 'R,G,B' tolerated)
# - "Binary TXT":      R G B R G B ... (decimal or hex tokens; line breaks, commas/semicolons OK)
# - Per-file RGB level is a statistic of all pixels per channel: mean (default), median, or RMS.
# - RGB levels are rounded to integers and clamped to 0..255 (only for BEFORE/AFTER rows).
# - Δ row shows signed differences (not clamped) so you can see the change.
#
# Run:  python rgb_level_tables_gui.py

import csv
import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

HEX_RE = re.compile(r'^(?:0x)?[0-9A-Fa-f]+$')

# -------------------- Readers --------------------

def read_pixels_csv(path: str) -> np.ndarray:
    """Read 'R,G,B' rows. Header tolerated. Returns Nx3 uint8 (RGB)."""
    vals = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            triple = []
            for cell in row[:3]:
                c = (cell or "").strip()
                # skip header tokens
                if c.lower() in ("r", "g", "b", "r,g,b"):
                    triple = []
                    break
                if c == "":
                    triple = []
                    break
                try:
                    v = int(float(c))
                except Exception:
                    triple = []
                    break
                # clamp to 0..255
                v = 0 if v < 0 else (255 if v > 255 else v)
                triple.append(v)
            if len(triple) == 3:
                vals.extend(triple)
    if not vals:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.asarray(vals, dtype=np.uint8).reshape(-1, 3)

def read_binary_txt(path: str) -> np.ndarray:
    """Read TXT with byte tokens: decimal or hex (FF/0xFF). Returns Nx3 uint8 (RGB)."""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        txt = f.read()
    for ch in [',', ';', '\t', '\r', '\n']:
        txt = txt.replace(ch, ' ')
    tokens = [t for t in txt.split() if t]

    values = []
    for t in tokens:
        v = None
        # decimal
        try:
            d = int(t, 10)
            if 0 <= d <= 255:
                v = d
        except Exception:
            pass
        # hex
        if v is None and HEX_RE.match(t):
            try:
                h = int(t, 16)
                if 0 <= h <= 255:
                    v = h
            except Exception:
                pass
        if v is not None:
            values.append(v)

    if not values:
        return np.zeros((0, 3), dtype=np.uint8)
    n = (len(values) // 3) * 3  # drop leftover 1–2 bytes
    return np.asarray(values[:n], dtype=np.uint8).reshape(-1, 3)

# -------------------- One-line RGB (0..255) --------------------

def rgb_single_level(rgbNx3: np.ndarray, stat: str = "mean") -> tuple[int, int, int]:
    """
    Compute a single RGB triple (0..255). stat ∈ {mean, median, rms}.
    Returns integers (rounded), each clamped to 0..255.
    """
    if rgbNx3.size == 0:
        return (0, 0, 0)
    R = rgbNx3[:, 0].astype(np.float64)
    G = rgbNx3[:, 1].astype(np.float64)
    B = rgbNx3[:, 2].astype(np.float64)

    if stat == "median":
        r = np.median(R); g = np.median(G); b = np.median(B)
    elif stat == "rms":
        r = np.sqrt(np.mean(R * R)); g = np.sqrt(np.mean(G * G)); b = np.sqrt(np.mean(B * B))
    else:
        r = np.mean(R); g = np.mean(G); b = np.mean(B)

    rr = int(np.clip(np.rint(r), 0, 255))
    gg = int(np.clip(np.rint(g), 0, 255))
    bb = int(np.clip(np.rint(b), 0, 255))
    return (rr, gg, bb)

def rgb_delta(after_rgb: tuple[int, int, int], before_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Signed channel deltas (after - before). Not clamped."""
    return (after_rgb[0] - before_rgb[0],
            after_rgb[1] - before_rgb[1],
            after_rgb[2] - before_rgb[2])

# -------------------- GUI --------------------

class RGBLevelApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RGB Levels (0–255) — Before/After Tables")
        self.root.geometry("860x520")

        # Paths
        self.pixels_before = tk.StringVar()
        self.pixels_after  = tk.StringVar()
        self.binary_before = tk.StringVar()
        self.binary_after  = tk.StringVar()

        # Stat
        self.stat = tk.StringVar(value="mean")  # mean | median | rms

        top = tk.Frame(self.root)
        top.pack(fill="both", expand=True, padx=12, pady=12)

        # --- File pickers ---
        # Pixels
        tk.Label(top, text="Pixels CSV (rows: R,G,B)", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(0,4))
        self._picker_row(top, "Before:", self.pixels_before, 1, [("CSV files","*.csv"), ("All files","*.*")])
        self._picker_row(top, "After:",  self.pixels_after,  2, [("CSV files","*.csv"), ("All files","*.*")])

        # Binary
        tk.Label(top, text="Binary TXT (flat: R G B ...)", font=("Segoe UI", 10, "bold")).grid(row=3, column=0, sticky="w", pady=(12,4))
        self._picker_row(top, "Before:", self.binary_before, 4, [("Text files","*.txt"), ("All files","*.*")])
        self._picker_row(top, "After:",  self.binary_after,  5, [("Text files","*.txt"), ("All files","*.*")])

        # Statistic choice
        stat_fr = tk.Frame(top)
        stat_fr.grid(row=6, column=0, columnspan=3, sticky="w", pady=(12, 6))
        tk.Label(stat_fr, text="Channel statistic:").pack(side="left", padx=(0,8))
        ttk.Radiobutton(stat_fr, text="Mean",   variable=self.stat, value="mean").pack(side="left", padx=4)
        ttk.Radiobutton(stat_fr, text="Median", variable=self.stat, value="median").pack(side="left", padx=4)
        ttk.Radiobutton(stat_fr, text="RMS",    variable=self.stat, value="rms").pack(side="left", padx=4)

        # Buttons
        btns = tk.Frame(top)
        btns.grid(row=7, column=0, columnspan=3, sticky="we", pady=(6, 12))
        ttk.Button(btns, text="Compute", command=self.compute).pack(side="left")
        ttk.Button(btns, text="Export Tables…", command=self.export_tables).pack(side="left", padx=10)
        ttk.Button(btns, text="Quit", command=self.root.destroy).pack(side="right")

        # --- Results: two tables ---
        tables = tk.Frame(self.root)
        tables.pack(fill="both", expand=True, padx=12, pady=(0,12))

        self.pix_group = ttk.LabelFrame(tables, text="Pixels (CSV)")
        self.bin_group = ttk.LabelFrame(tables, text="Binary (TXT)")
        self.pix_group.pack(side="left", fill="both", expand=True, padx=(0,6))
        self.bin_group.pack(side="left", fill="both", expand=True, padx=(6,0))

        self.pix_tree = self._make_tree(self.pix_group)
        self.bin_tree = self._make_tree(self.bin_group)

        # Stored last results for export
        self.last_pixels_rows = None  # list of (label, R, G, B)
        self.last_binary_rows = None

        self.root.mainloop()

    def _picker_row(self, parent, label, var, row, types):
        tk.Label(parent, text=label).grid(row=row, column=0, sticky="e")
        tk.Entry(parent, textvariable=var, width=70).grid(row=row, column=1, sticky="we", padx=6)
        tk.Button(parent, text="Browse…",
                  command=lambda v=var, t=types: self._browse(v, t)).grid(row=row, column=2, sticky="w")

    def _browse(self, var, types):
        path = filedialog.askopenfilename(title="Select file", filetypes=types)
        if path:
            var.set(path)

    def _make_tree(self, parent):
        tree = ttk.Treeview(parent, columns=("R","G","B"), show="headings", height=8)
        for col in ("R","G","B"):
            tree.heading(col, text=col)
            tree.column(col, width=80, anchor="center")
        tree.pack(fill="both", expand=True, padx=8, pady=8)
        return tree

    def _load_table_rows(self, before_path, after_path, reader, stat_name):
        """
        Returns rows: [(label, R, G, B), ...]
        label ∈ {"Before", "After", "Δ (after−before)"} ; RGB are ints (0..255 for Before/After; signed for Δ).
        """
        rows = []
        before_rgb = None
        after_rgb  = None

        if before_path:
            arr = reader(before_path)
            before_rgb = rgb_single_level(arr, stat=stat_name)
            rows.append(("Before", before_rgb[0], before_rgb[1], before_rgb[2]))

        if after_path:
            arr = reader(after_path)
            after_rgb = rgb_single_level(arr, stat=stat_name)
            rows.append(("After",  after_rgb[0], after_rgb[1], after_rgb[2]))

        if before_rgb is not None and after_rgb is not None:
            dr, dg, db = rgb_delta(after_rgb, before_rgb)
            rows.append(("Δ (after−before)", dr, dg, db))

        return rows

    def compute(self):
        stat_name = self.stat.get()

        # Pixels
        pix_rows = self._load_table_rows(self.pixels_before.get(), self.pixels_after.get(),
                                         reader=read_pixels_csv, stat_name=stat_name)
        self._fill_tree(self.pix_tree, pix_rows)
        self.last_pixels_rows = pix_rows if pix_rows else None

        # Binary
        bin_rows = self._load_table_rows(self.binary_before.get(), self.binary_after.get(),
                                         reader=read_binary_txt, stat_name=stat_name)
        self._fill_tree(self.bin_tree, bin_rows)
        self.last_binary_rows = bin_rows if bin_rows else None

        if not pix_rows and not bin_rows:
            messagebox.showwarning("No data", "Please select at least one BEFORE or AFTER file and click Compute.")
        else:
            messagebox.showinfo("Done", "Tables updated.")

    def _fill_tree(self, tree, rows):
        # clear
        for i in tree.get_children():
            tree.delete(i)
        # insert
        for label, r, g, b in rows:
            tree.insert("", "end", values=(f"{label}:  R={r}", f"G={g}", f"B={b}"))

    def export_tables(self):
        if not self.last_pixels_rows and not self.last_binary_rows:
            messagebox.showwarning("Nothing to export", "Compute tables first.")
            return
        outdir = filedialog.askdirectory(title="Choose folder to save CSV tables")
        if not outdir:
            return

        saved = []
        try:
            if self.last_pixels_rows:
                p_out = os.path.join(outdir, "pixels_rgb_levels.csv")
                self._write_levels_csv(p_out, self.last_pixels_rows)
                saved.append(p_out)
            if self.last_binary_rows:
                b_out = os.path.join(outdir, "binary_rgb_levels.csv")
                self._write_levels_csv(b_out, self.last_binary_rows)
                saved.append(b_out)
        except Exception as e:
            messagebox.showerror("Export error", str(e))
            return

        messagebox.showinfo("Exported", "Saved:\n" + "\n".join(saved))

    @staticmethod
    def _write_levels_csv(path, rows):
        # rows: [(label, R, G, B), ...]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["label", "R", "G", "B"])
            for label, r, g, b in rows:
                w.writerow([label, r, g, b])

if __name__ == "__main__":
    RGBLevelApp()
