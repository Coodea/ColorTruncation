#!/usr/bin/env python3
# rgb_levels_gui_table.py
#
# GUI that reads BEFORE/AFTER Pixel CSVs (decimal) and Binary files (TXT decimal/hex OR CSV with bit-strings)
# and shows tables like the screenshot:
#
#     R            G            B
#  Before: R=..  G=..        B=..
#  After:  R=..  G=..        B=..
#  Δ:      R=..  G=..        B=..
#
# Choose statistic: Mean / Median / RMS (values are 0..255; Δ may be negative).
#
# Requires: numpy, Pillow (Tk works without Pillow). Pillow not imported here.

import csv
import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Tuple

import numpy as np

# ---------- parsing helpers ----------

HEX_WORD = re.compile(r'^(?:0x)?[0-9A-Fa-f]+$')
BIN_WORD = re.compile(r'^[01]+$')
NUM_TOKEN = re.compile(r'\b(?:0x[0-9a-fA-F]{1,2}|0X[0-9a-fA-F]{1,2}|[0-9]{1,3})\b')
HEX_PAIR_BLOCK = re.compile(r'^[0-9A-Fa-f]+$')

def clamp_byte(v: int) -> int:
    return 0 if v < 0 else (255 if v > 255 else v)

def parse_cell_to_byte(cell: str) -> int:
    c = (cell or "").strip()
    if c == "":
        raise ValueError("empty")
    if BIN_WORD.match(c):       # "10101010"
        return clamp_byte(int(c, 2))
    if HEX_WORD.match(c):       # "FF" / "0x1a" / "1A"
        try:
            return clamp_byte(int(c, 16))
        except Exception:
            pass
    # decimal, including "123.0"
    return clamp_byte(int(float(c)))

# ---------- readers ----------

def read_pixels_csv_decimal(path: str) -> np.ndarray:
    """CSV rows: R,G,B in decimal. Header tolerated."""
    vals: List[int] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            # header?
            if ",".join(x.strip().lower() for x in row[:3]) in ("r,g,b", "r, g, b"):
                continue
            if len(row) < 3:
                continue
            try:
                r = clamp_byte(int(float(row[0])))
                g = clamp_byte(int(float(row[1])))
                b = clamp_byte(int(float(row[2])))
                vals.extend([r,g,b])
            except Exception:
                # ignore malformed rows
                continue
    if not vals:
        return np.zeros((0,3), dtype=np.uint8)
    return np.asarray(vals, dtype=np.uint8).reshape(-1,3)

def read_binary_any(path: str) -> np.ndarray:
    """
    Accept either:
      - TXT with tokens (decimal 0..255 or hex 0x?? / ??), possibly with commas/semicolons/newlines
      - CSV with cells that are decimal or 8-bit binary strings
    Returns Nx3 uint8 RGB; drops leftover 1–2 bytes.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        # parse each cell via parse_cell_to_byte (handles binary strings)
        vals: List[int] = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            rdr = csv.reader(f)
            for row in rdr:
                if not row:
                    continue
                if ",".join(x.strip().lower() for x in row[:3]) in ("r,g,b","r, g, b"):
                    continue
                if len(row) < 3:
                    continue
                try:
                    r = parse_cell_to_byte(row[0])
                    g = parse_cell_to_byte(row[1])
                    b = parse_cell_to_byte(row[2])
                    vals.extend([r,g,b])
                except Exception:
                    continue
        if not vals:
            return np.zeros((0,3), dtype=np.uint8)
        arr = np.asarray(vals, dtype=np.uint8)
        n = (arr.size // 3) * 3
        return arr[:n].reshape(-1,3)

    # TXT route: tolerant decimal/hex tokens; also handles continuous hex block
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    tokens = NUM_TOKEN.findall(raw)

    values: List[int] = []
    for t in tokens:
        if t.lower().startswith("0x"):
            v = int(t, 16)
        else:
            if re.search(r'[a-fA-F]', t):
                v = int(t, 16)
            else:
                v = int(t, 10)
        if 0 <= v <= 255:
            values.append(v)

    if not values:
        # try continuous hex like "FFA30C..."
        compact = "".join(ch for ch in raw if ch.strip())
        if HEX_PAIR_BLOCK.match(compact) and len(compact) >= 2:
            if len(compact) % 2 == 1:
                compact = compact[:-1]
            for i in range(0, len(compact), 2):
                values.append(int(compact[i:i+2], 16))

    if not values:
        return np.zeros((0,3), dtype=np.uint8)

    n = (len(values)//3)*3
    return np.asarray(values[:n], dtype=np.uint8).reshape(-1,3)

# ---------- single RGB (0..255) ----------

def rgb_single_level(rgbNx3: np.ndarray, stat: str) -> Tuple[int,int,int]:
    if rgbNx3.size == 0:
        return (0,0,0)
    R = rgbNx3[:,0].astype(np.float64)
    G = rgbNx3[:,1].astype(np.float64)
    B = rgbNx3[:,2].astype(np.float64)
    if stat == "median":
        r = np.median(R); g = np.median(G); b = np.median(B)
    elif stat == "rms":
        r = np.sqrt(np.mean(R*R)); g = np.sqrt(np.mean(G*G)); b = np.sqrt(np.mean(B*B))
    else:
        r = np.mean(R); g = np.mean(G); b = np.mean(B)
    rr = int(np.clip(np.rint(r), 0, 255))
    gg = int(np.clip(np.rint(g), 0, 255))
    bb = int(np.clip(np.rint(b), 0, 255))
    return (rr, gg, bb)

# ---------- GUI ----------

class RGBLevelTablesGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RGB Levels (0–255) — Pixels & Binary (Before/After)")
        self.root.geometry("880x560")

        # file paths
        self.px_before = tk.StringVar()
        self.px_after  = tk.StringVar()
        self.bn_before = tk.StringVar()
        self.bn_after  = tk.StringVar()
        # stat
        self.stat = tk.StringVar(value="mean")

        container = tk.Frame(self.root); container.pack(fill="both", expand=True, padx=12, pady=12)

        # Pixels picker
        tk.Label(container, text="Pixels CSV (rows: R,G,B)", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(0,4))
        self._picker_row(container, "Before:", self.px_before, 1, [("CSV files","*.csv"), ("All files","*.*")])
        self._picker_row(container, "After:",  self.px_after,  2, [("CSV files","*.csv"), ("All files","*.*")])

        # Binary picker
        tk.Label(container, text="Binary (TXT decimal/hex  OR  CSV bit-strings)", font=("Segoe UI", 10, "bold")).grid(row=3, column=0, sticky="w", pady=(12,4))
        self._picker_row(container, "Before:", self.bn_before, 4, [("Text/CSV","*.txt;*.csv"), ("Text","*.txt"), ("CSV","*.csv"), ("All files","*.*")])
        self._picker_row(container, "After:",  self.bn_after,  5, [("Text/CSV","*.txt;*.csv"), ("Text","*.txt"), ("CSV","*.csv"), ("All files","*.*")])

        # stat & buttons
        opts = tk.Frame(container); opts.grid(row=6, column=0, columnspan=3, sticky="we", pady=(10,6))
        tk.Label(opts, text="Channel statistic:").pack(side="left", padx=(0,8))
        ttk.Radiobutton(opts, text="Mean",   variable=self.stat, value="mean").pack(side="left", padx=4)
        ttk.Radiobutton(opts, text="Median", variable=self.stat, value="median").pack(side="left", padx=4)
        ttk.Radiobutton(opts, text="RMS",    variable=self.stat, value="rms").pack(side="left", padx=4)

        btns = tk.Frame(container); btns.grid(row=7, column=0, columnspan=3, sticky="we", pady=(6,12))
        ttk.Button(btns, text="Compute", command=self.compute).pack(side="left")
        ttk.Button(btns, text="Quit", command=self.root.destroy).pack(side="right")

        # Tables area
        tables = tk.Frame(self.root); tables.pack(fill="both", expand=True, padx=12, pady=(0,12))
        self.pix_group = ttk.LabelFrame(tables, text="Pixels (CSV)")
        self.bin_group = ttk.LabelFrame(tables, text="Binary (TXT/CSV)")
        self.pix_group.pack(side="left", fill="both", expand=True, padx=(0,6))
        self.bin_group.pack(side="left", fill="both", expand=True, padx=(6,0))

        self.pix_tree = self._make_tree(self.pix_group)
        self.bin_tree = self._make_tree(self.bin_group)

        self.root.mainloop()

    def _picker_row(self, parent, label, var, row, types):
        tk.Label(parent, text=label).grid(row=row, column=0, sticky="e")
        tk.Entry(parent, textvariable=var, width=70).grid(row=row, column=1, sticky="we", padx=6)
        tk.Button(parent, text="Browse…", command=lambda v=var, t=types: self._browse(v, t)).grid(row=row, column=2, sticky="w")

    def _browse(self, var, types):
        path = filedialog.askopenfilename(title="Select file", filetypes=types)
        if path:
            var.set(path)

    def _make_tree(self, parent):
        tree = ttk.Treeview(parent, columns=("R","G","B"), show="headings", height=8)
        for col in ("R","G","B"):
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor="center")
        tree.pack(fill="both", expand=True, padx=8, pady=8)
        return tree

    def _fill_table_like_screenshot(self, tree: ttk.Treeview, before, after):
        """
        Fill Treeview with 3 rows: Before, After, Δ — formatting like the screenshot.
        before/after are (R,G,B) tuples or None.
        """
        for i in tree.get_children():
            tree.delete(i)
        if before:
            r,g,b = before
            tree.insert("", "end", values=(f"Before:  R={r}", f"G={g}", f"B={b}"))
        if after:
            r,g,b = after
            tree.insert("", "end", values=(f"After:   R={r}", f"G={g}", f"B={b}"))
        if before and after:
            dr = after[0] - before[0]
            dg = after[1] - before[1]
            db = after[2] - before[2]
            tree.insert("", "end", values=(f"Δ (after−before):  R={dr}",
                                           f"G={dg}", f"B={db}"))

    def compute(self):
        # normalize paths and check existence
        px_b = self.px_before.get().strip()
        px_a = self.px_after.get().strip()
        bn_b = self.bn_before.get().strip()
        bn_a = self.bn_after.get().strip()
        def norm(p): return os.path.abspath(os.path.expanduser(p)) if p else ""
        px_b, px_a, bn_b, bn_a = map(norm, (px_b, px_a, bn_b, bn_a))

        if not any([px_b, px_a, bn_b, bn_a]):
            messagebox.showwarning("No input", "Select at least one BEFORE or AFTER file.")
            return

        for label, p in [("Pixels BEFORE", px_b), ("Pixels AFTER", px_a),
                         ("Binary BEFORE", bn_b), ("Binary AFTER", bn_a)]:
            if p and not os.path.isfile(p):
                messagebox.showerror("File not found", f"{label} does not exist:\n{p}")
                return

        stat = self.stat.get()

        # Pixels
        px_before = px_after = None
        try:
            if px_b:
                arr = read_pixels_csv_decimal(px_b)
                px_before = rgb_single_level(arr, stat)
            if px_a:
                arr = read_pixels_csv_decimal(px_a)
                px_after = rgb_single_level(arr, stat)
            self._fill_table_like_screenshot(self.pix_tree, px_before, px_after)
        except Exception as e:
            messagebox.showerror("Pixels error", str(e)); return

        # Binary
        bn_before_rgb = bn_after_rgb = None
        try:
            if bn_b:
                arr = read_binary_any(bn_b)
                bn_before_rgb = rgb_single_level(arr, stat)
            if bn_a:
                arr = read_binary_any(bn_a)
                bn_after_rgb = rgb_single_level(arr, stat)
            self._fill_table_like_screenshot(self.bin_tree, bn_before_rgb, bn_after_rgb)
        except Exception as e:
            messagebox.showerror("Binary error", str(e)); return

        messagebox.showinfo("Done", "Tables updated.")

if __name__ == "__main__":
    RGBLevelTablesGUI()
