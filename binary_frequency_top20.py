#!/usr/bin/env python3
"""
binary_string_frequency_full.py
--------------------------------
Analyze frequency of all unique 8-bit binary strings from a binary CSV file
(e.g., *_before_binary.csv or *_after_binary.csv).

Each channel (R/G/B) can be selected. Shows:
  - Binary (8b)
  - #Occurrences
  - % of total (unique)

Then exports the full frequency list to CSV if desired.
"""

import csv
import tkinter as tk
from tkinter import filedialog, messagebox
from collections import Counter
from typing import List, Tuple

# -----------------------------------------------------------------------------
# CSV reading helpers
# -----------------------------------------------------------------------------

def _clean_cell(x: str) -> str:
    """Trim quotes/whitespace."""
    x = (x or "").strip()
    if len(x) >= 2 and ((x[0] == x[-1] == '"') or (x[0] == x[-1] == "'")):
        x = x[1:-1]
    return x.strip()

def _is_8bit_bin(s: str) -> bool:
    """Return True if s is exactly 8 chars of 0/1."""
    return len(s) == 8 and set(s) <= {"0", "1"}

def _read_binary_csv_8bit(path: str) -> List[Tuple[str, str, str]]:
    """Read a binary CSV (R,G,B as 8-bit). Returns list of (r,g,b)."""
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        first = True
        for row in rdr:
            if not row or len(row) < 3:
                continue
            if first:
                head = ",".join(_clean_cell(x).lower() for x in row[:3])
                if head in ("r,g,b", "r, g, b"):
                    first = False
                    continue
                first = False
            r, g, b = (_clean_cell(c) for c in row[:3])
            if _is_8bit_bin(r) and _is_8bit_bin(g) and _is_8bit_bin(b):
                rows.append((r, g, b))
    return rows

# -----------------------------------------------------------------------------
# Core logic
# -----------------------------------------------------------------------------

def analyze_channel(rows: List[Tuple[str, str, str]], channel: str):
    """Return counts and percentages for the chosen channel."""
    if not rows:
        return [], 0

    if channel == "R":
        col = [r for (r, _, _) in rows]
    elif channel == "G":
        col = [g for (_, g, _) in rows]
    else:
        col = [b for (_, _, b) in rows]

    counts = Counter(col)
    total = sum(counts.values())

    # full list sorted descending by count
    freq_list = []
    for binary_str, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        pct = (cnt / total * 100.0) if total else 0.0
        freq_list.append((binary_str, cnt, pct))

    return freq_list, total

# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

class FrequencyGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Binary String Frequency (Full Unique List)")
        self.root.geometry("1000x700")

        # State
        self.csv_path = tk.StringVar()
        self.channel = tk.StringVar(value="R")
        self.freq_data = []
        self.total_count = 0

        # Top controls
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=10)

        tk.Label(top, text="Binary CSV:").grid(row=0, column=0, sticky="w")
        tk.Entry(top, textvariable=self.csv_path, width=80).grid(row=0, column=1, sticky="we", padx=6)
        tk.Button(top, text="Browse…", command=self.pick_csv).grid(row=0, column=2)

        tk.Label(top, text="Channel:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        for i, ch in enumerate(("R", "G", "B")):
            tk.Radiobutton(top, text=ch, variable=self.channel, value=ch).grid(row=1, column=i + 1, sticky="w", pady=(6, 0))

        tk.Button(top, text="Compute", command=self.compute).grid(row=1, column=4, padx=10)
        tk.Button(top, text="Export CSV", command=self.export_csv).grid(row=1, column=5)
        tk.Button(top, text="Quit", command=self.root.destroy).grid(row=1, column=6, padx=10)

        self.info = tk.StringVar(value="Select a CSV and channel, then click Compute.")
        tk.Label(self.root, textvariable=self.info, anchor="w").pack(fill="x", padx=10, pady=(0, 5))

        self.text = tk.Text(self.root, wrap="none", font=("Consolas", 10))
        self.text.pack(fill="both", expand=True, padx=10, pady=10)

        # Scrollbars
        scroll_y = tk.Scrollbar(self.text, orient="vertical", command=self.text.yview)
        scroll_x = tk.Scrollbar(self.text, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")

        self.root.mainloop()

    # -------------------------------------------------------------------------
    def pick_csv(self):
        p = filedialog.askopenfilename(
            title="Select binary CSV file",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if p:
            self.csv_path.set(p)

    # -------------------------------------------------------------------------
    def compute(self):
        path = self.csv_path.get().strip()
        if not path:
            messagebox.showwarning("Missing", "Please select a binary CSV file.")
            return
        try:
            rows = _read_binary_csv_8bit(path)
            data, total = analyze_channel(rows, self.channel.get())
            self.freq_data = data
            self.total_count = total

            self.text.delete("1.0", "end")
            if not data:
                self.text.insert("end", "No valid binary data found.\n")
                return

            header = f"{'Binary (8b)':<12} {'#Occurrences':>15} {'% of total (unique)':>20}\n"
            self.text.insert("end", header)
            self.text.insert("end", "-" * len(header) + "\n")

            for b, cnt, pct in data:
                self.text.insert("end", f"{b:<12} {cnt:>15} {pct:>20.2f}\n")

            self.text.insert("end", "-" * len(header) + "\n")
            self.text.insert("end", f"{self.channel.get()} TOTAL ENTRIES: {total:>10} {100.00:>20.2f}\n")

            self.info.set(f"Channel {self.channel.get()} | Unique binaries: {len(data)} | Total entries: {total}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # -------------------------------------------------------------------------
    def export_csv(self):
        if not self.freq_data:
            messagebox.showinfo("Nothing to export", "Compute results first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save frequency table as CSV",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Binary (8b)", "#Occurrences", "% of total (unique)"])
                for b, cnt, pct in self.freq_data:
                    w.writerow([b, cnt, f"{pct:.2f}"])
                w.writerow([])
                w.writerow([f"{self.channel.get()} TOTAL ENTRIES", self.total_count, "100.00"])
            messagebox.showinfo("Saved", f"Exported to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    FrequencyGUI()
