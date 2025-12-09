#!/usr/bin/env python3
"""
cvd_analyzer_gui.py
-------------------
CVD (color vision deficiency) simulation + binary frequency + delta operation stats.

Features:
- Open image
- Simulate Deuteranopia / Protanopia / Tritanopia (Machado 2009, DaltonLens)
- Adjustable severity (0–1)
- Before / After preview
- For AFTER image:
    * Per-channel (R/G/B) unique 8-bit frequencies
    * Sorted by #Occurrences (descending)
    * 00000000 excluded from listing and from % denominator
- Export per-channel frequency CSVs (binary + decimal)
- Export per-channel BEFORE→AFTER mapping CSVs
- Export per-channel delta operation stats CSVs:
    Operation = IDENTITY / ADDITION / SUBTRACTION,
    rows = delta value (e.g. +1, -2) with
           Count, % of Operation, % of Channel
- Show per-channel top-20 bar graphs
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import Counter
from daltonlens import simulate

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def image_to_8bit_binary(img: Image.Image):
    """Convert image to per-pixel (R,G,B) 8-bit binary strings."""
    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)
    rows = [(f"{r:08b}", f"{g:08b}", f"{b:08b}") for r, g, b in flat]
    return rows, h, w


def analyze_channel(rows, channel):
    """
    Count unique 8-bit strings in chosen channel (AFTER image).

    - Excludes ZERO from listing and from percentage denominator.
    - Returns:
        freq          : list of (binary_str, count, pct)
        total_all     : total entries including zeros
        total_nonzero : total entries excluding zeros (percent base)
        zero_count    : number of 00000000 entries
    """
    idx = {"R": 0, "G": 1, "B": 2}[channel]
    col = [r[idx] for r in rows]

    counts = Counter(col)
    total_all = sum(counts.values())
    zero_count = counts.get("00000000", 0)
    total_nonzero = total_all - zero_count

    freq = []
    base = total_nonzero if total_nonzero > 0 else 1

    # Sort descending by count, then by binary string
    for s, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if s == "00000000":
            continue
        pct = 100.0 * c / base
        freq.append((s, c, pct))

    return freq, total_all, total_nonzero, zero_count


# ---------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------

class CVDAnalyzerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CVD Filter + Binary Frequency Analyzer")
        self.root.geometry("1350x880")

        self.img_before = None
        self.img_after = None
        self.tk_before = None
        self.tk_after = None

        self.filter_mode = tk.StringVar(value="Deuteranopia")
        self.severity = tk.DoubleVar(value=1.0)
        self.image_path = tk.StringVar()

        # DaltonLens Machado 2009 simulator
        self.simulator = simulate.Simulator_Machado2009()

        self.freq_results = {}  # channel -> [(bin, count, pct), ...] for AFTER

        self.build_ui()
        self.root.mainloop()

    # ---------------------------------------------------------------
    def build_ui(self):
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=10)

        # Row 0: open + path
        tk.Button(top, text="Open Image", command=self.load_image).grid(row=0, column=0, padx=5)
        tk.Entry(top, textvariable=self.image_path, width=90).grid(row=0, column=1, columnspan=7, padx=5, sticky="we")

        # Row 1: filters, severity, exports
        for i, m in enumerate(("Deuteranopia", "Protanopia", "Tritanopia")):
            tk.Radiobutton(top, text=m, variable=self.filter_mode, value=m,
                           command=self.update_filter).grid(row=1, column=i, padx=5, sticky="w")

        tk.Label(top, text="Severity:").grid(row=1, column=3, sticky="e")
        tk.Scale(
            top, variable=self.severity, from_=0, to=1, resolution=0.1,
            orient="horizontal", command=lambda v: self.update_filter(),
            length=200
        ).grid(row=1, column=4, sticky="w")

        tk.Button(top, text="Export Frequency CSVs", command=self.export_freq_csvs)\
            .grid(row=1, column=5, padx=5)
        tk.Button(top, text="Export Mapping CSVs", command=self.export_mapping_csvs)\
            .grid(row=1, column=6, padx=5)
        tk.Button(top, text="Export Operation Stats", command=self.export_operation_stats)\
            .grid(row=1, column=7, padx=5)
        tk.Button(top, text="Show Graphs", command=self.show_graphs)\
            .grid(row=1, column=8, padx=5)

        # Image preview frame
        frame_imgs = tk.Frame(self.root)
        frame_imgs.pack(fill="both", expand=True, padx=10, pady=10)

        tk.Label(frame_imgs, text="Before (original)").grid(row=0, column=0)
        tk.Label(frame_imgs, text="After (CVD simulated)").grid(row=0, column=1)

        self.canvas_before = tk.Label(frame_imgs)
        self.canvas_before.grid(row=1, column=0, padx=5)
        self.canvas_after = tk.Label(frame_imgs)
        self.canvas_after.grid(row=1, column=1, padx=5)

        # Text output (full frequency table)
        self.txt = tk.Text(self.root, font=("Consolas", 10))
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------------------------------------------------------------
    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
        )
        if not path:
            return
        self.image_path.set(path)
        self.img_before = Image.open(path)
        self.update_filter()

    def simulate_cvd(self):
        """Run DaltonLens simulation for current mode + severity."""
        if not self.img_before:
            return None

        arr = np.array(self.img_before.convert("RGB"), dtype=np.uint8)
        mode = self.filter_mode.get()
        sev = float(self.severity.get())

        if mode == "Deuteranopia":
            out = self.simulator.simulate_cvd(arr, simulate.Deficiency.DEUTAN, severity=sev)
        elif mode == "Protanopia":
            out = self.simulator.simulate_cvd(arr, simulate.Deficiency.PROTAN, severity=sev)
        else:
            out = self.simulator.simulate_cvd(arr, simulate.Deficiency.TRITAN, severity=sev)

        return Image.fromarray(np.uint8(out))

    def update_filter(self):
        """Recompute AFTER image + frequency when filter/severity changes."""
        if not self.img_before:
            return
        self.img_after = self.simulate_cvd()
        self.display_images()
        self.compute_frequency()

    def display_images(self):
        if not self.img_before or not self.img_after:
            return
        w, h = 400, 300
        img_b = self.img_before.resize((w, h))
        img_a = self.img_after.resize((w, h))
        self.tk_before = ImageTk.PhotoImage(img_b)
        self.tk_after = ImageTk.PhotoImage(img_a)
        self.canvas_before.config(image=self.tk_before)
        self.canvas_after.config(image=self.tk_after)

    # ---------------------------------------------------------------
    def compute_frequency(self):
        """Compute per-channel frequency table for AFTER image."""
        if not self.img_after:
            return

        rows_after, _, _ = image_to_8bit_binary(self.img_after)

        self.freq_results = {}
        self.txt.delete("1.0", "end")

        for ch in ("R", "G", "B"):
            freq, total_all, total_nonzero, zero_count = analyze_channel(rows_after, ch)
            self.freq_results[ch] = freq

            self.txt.insert("end", f"\n=== {ch} CHANNEL ===\n")
            self.txt.insert("end", f"TOTAL entries (including zeros): {total_all}\n")
            self.txt.insert("end", f"TOTAL entries used for % (excl zeros): {total_nonzero}\n")
            self.txt.insert("end", f"Skipped 00000000: {zero_count}\n")
            self.txt.insert("end", f"{'Binary (8b)':<12} {'#Occur':>12} {'% of total (excl 0)':>22}\n")
            self.txt.insert("end", "-" * 48 + "\n")

            # show ALL unique strings
            for b, c, pct in freq:
                self.txt.insert("end", f"{b:<12} {c:>12} {pct:>20.2f}\n")

    # ---------------------------------------------------------------
    def export_freq_csvs(self):
        """Export per-channel frequency tables (binary + decimal)."""
        if not self.freq_results:
            messagebox.showinfo("No data", "Compute frequencies first (load image & choose filter).")
            return

        base = filedialog.asksaveasfilename(
            title="Export frequency CSVs (base name)",
            defaultextension=".csv"
        )
        if not base:
            return

        if base.lower().endswith(".csv"):
            base = base[:-4]

        mode = self.filter_mode.get().lower()
        sev = int(float(self.severity.get()) * 100)

        try:
            for ch, freq in self.freq_results.items():
                path_bin = f"{base}_{mode}_sev{sev:03d}_{ch}_binary.csv"
                path_dec = f"{base}_{mode}_sev{sev:03d}_{ch}_decimal.csv"

                # Binary CSV
                with open(path_bin, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["Binary(8b)", "Count", "% of total (excl 00000000)"])
                    for b, c, pct in freq:
                        w.writerow([b, c, f"{pct:.2f}"])

                # Decimal CSV
                with open(path_dec, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["Decimal", "Count", "% of total (excl 0)"])
                    for b, c, pct in freq:
                        w.writerow([int(b, 2), c, f"{pct:.2f}"])

            messagebox.showinfo("Done", "Exported per-channel frequency CSVs.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ---------------------------------------------------------------
    def export_mapping_csvs(self):
        """
        Export BEFORE→AFTER mapping for each channel as separate CSVs.

        For each channel (R/G/B) we produce:
            BEFORE (8b), AFTER (8b), Count, % of BEFORE(b)
        based on pixel-wise mapping between BEFORE and AFTER images.
        """
        if not self.img_before or not self.img_after:
            messagebox.showinfo("No data", "Load image and compute CVD first.")
            return

        rows_before, _, _ = image_to_8bit_binary(self.img_before)
        rows_after, _, _  = image_to_8bit_binary(self.img_after)

        if len(rows_before) != len(rows_after):
            messagebox.showerror("Size mismatch", "Before and After images differ in size.")
            return

        base = filedialog.asksaveasfilename(
            title="Export BEFORE→AFTER mapping CSVs (base name)",
            defaultextension=".csv"
        )
        if not base:
            return
        if base.lower().endswith(".csv"):
            base = base[:-4]

        mode = self.filter_mode.get().lower()
        sev = int(float(self.severity.get()) * 100)

        try:
            for ch in ("R", "G", "B"):
                idx = {"R": 0, "G": 1, "B": 2}[ch]
                b_list = [r[idx] for r in rows_before]
                a_list = [r[idx] for r in rows_after]

                pair_counts = Counter(zip(b_list, a_list))
                before_totals = Counter(b_list)

                path = f"{base}_{mode}_sev{sev:03d}_{ch}_mapping.csv"
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["BEFORE (8b)", "AFTER (8b)", "Count", "% of BEFORE(b)"])

                    # Iterate over BEFORE values in sorted order
                    for b in sorted(before_totals.keys()):
                        total_b = before_totals[b]
                        after_dict = {
                            a: c for (bb, a), c in pair_counts.items() if bb == b
                        }
                        for a, cnt in sorted(after_dict.items(),
                                             key=lambda x: (-x[1], x[0])):
                            pct = 100.0 * cnt / total_b if total_b else 0.0
                            w.writerow([b, a, cnt, f"{pct:.2f}%"])

            messagebox.showinfo("Done", "Exported BEFORE→AFTER mapping CSVs for all channels.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ---------------------------------------------------------------
    def compute_operation_stats(self):
        """
        Compute delta operation statistics BETWEEN BEFORE and AFTER.

        For each channel:
            - total_pixels
            - for each op in {IDENTITY, ADDITION, SUBTRACTION}:
                  total_count, pct_of_channel,
                  rows = list of (delta_value, count, % of op, % of channel)

        Includes assertions to ensure internal consistency.
        """
        if not self.img_before or not self.img_after:
            raise RuntimeError("Images not loaded")

        arr_before = np.array(self.img_before.convert("RGB"), dtype=np.int16)
        arr_after  = np.array(self.img_after.convert("RGB"), dtype=np.int16)

        if arr_before.shape != arr_after.shape:
            raise RuntimeError("Before and After images differ in size")

        delta = arr_after - arr_before  # shape (h, w, 3)
        h, w, _ = delta.shape
        total_pixels = h * w

        stats = {}

        for ch, idx in (("R", 0), ("G", 1), ("B", 2)):
            d = delta[:, :, idx].reshape(-1)
            counts = Counter(d)

            # Sanity: total_pixels must equal sum of all counts
            assert sum(counts.values()) == total_pixels, "Count mismatch in channel " + ch

            totals_op = {"IDENTITY": 0, "ADDITION": 0, "SUBTRACTION": 0}

            # First accumulate total per operation
            for val, cnt in counts.items():
                if val == 0:
                    op = "IDENTITY"
                elif val > 0:
                    op = "ADDITION"
                else:
                    op = "SUBTRACTION"
                totals_op[op] += cnt

            # Sanity: operations must partition the channel
            assert sum(totals_op.values()) == total_pixels, "Op totals mismatch in channel " + ch

            op_details = {}
            for op in ("IDENTITY", "ADDITION", "SUBTRACTION"):
                total_op = totals_op[op]
                rows = []
                if total_op > 0:
                    if op == "IDENTITY":
                        # Only delta 0
                        val = 0
                        cnt = counts.get(0, 0)
                        if cnt > 0:
                            pct_op = 100.0 * cnt / total_op
                            pct_ch = 100.0 * cnt / total_pixels
                            rows.append((val, cnt, pct_op, pct_ch))
                    else:
                        if op == "ADDITION":
                            iter_items = [(v, c) for v, c in counts.items() if v > 0]
                        else:
                            iter_items = [(v, c) for v, c in counts.items() if v < 0]

                        # sort by count descending, then by delta value
                        iter_items.sort(key=lambda x: (-x[1], x[0]))
                        for val, cnt in iter_items:
                            if cnt <= 0:
                                continue
                            pct_op = 100.0 * cnt / total_op
                            pct_ch = 100.0 * cnt / total_pixels
                            rows.append((val, cnt, pct_op, pct_ch))

                    # Sanity: sum of counts in rows must equal total_op
                    assert sum(r[1] for r in rows) == total_op, f"Row sum mismatch in {ch} {op}"

                op_details[op] = {
                    "total_count": total_op,
                    "pct_channel": 100.0 * total_op / total_pixels if total_pixels else 0.0,
                    "rows": rows,
                }

            stats[ch] = {
                "total_pixels": total_pixels,
                "operations": op_details,
            }

        return stats

    # ---------------------------------------------------------------
    def export_operation_stats(self):
        """
        Export per-channel delta operation stats as CSVs.

        For each channel (R/G/B) produce:
            <base>_<mode>_sevXXX_<ch>_ops.csv

        Each file contains sections:

            Channel, R
            Total pixels, N

            Operation, ADDITION
            Total Count, <n_add>, Percent of Channel, xx.xx%
            Value (delta), Count, % of Operation, % of Channel
            +1, 3763, 17.81%, 4.90%
            ...

        And similarly for SUBTRACTION and IDENTITY.
        """
        if not self.img_before or not self.img_after:
            messagebox.showinfo("No data", "Load image and compute CVD first.")
            return

        base = filedialog.asksaveasfilename(
            title="Export operation stats CSVs (base name)",
            defaultextension=".csv"
        )
        if not base:
            return
        if base.lower().endswith(".csv"):
            base = base[:-4]

        mode = self.filter_mode.get().lower()
        sev = int(float(self.severity.get()) * 100)

        try:
            stats = self.compute_operation_stats()

            for ch, ch_info in stats.items():
                path = f"{base}_{mode}_sev{sev:03d}_{ch}_ops.csv"
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)

                    total_pixels = ch_info["total_pixels"]
                    w.writerow(["Channel", ch])
                    w.writerow(["Total pixels", total_pixels])
                    w.writerow([])

                    for op in ("IDENTITY", "ADDITION", "SUBTRACTION"):
                        op_info = ch_info["operations"][op]
                        total_op = op_info["total_count"]
                        pct_ch_op = op_info["pct_channel"]

                        w.writerow(["Operation", op])
                        w.writerow(["Total Count", total_op, "Percent of Channel", f"{pct_ch_op:.2f}%"])
                        w.writerow(["Value (delta)", "Count", "% of Operation", "% of Channel"])

                        for val, cnt, pct_op, pct_ch in op_info["rows"]:
                            label = f"{val:+d}" if val != 0 else "0"
                            w.writerow([label, cnt, f"{pct_op:.2f}%", f"{pct_ch:.2f}%"])
                        w.writerow([])  # blank line between operations

            messagebox.showinfo("Done", "Exported per-channel operation stats CSVs.")
        except AssertionError as ae:
            messagebox.showerror("Assertion error", f"Internal consistency check failed:\n{ae}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ---------------------------------------------------------------
    def show_graphs(self):
        """Show bar plots of top-20 values for each channel (AFTER image)."""
        if not self.freq_results:
            messagebox.showinfo("No data", "Compute frequencies first.")
            return

        try:
            for ch, data in self.freq_results.items():
                if not data:
                    continue
                top_bins = [int(b, 2) for b, _, _ in data[:20]]
                top_counts = [c for _, c, _ in data[:20]]

                plt.figure()
                plt.bar(range(len(top_bins)), top_counts)
                plt.xticks(range(len(top_bins)), top_bins, rotation=45)
                plt.title(f"{ch} Channel - Top 20 values ({self.filter_mode.get()})")
                plt.xlabel("Decimal Value")
                plt.ylabel("Occurrences")

            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))


# ---------------------------------------------------------------------
if __name__ == "__main__":
    CVDAnalyzerGUI()
