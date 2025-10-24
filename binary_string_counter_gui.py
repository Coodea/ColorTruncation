#!/usr/bin/env python3
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
from collections import Counter, OrderedDict
from typing import List, Tuple
from PIL import Image
import matplotlib.pyplot as plt  # plotting

def _clean_cell(x: str) -> str:
    x = (x or "").strip()
    if len(x) >= 2 and ((x[0] == x[-1] == '"') or (x[0] == x[-1] == "'")):
        x = x[1:-1]
    return x.strip()

def _to_8bit(cell: str) -> str:
    c = _clean_cell(cell)
    if c.lower().startswith("0b"):
        c = c[2:]
    if len(c) == 8 and set(c) <= {"0","1"}:
        return c
    try:
        v = int(c)
        if 0 <= v <= 255:
            return f"{v:08b}"
    except Exception:
        pass
    return ""

def _read_binary_csv_8bit(path: str) -> Tuple[List[Tuple[str,str,str]], int]:
    rows, skipped = [], 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        first = True
        for row in rdr:
            if not row or len(row) < 3:
                skipped += 1
                continue
            if first:
                head = ",".join(_clean_cell(x).lower() for x in row[:3])
                if head in ("r,g,b","r, g, b"):
                    first = False
                    continue
                first = False
            r8 = _to_8bit(row[0])
            g8 = _to_8bit(row[1])
            b8 = _to_8bit(row[2])
            if r8 and g8 and b8:
                rows.append((r8,g8,b8))
            else:
                skipped += 1
    return rows, skipped

def channel_lists_and_first_seen(before_rows: List[Tuple[str,str,str]],
                                 after_rows:  List[Tuple[str,str,str]],
                                 ch: str):
    if ch == "R":
        b_list = [r for (r,_,_) in before_rows]
        a_list = [r for (r,_,_) in after_rows]
    elif ch == "G":
        b_list = [g for (_,g,_) in before_rows]
        a_list = [g for (_,g,_) in after_rows]
    else:
        b_list = [b for (_,_,b) in before_rows]
        a_list = [b for (_,_,b) in after_rows]

    first_seen = OrderedDict()
    for s in b_list:
        if s not in first_seen:
            first_seen[s] = None

    return b_list, a_list, list(first_seen.keys())

def most_common_mapping(pair_counts: Counter, b: str):
    opts = [(a,c) for (bb,a),c in pair_counts.items() if bb == b]
    if not opts:
        return "", 0, False
    maxc = max(c for _,c in opts)
    winners = sorted([a for a,c in opts if c == maxc])
    return winners[0], maxc, (len(winners) > 1)

def build_table(before_rows, after_rows, ch: str):
    b_list, a_list, first_seen_order = channel_lists_and_first_seen(before_rows, after_rows, ch)

    n = min(len(b_list), len(a_list))
    b_used = b_list[:n]
    a_used = a_list[:n]

    c_before    = Counter(b_used)
    c_after     = Counter(a_used)
    pair_counts = Counter(zip(b_used, a_used))

    identity_count = sum(pair_counts.get((x, x), 0) for x in c_before.keys())
    total_changed = n - identity_count
    pct_total_changed = (100.0 * total_changed / n) if n else 0.0

    top_pairs_all = [(b,a,c) for (b,a),c in pair_counts.items() if b != a]
    top_pairs_all.sort(key=lambda t: (-t[2], t[0], t[1]))

    table = []
    for b in first_seen_order:
        cb = c_before.get(b, 0)

        a_star, mapped_star, tied = most_common_mapping(pair_counts, b)

        ca_same = c_after.get(b, 0)
        delta_same = ca_same - cb

        pct_loss = (max(cb - ca_same, 0) / cb * 100.0) if cb else 0.0

        mapped_total = cb  # internal only

        table.append([
            b,          # 0 before binary string
            a_star,     # 1 most common mapped-to
            tied,       # 2 tie flag
            mapped_star,# 3 #Mapped→edited
            cb,         # 4 Count before
            ca_same,    # 5 Count after (same)
            delta_same, # 6 Δ same
            pct_loss,   # 7 % loss
            mapped_total# 8 internal
        ])

    totals = {
        "n_rows":            n,
        "identity_count":    identity_count,
        "total_changed":     total_changed,
        "pct_total_changed": pct_total_changed,
        "unique_before":     len(first_seen_order),
        "sum_before_counts": n,
        "sum_after_counts":  n,
        "pairs_used":        n,
        "top_pairs":         top_pairs_all,
    }

    return table, totals

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Unique 8-bit Counter + Edited Mapping (First-seen from BEFORE)")
        self.root.geometry("1240x840")

        self.image_path  = tk.StringVar()
        self.before_path = tk.StringVar()
        self.after_path  = tk.StringVar()

        self.channel      = tk.StringVar(value="R")
        self.sort_by      = tk.StringVar(value="FirstSeen")
        self.only_changes = tk.BooleanVar(value=False)
        self.hide_zero_ed = tk.BooleanVar(value=True)

        # --- file pickers row ---
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=(10,6))

        tk.Label(top, text="Image:").grid(row=0, column=0, sticky="w")
        tk.Entry(top, textvariable=self.image_path, width=95).grid(row=0, column=1, sticky="we", padx=6)
        tk.Button(top, text="Browse…", command=self.pick_image).grid(row=0, column=2)

        tk.Label(top, text="Before (binary CSV):").grid(row=1, column=0, sticky="w", pady=(6,0))
        tk.Entry(top, textvariable=self.before_path, width=95).grid(row=1, column=1, sticky="we", padx=6, pady=(6,0))
        tk.Button(top, text="Browse…", command=self.pick_before).grid(row=1, column=2, pady=(6,0))

        tk.Label(top, text="After (binary CSV):").grid(row=2, column=0, sticky="w", pady=(6,0))
        tk.Entry(top, textvariable=self.after_path, width=95).grid(row=2, column=1, sticky="we", padx=6, pady=(6,0))
        tk.Button(top, text="Browse…", command=self.pick_after).grid(row=2, column=2, pady=(6,0))

        # --- options / actions row ---
        opts = tk.Frame(self.root)
        opts.pack(fill="x", padx=10, pady=6)

        tk.Label(opts, text="Channel:").pack(side="left")
        for ch in ("R","G","B"):
            tk.Radiobutton(
                opts, text=ch, variable=self.channel, value=ch,
                command=lambda: self.compute()
            ).pack(side="left", padx=(6,12))

        tk.Label(opts, text="Sort by:").pack(side="left", padx=(12,0))
        tk.OptionMenu(
            opts, self.sort_by,
            "FirstSeen", "Binary", "Before", "AfterSame", "Loss", "DeltaSame", "Mapped"
        ).pack(side="left", padx=(0,12))

        tk.Checkbutton(
            opts,
            text="Only rows with change",
            variable=self.only_changes,
            command=lambda: self.compute(redraw_only=True)
        ).pack(side="left", padx=(0,12))

        tk.Checkbutton(
            opts,
            text="Hide edited = 00000000",
            variable=self.hide_zero_ed,
            command=lambda: self.compute(redraw_only=True)
        ).pack(side="left")

        tk.Button(opts, text="Compute", command=self.compute).pack(side="left", padx=10)
        tk.Button(opts, text="Export CSV…", command=self.export_csv).pack(side="left", padx=(0,10))

        # NEW: Trend buttons
        tk.Button(opts, text="Show Trend (loss vs freq)", command=self.show_trend).pack(side="left", padx=(0,10))
        tk.Button(opts, text="Show Bit Loss", command=self.show_bit_loss).pack(side="left", padx=(0,10))

        tk.Button(opts, text="Quit", command=self.root.destroy).pack(side="right")

        # status + text view
        self.info = tk.StringVar(value="Pick image + CSVs, choose channel, then Compute.")
        tk.Label(self.root, textvariable=self.info, anchor="w").pack(fill="x", padx=10)

        self.text = tk.Text(self.root, height=35, font=("Consolas", 10))
        self.text.pack(fill="both", expand=True, padx=10, pady=(0,10))

        # state
        self._last_raw_table = None
        self._last_totals    = None
        self._img_pixels     = 0
        self._view_rows      = None

        self.root.mainloop()

    # --- pickers ---
    def pick_image(self):
        p = filedialog.askopenfilename(
            title="Choose image",
            filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files","*.*")]
        )
        if p:
            self.image_path.set(p)

    def pick_before(self):
        p = filedialog.askopenfilename(
            title="Choose BEFORE binary CSV",
            filetypes=[("CSV","*.csv"), ("All files","*.*")]
        )
        if p:
            self.before_path.set(p)

    def pick_after(self):
        p = filedialog.askopenfilename(
            title="Choose AFTER binary CSV",
            filetypes=[("CSV","*.csv"), ("All files","*.*")]
        )
        if p:
            self.after_path.set(p)

    # --- compute + render ---
    def compute(self, redraw_only: bool=False):
        try:
            if not redraw_only:
                imgp = self.image_path.get().strip()
                bp   = self.before_path.get().strip()
                ap   = self.after_path.get().strip()
                if not (imgp and bp and ap):
                    messagebox.showwarning("Missing", "Select image, BEFORE CSV, and AFTER CSV.")
                    return

                with Image.open(imgp) as im:
                    w,h = im.size
                self._img_pixels = w*h

                before_rows, skipped_b = _read_binary_csv_8bit(bp)
                after_rows,  skipped_a = _read_binary_csv_8bit(ap)

                raw_table, totals = build_table(before_rows, after_rows, self.channel.get())
                self._last_raw_table = raw_table
                self._last_totals = {
                    **totals,
                    "skipped_before": skipped_b,
                    "skipped_after":  skipped_a,
                }

            if self._last_raw_table is None:
                return

            # filter rows if needed
            table = []
            for b, a, tied, mapped_star, cb, ca_same, delta_same, pct_loss, mapped_total in self._last_raw_table:
                if self.only_changes.get() and b == a:
                    continue
                table.append([b, a, tied, mapped_star, cb, ca_same, delta_same, pct_loss, mapped_total])

            # sort rows
            sorter = self.sort_by.get()
            if sorter == "FirstSeen":
                pass
            elif sorter == "Binary":
                table.sort(key=lambda r: r[0])
            elif sorter == "Before":
                table.sort(key=lambda r: (-r[4], r[0]))
            elif sorter == "AfterSame":
                table.sort(key=lambda r: (-r[5], r[0]))
            elif sorter == "DeltaSame":
                table.sort(key=lambda r: (-r[6], r[0]))
            elif sorter == "Loss":
                table.sort(key=lambda r: (-r[7], r[0]))
            else:  # "Mapped"
                table.sort(key=lambda r: (-r[3], r[0]))

            # render table to text widget
            self.text.delete("1.0","end")

            header = [
                "Binary (8b)",
                "Edited (most common)",
                "#Mapped→edited",
                "Count before",
                "Count after (same)",
                "Δ same",
                "% loss"
            ]
            widths = [12, 24, 16, 14, 18, 10, 8]

            def fmt_row(cols):
                return " ".join(
                    (f"{v:.2f}" if isinstance(v,float) else str(v)).ljust(w)
                    for v,w in zip(cols,widths)
                )

            self.text.insert("end", fmt_row(header) + "\n")
            self.text.insert("end", "-" * (sum(widths) + 3) + "\n")

            for b,a,tied,mapped_star,cb,ca_same,delta_same,pct_loss,_mapped_total in table:
                label = "—" if (not a or (self.hide_zero_ed.get() and a == "00000000")) else a + ("*" if tied else "")
                self.text.insert(
                    "end",
                    fmt_row([b, label, mapped_star, cb, ca_same, delta_same, pct_loss]) + "\n"
                )

            # summary stats
            t = self._last_totals
            self.text.insert("end","\n")
            self.text.insert("end", f"Rows analyzed (window): {t['n_rows']} | Unique BEFORE strings: {t['unique_before']}\n")
            self.text.insert("end", f"No-change (identity) pairs: {t['identity_count']} | Changed rows: {t['total_changed']} "
                                     f"({t['pct_total_changed']:.2f}% of all rows)\n")
            self.text.insert("end", f"Total pixels (image): {self._img_pixels}\n")
            if t['n_rows'] != self._img_pixels:
                self.text.insert("end", "Note: CSV window rows != image pixels (that can be fine if CSVs differ).\n")
            if self._last_totals.get('skipped_before') or self._last_totals.get('skipped_after'):
                self.text.insert("end", f"Skipped invalid rows — before: {self._last_totals['skipped_before']}, "
                                         f"after: {self._last_totals['skipped_after']}\n")

            # top combinations section
            top_pairs = t["top_pairs"][:20]
            if top_pairs:
                self.text.insert("end", "\nTop combinations (excluding identity):\n")
                self.text.insert("end", "b (before)  ->  a (after)    count    % of all pairs   % of BEFORE(b)\n")

                cb_by_b = { row[0]: row[4] for row in self._last_raw_table }
                total_pairs = 3 * t["n_rows"] if t["n_rows"] else 1

                for b,a,c in top_pairs:
                    pct_all = 100.0 * c / total_pairs if total_pairs else 0.0
                    denom = cb_by_b.get(b, 0)
                    pct_from_b = 100.0 * c / denom if denom else 0.0
                    self.text.insert("end",
                        f"{b} -> {a}   {str(c).rjust(8)}   {pct_all:8.2f}%        {pct_from_b:8.2f}%\n"
                    )

            self._view_rows = table
            self.info.set(
                f"Channel {self.channel.get()} | Sorted by {self.sort_by.get()} | "
                f"{len(table)} rows shown."
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # --- plot: scatter loss vs frequency ---
    def show_trend(self):
        if not self._last_raw_table:
            messagebox.showinfo("No data", "Compute first, then Show Trend.")
            return

        freq_loss_pairs = [(row[4], row[7]) for row in self._last_raw_table if row[4] > 0]
        if not freq_loss_pairs:
            messagebox.showinfo("No data", "No valid rows to plot.")
            return

        freq_loss_pairs.sort(key=lambda x: x[0])
        xs = [cb for (cb, _) in freq_loss_pairs]
        ys = [pl for (_, pl) in freq_loss_pairs]

        plt.figure()
        plt.scatter(xs, ys)
        plt.xlabel("Count before (frequency of 8-bit code in BEFORE)")
        plt.ylabel("% loss (did not survive identical)")
        plt.title(f"Loss vs Frequency — Channel {self.channel.get()}")
        plt.grid(True)
        plt.show()

    # --- NEW plot: average %loss per bit position 0..7 ---
    def show_bit_loss(self):
        """
        For each bit position k (0=LSB .. 7=MSB):
          - collect pct_loss for all BEFORE codes where that bit is 1
          - average them
        Plot bar chart: x = bit index, y = average % loss.
        """
        if not self._last_raw_table:
            messagebox.showinfo("No data", "Compute first, then Show Bit Loss.")
            return

        # buckets[bit] = list of pct_loss values
        buckets = {k: [] for k in range(8)}

        # row[0] = binary string like "10101100"
        # row[7] = pct_loss for that code
        for row in self._last_raw_table:
            bcode = row[0]
            pct_loss = row[7]
            if len(bcode) != 8:
                continue
            # bit index 0 = LSB (rightmost char), bit index 7 = MSB (leftmost char)
            # We'll treat bit 0 as bcode[7], bit 7 as bcode[0]
            for bit_index in range(8):
                # map bit_index -> character index in string
                char_index = 7 - bit_index
                if bcode[char_index] == "1":
                    buckets[bit_index].append(pct_loss)

        avg_loss = []
        bits = list(range(8))
        for bit_index in bits:
            vals = buckets[bit_index]
            if vals:
                avg_loss.append(sum(vals)/len(vals))
            else:
                avg_loss.append(0.0)

        plt.figure()
        plt.bar(bits, avg_loss)
        plt.xlabel("Bit position (0 = LSB, 7 = MSB)")
        plt.ylabel("Average % loss")
        plt.title(f"Average loss per bit — Channel {self.channel.get()}")
        plt.grid(True, axis="y")
        plt.xticks(bits)
        plt.show()

    # --- CSV export unchanged from last version ---
    def export_csv(self):
        if not self._view_rows:
            messagebox.showinfo("Nothing to export", "Compute first.")
            return

        p = filedialog.asksaveasfilename(
            title="Save table as CSV",
            defaultextension=".csv",
            filetypes=[("CSV","*.csv")]
        )
        if not p:
            return

        try:
            with open(p, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)

                # 1. Main table
                w.writerow([
                    "Binary (8b)",
                    "Edited (most common)",
                    "Mapped to edited",
                    "Count before",
                    "Count after (same)",
                    "Delta same",
                    "Percent loss"
                ])
                for b,a,tied,mapped_star,cb,ca_same,delta_same,pct_loss,_mapped_total in self._view_rows:
                    label = "" if (not a or (self.hide_zero_ed.get() and a == "00000000")) else a + ("*" if tied else "")
                    w.writerow([
                        b,
                        label,
                        mapped_star,
                        cb,
                        ca_same,
                        delta_same,
                        f"{pct_loss:.2f}"
                    ])

                # 2. Top combinations
                t = self._last_totals
                w.writerow([])
                w.writerow(["Top combinations (excluding identity)"])
                w.writerow(["Before", "After", "Count", "% of all pairs", "% of BEFORE(b)"])

                cb_by_b = { row[0]: row[4] for row in self._last_raw_table }
                total_pairs = 3 * t["n_rows"] if t["n_rows"] else 1

                for b,a,c in t["top_pairs"][:20]:
                    pct_all = 100.0 * c / total_pairs if total_pairs else 0.0
                    denom = cb_by_b.get(b, 0)
                    pct_from_b = 100.0 * c / denom if denom else 0.0
                    w.writerow([
                        b,
                        a,
                        c,
                        f"{pct_all:.2f}%",
                        f"{pct_from_b:.2f}%"
                    ])

                # 3. Count totals by Before (least → greatest)
                before_sorted = sorted(
                    [(row[0], row[4], row[5], row[7]) for row in self._last_raw_table],
                    key=lambda x: x[1]
                )
                w.writerow([])
                w.writerow(["Count totals by Before (least → greatest)"])
                w.writerow(["Binary (8b)", "Count before", "Count after (same)", "% loss"])
                for b_val, cb, ca_same, pct_loss in before_sorted:
                    w.writerow([b_val, cb, ca_same, f"{pct_loss:.2f}%"])

                # 4. Count totals by After (least → greatest)
                after_sorted = sorted(
                    [(row[0], row[4], row[5], row[7]) for row in self._last_raw_table],
                    key=lambda x: x[2]
                )
                w.writerow([])
                w.writerow(["Count totals by After (least → greatest)"])
                w.writerow(["Binary (8b)", "Count after (same)", "Count before", "% loss"])
                for b_val, cb, ca_same, pct_loss in after_sorted:
                    w.writerow([b_val, ca_same, cb, f"{pct_loss:.2f}%"])

                # 5. Loss vs Frequency (sorted by Count before)
                freq_sorted = sorted(
                    [(row[4], row[7]) for row in self._last_raw_table],
                    key=lambda x: x[0]
                )
                w.writerow([])
                w.writerow(["Loss vs Frequency (sorted by Count before)"])
                w.writerow(["Rank", "Count before", "Percent loss"])
                for idx, (cb, pct_loss) in enumerate(freq_sorted, start=1):
                    w.writerow([idx, cb, f"{pct_loss:.2f}%"])

            messagebox.showinfo("Saved", f"Saved to:\n{p}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    App()
