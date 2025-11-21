#!/usr/bin/env python3
# Unique 8-bit counter + "what it became" (R/G/B)
# Rows come ONLY from unique strings that appear in BEFORE.
# Default ordering = First occurrence in BEFORE (no lexicographic walk).
# Added:
#  - Percent loss vs SAME binary in AFTER (per row)
#  - "Top combinations" section for highest before->after pairs (excluding identity)
#  - Percent total change over all rows
#  - NEW: For every BEFORE value, show TOP 20 AFTER targets with % of that BEFORE's own count
#         (included in GUI text and CSV export). Identity included.

import csv
import tkinter as tk
from tkinter import filedialog, messagebox
from collections import Counter, OrderedDict, defaultdict
from typing import List, Tuple, Dict
from PIL import Image

# -------- CSV helpers --------

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
                skipped += 1; continue
            if first:
                head = ",".join(_clean_cell(x).lower() for x in row[:3])
                if head in ("r,g,b","r, g, b"):
                    first = False; continue
                first = False
            r8 = _to_8bit(row[0]); g8 = _to_8bit(row[1]); b8 = _to_8bit(row[2])
            if r8 and g8 and b8:
                rows.append((r8,g8,b8))
            else:
                skipped += 1
    return rows, skipped

# -------- Core --------

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

def most_common_mapping(pair_counts: Counter, b: str) -> Tuple[str,int,bool]:
    opts = [(a,c) for (bb,a),c in pair_counts.items() if bb == b]
    if not opts:
        return "",0,False
    maxc = max(c for _,c in opts)
    winners = sorted([a for a,c in opts if c == maxc])
    return winners[0], maxc, (len(winners) > 1)

def per_before_topk(pair_counts: Counter, c_before: Counter, k: int=20
                   ) -> Dict[str, List[Tuple[str,int,float]]]:
    """
    For each BEFORE binary b:
      - gather all (a, count) from pair_counts where key=(b,a)
      - sort by count desc, then a
      - return top k along with percentage of BEFORE(b): 100 * count / cb
      - includes identity (b->b)
    """
    per_b: Dict[str, List[Tuple[str,int,float]]] = {}
    # Build mapping lists per b without scanning all pairs repeatedly
    temp = defaultdict(list)
    for (b,a), c in pair_counts.items():
        temp[b].append((a,c))
    for b, lst in temp.items():
        cb = max(1, c_before.get(b, 0))  # avoid div by zero
        lst.sort(key=lambda t: (-t[1], t[0]))
        top = []
        for a, c in lst[:k]:
            pct = (100.0 * c / cb) if cb else 0.0
            top.append((a, c, pct))
        per_b[b] = top
    return per_b

def build_table(before_rows, after_rows, ch: str):
    """
    Counts are limited to the same first-n rows used for mapping
    (n = min(len(before), len(after))).
    We compute:
      - cb: BEFORE count (per binary) in window
      - ca_same: AFTER count of the *same* binary
      - pct_loss: max((cb - ca_same) / cb, 0) * 100
      - a_star / mapped_star: most common edited target and its pair count
      - mapped_total: total mapped out of this binary across ANY after (== cb)
      Also produce:
      - identity_count, total_changed, pct_total_changed
      - top_pairs: list of most frequent before->after pairs (excluding identity)
      - per_b_top: dict b -> list of (a,count,% of BEFORE(b)) for top 20
    """
    b_list, a_list, first_seen_order = channel_lists_and_first_seen(before_rows, after_rows, ch)

    n = min(len(b_list), len(a_list))
    b_used = b_list[:n]
    a_used = a_list[:n]

    # counts within same window
    c_before = Counter(b_used)
    c_after  = Counter(a_used)
    pair_counts = Counter(zip(b_used, a_used))

    # identity / total change
    identity_count = sum(pair_counts.get((x, x), 0) for x in c_before.keys())
    total_changed = n - identity_count
    pct_total_changed = (100.0 * total_changed / n) if n else 0.0

    # top combinations (exclude identity)
    top_pairs_all = [ (b,a,c) for (b,a),c in pair_counts.items() if b != a ]
    top_pairs_all.sort(key=lambda t: (-t[2], t[0], t[1]))

    # NEW: per-before top 20 breakdown (includes identity)
    per_b_top = per_before_topk(pair_counts, c_before, k=20)

    # build rows
    table = []
    for b in first_seen_order:
        cb = c_before.get(b, 0)
        a_star, mapped_star, tied = most_common_mapping(pair_counts, b)
        ca_same = c_after.get(b, 0)                      # AFTER count of the *same* binary
        delta_same = ca_same - cb
        pct_loss = (max(cb - ca_same, 0) / cb * 100.0) if cb else 0.0
        mapped_total = cb                                 # total mapped from b across ANY after

        # Row fields:
        # [b, a_star, tied, mapped_star, cb, ca_same, delta_same, pct_loss, mapped_total]
        table.append([b, a_star, tied, mapped_star, cb, ca_same, delta_same, pct_loss, mapped_total])

    totals = {
        "n_rows":             n,
        "identity_count":     identity_count,
        "total_changed":      total_changed,
        "pct_total_changed":  pct_total_changed,
        "unique_before":      len(first_seen_order),
        "sum_before_counts":  n,
        "sum_after_counts":   n,
        "pairs_used":         n,
        "top_pairs":          top_pairs_all,   # list of (b,a,count)
        "per_b_top":          per_b_top,       # dict b -> list of (a,count,pct_of_b)
    }
    return table, totals

# -------- GUI --------

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Unique 8-bit Counter + Edited Mapping (First-seen from BEFORE)")
        self.root.geometry("1200x780")

        self.image_path  = tk.StringVar()
        self.before_path = tk.StringVar()
        self.after_path  = tk.StringVar()
        self.channel     = tk.StringVar(value="R")  # R/G/B

        # Sorting: add "First seen (before)" default
        self.sort_by     = tk.StringVar(value="FirstSeen")  # FirstSeen | Binary | Before | AfterSame | Loss | Mapped | DeltaSame
        self.only_changes = tk.BooleanVar(value=False)
        self.hide_zero_ed = tk.BooleanVar(value=True)

        # pickers
        top = tk.Frame(self.root); top.pack(fill="x", padx=10, pady=(10,6))
        tk.Label(top, text="Image:").grid(row=0, column=0, sticky="w")
        tk.Entry(top, textvariable=self.image_path, width=95).grid(row=0, column=1, sticky="we", padx=6)
        tk.Button(top, text="Browse…", command=self.pick_image).grid(row=0, column=2)

        tk.Label(top, text="Before (binary CSV):").grid(row=1, column=0, sticky="w", pady=(6,0))
        tk.Entry(top, textvariable=self.before_path, width=95).grid(row=1, column=1, sticky="we", padx=6, pady=(6,0))
        tk.Button(top, text="Browse…", command=self.pick_before).grid(row=1, column=2, pady=(6,0))

        tk.Label(top, text="After (binary CSV):").grid(row=2, column=0, sticky="w", pady=(6,0))
        tk.Entry(top, textvariable=self.after_path, width=95).grid(row=2, column=1, sticky="we", padx=6, pady=(6,0))
        tk.Button(top, text="Browse…", command=self.pick_after).grid(row=2, column=2, pady=(6,0))

        # options
        opts = tk.Frame(self.root); opts.pack(fill="x", padx=10, pady=6)
        tk.Label(opts, text="Channel:").pack(side="left")
        for ch in ("R","G","B"):
            tk.Radiobutton(opts, text=ch, variable=self.channel, value=ch,
                           command=lambda: self.compute()).pack(side="left", padx=(6,12))

        tk.Label(opts, text="Sort by:").pack(side="left", padx=(12,0))
        tk.OptionMenu(opts, self.sort_by,
                      "FirstSeen", "Binary", "Before", "AfterSame", "Loss", "DeltaSame", "Mapped"
                      ).pack(side="left", padx=(0,12))

        tk.Checkbutton(opts, text="Only rows with change", variable=self.only_changes,
                       command=lambda: self.compute(redraw_only=True)).pack(side="left", padx=(0,12))
        tk.Checkbutton(opts, text="Hide edited = 00000000", variable=self.hide_zero_ed,
                       command=lambda: self.compute(redraw_only=True)).pack(side="left")

        tk.Button(opts, text="Compute", command=self.compute).pack(side="left", padx=10)
        tk.Button(opts, text="Export CSV…", command=self.export_csv).pack(side="left")
        tk.Button(opts, text="Quit", command=self.root.destroy).pack(side="right")

        self.info = tk.StringVar(value="Pick image + CSVs, choose channel, then Compute.")
        tk.Label(self.root, textvariable=self.info, anchor="w").pack(fill="x", padx=10)

        self.text = tk.Text(self.root, height=30, font=("Consolas", 10))
        self.text.pack(fill="both", expand=True, padx=10, pady=(0,10))

        self._last_raw_table = None
        self._last_totals = None
        self._img_pixels = 0
        self._view_rows = None

        self.root.mainloop()

    # pickers
    def pick_image(self):
        p = filedialog.askopenfilename(title="Choose image",
            filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files","*.*")])
        if p: self.image_path.set(p)
    def pick_before(self):
        p = filedialog.askopenfilename(title="Choose BEFORE binary CSV",
            filetypes=[("CSV","*.csv"), ("All files","*.*")])
        if p: self.before_path.set(p)
    def pick_after(self):
        p = filedialog.askopenfilename(title="Choose AFTER binary CSV",
            filetypes=[("CSV","*.csv"), ("All files","*.*")])
        if p: self.after_path.set(p)

    # compute
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
                self._last_totals = {**totals, "skipped_before": skipped_b, "skipped_after": skipped_a}

            if self._last_raw_table is None:
                return

            # view filters
            table = []
            for b, a, tied, mapped_star, cb, ca_same, delta_same, pct_loss, mapped_total in self._last_raw_table:
                if self.only_changes.get() and b == a:
                    continue
                table.append([b, a, tied, mapped_star, cb, ca_same, delta_same, pct_loss, mapped_total])

            # sorting
            sorter = self.sort_by.get()
            if sorter == "FirstSeen":
                pass
            elif sorter == "Binary":
                table.sort(key=lambda r: r[0])
            elif sorter == "Before":
                table.sort(key=lambda r: (-r[4], r[0]))        # cb
            elif sorter == "AfterSame":
                table.sort(key=lambda r: (-r[5], r[0]))        # ca_same
            elif sorter == "DeltaSame":
                table.sort(key=lambda r: (-r[6], r[0]))        # delta_same
            elif sorter == "Loss":
                table.sort(key=lambda r: (-r[7], r[0]))        # pct_loss
            else:  # Mapped (to edited)
                table.sort(key=lambda r: (-r[3], r[0]))        # mapped_star

            # render main table
            self.text.delete("1.0","end")
            header = ["Binary (8b)", "Edited (most common)", "#Mapped→edited",
                      "Count before", "Count after (same)", "Δ same", "% loss", "Mapped total (any)"]
            widths = [12, 24, 16, 14, 18, 10, 8, 18]
            def fmt_row(cols): return " ".join((f"{v:.2f}" if isinstance(v,float) else str(v)).ljust(w) for v,w in zip(cols,widths))

            self.text.insert("end", fmt_row(header) + "\n")
            self.text.insert("end", "-" * (sum(widths) + 3) + "\n")
            for b,a,tied,mapped_star,cb,ca_same,delta_same,pct_loss,mapped_total in table:
                label = "—" if (not a or (self.hide_zero_ed.get() and a == "00000000")) else a + ("*" if tied else "")
                self.text.insert("end", fmt_row([b, label, mapped_star, cb, ca_same, delta_same, pct_loss, mapped_total]) + "\n")

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

            # Top combinations (excluding identity)
            top_pairs = t["top_pairs"][:20]
            if top_pairs:
                self.text.insert("end", "\nTop combinations (excluding identity):\n")
                self.text.insert("end", "b (before)  ->  a (after)    count    % of all rows   % of BEFORE(b)\n")
                # denominators per b from raw table
                cb_by_b = { row[0]: row[4] for row in self._last_raw_table }  # b -> cb
                for b,a,c in top_pairs:
                    pct_all = 100.0 * c / t["n_rows"] if t["n_rows"] else 0.0
                    denom = cb_by_b.get(b, 0)
                    pct_from_b = 100.0 * c / denom if denom else 0.0
                    self.text.insert("end", f"{b} -> {a}   {str(c).rjust(8)}   {pct_all:8.2f}%        {pct_from_b:8.2f}%\n")

            # NEW: Per-before mapping breakdown (Top 20 targets each; identity INCLUDED)
            self.text.insert("end", "\nPer-before mappings — Top 20 AFTER targets (identity included):\n")
            self.text.insert("end", "BEFORE    |  AFTER  |  count  |  % of BEFORE(b)\n")
            self.text.insert("end", "----------+---------+---------+----------------\n")
            per_b = t["per_b_top"]
            # Keep first-seen order of BEFORE values
            for b, a_star, tied, mapped_star, cb, *_ in self._last_raw_table:
                sub = per_b.get(b, [])
                for a, c, pct in sub:
                    self.text.insert("end", f"{b}  ->  {a}   {str(c).rjust(7)}    {pct:8.2f}%\n")

            self._view_rows = table

            self.info.set(
                f"Channel {self.channel.get()} | Sorted by {self.sort_by.get()} | "
                f"{len(table)} rows shown."
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # export
    def export_csv(self):
        if not self._view_rows:
            messagebox.showinfo("Nothing to export", "Compute first."); return
        p = filedialog.asksaveasfilename(title="Save table as CSV",
                                         defaultextension=".csv",
                                         filetypes=[("CSV","*.csv")])
        if not p: return
        try:
            with open(p, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Binary (8b)", "Edited (most common)", "Mapped to edited",
                            "Count before", "Count after (same)", "Delta same", "Percent loss",
                            "Mapped total (any)"])
                for b,a,tied,mapped_star,cb,ca_same,delta_same,pct_loss,mapped_total in self._view_rows:
                    label = "" if (not a or (self.hide_zero_ed.get() and a == "00000000")) else a + ("*" if tied else "")
                    w.writerow([b, label, mapped_star, cb, ca_same, delta_same, f"{pct_loss:.2f}", mapped_total])

                # footer + top pairs
                t = self._last_totals
                w.writerow([])
                w.writerow(["Channel", self.channel.get()])
                w.writerow(["Sorted by", self.sort_by.get()])
                w.writerow(["Only rows with change", self.only_changes.get()])
                w.writerow(["Hide edited = 00000000", self.hide_zero_ed.get()])
                w.writerow(["Rows analyzed (window)", t["n_rows"]])
                w.writerow(["Unique BEFORE strings", t["unique_before"]])
                w.writerow(["Identity pairs", t["identity_count"]])
                w.writerow(["Changed rows", t["total_changed"]])
                w.writerow(["Percent total changed", f"{t['pct_total_changed']:.2f}%"])
                w.writerow(["Total pixels (image)", self._img_pixels])
                w.writerow(["Skipped invalid rows (before)", self._last_totals['skipped_before']])
                w.writerow(["Skipped invalid rows (after)",  self._last_totals['skipped_after']])

                # Top combinations
                w.writerow([])
                w.writerow(["Top combinations (excluding identity)"])
                w.writerow(["Before", "After", "Count", "% of all rows", "% of BEFORE(b)"])
                cb_by_b = { row[0]: row[4] for row in self._last_raw_table }  # b -> cb
                for b,a,c in t["top_pairs"][:20]:
                    pct_all = 100.0 * c / t["n_rows"] if t["n_rows"] else 0.0
                    denom = cb_by_b.get(b, 0)
                    pct_from_b = 100.0 * c / denom if denom else 0.0
                    w.writerow([b, a, c, f"{pct_all:.2f}%", f"{pct_from_b:.2f}%"])

                # NEW: Per-before mapping breakdown (TOP 20, identity INCLUDED)
                w.writerow([])
                w.writerow(["Per-before mappings — Top 20 AFTER targets (identity included)"])
                w.writerow(["BEFORE", "AFTER", "Count", "% of BEFORE(b)"])
                per_b = t["per_b_top"]
                # Emit in first-seen order
                for row in self._last_raw_table:
                    b = row[0]
                    cb = row[4]
                    sub = per_b.get(b, [])
                    for a, c, pct in sub:
                        w.writerow([b, a, c, f"{pct:.2f}%"])

            messagebox.showinfo("Saved", f"Saved to:\n{p}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    App()
