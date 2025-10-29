#!/usr/bin/env python3
"""
color_tool_gui_cvd_v7.py
ColorTruncation CVD simulator/editor (daltonlens-backed)

This version fixes the API call for daltonlens based on your traceback:
we now use simulate.Simulator_Machado2009().simulate_cvd(...) instead of
simulate.simulate_cvd_Machado2009(...).
"""

import os
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

# Try daltonlens
try:
    from daltonlens import simulate
    DALTONLENS_AVAILABLE = True
except Exception:
    DALTONLENS_AVAILABLE = False


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _to_8bit_string(val: int) -> str:
    """0-255 int -> 'xxxxxxxx' 8-bit binary string."""
    if val < 0:
        val = 0
    if val > 255:
        val = 255
    return f"{val:08b}"

def pil_to_numpy_uint8(img: Image.Image) -> np.ndarray:
    """PIL -> np.uint8 array shape (H,W,3) RGB."""
    return np.array(img.convert("RGB"), dtype=np.uint8)

def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """(H,W,3) uint8 -> PIL.Image RGB."""
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")

def get_simulator():
    """
    Build (or cache) a daltonlens simulator instance.

    daltonlens exposes Simulator_Machado2009 which then gives a .simulate_cvd()
    method that expects:
        simulator.simulate_cvd(img_uint8, deficiency, severity)
    """
    # We recreate it each call for safety/simplicity. You could cache globally.
    return simulate.Simulator_Machado2009()

def simulate_cvd_with_daltonlens(pil_img: Image.Image,
                                 deficiency_key: str,
                                 severity_0to1: float) -> Image.Image:
    """
    Run CVD simulation using daltonlens Simulator_Machado2009.

    deficiency_key should be one of:
        "none" / "normal" / "off" -> return original copy
        "protan"
        "deutan"
        "tritan"

    severity_0to1 is clamped into [0.0, 1.0]
    """
    # If daltonlens isn't available, just return original
    if not DALTONLENS_AVAILABLE:
        return pil_img.copy()

    # Normalize type
    key = deficiency_key.lower().strip()
    if key in ("none", "normal", "off"):
        return pil_img.copy()

    if "prot" in key:
        dtype = simulate.Deficiency.PROTAN
    elif "deut" in key:
        dtype = simulate.Deficiency.DEUTAN
    elif "trit" in key:
        dtype = simulate.Deficiency.TRITAN
    else:
        # Unknown -> no change
        return pil_img.copy()

    sev = float(severity_0to1)
    if sev < 0.0:
        sev = 0.0
    if sev > 1.0:
        sev = 1.0

    # Convert PIL -> np.uint8 (H,W,3)
    arr = pil_to_numpy_uint8(pil_img)

    # Build simulator and run
    sim_obj = get_simulator()
    # According to the traceback hint, new API is something like:
    # sim_obj.simulate_cvd(image_uint8, deficiency, severity)
    sim_arr = sim_obj.simulate_cvd(arr, dtype, sev)

    # sim_arr should be uint8 HxWx3
    return numpy_to_pil(sim_arr)

def export_pixels_and_binary_csvs(base_path_no_ext: str,
                                  before_img: Image.Image,
                                  after_img: Image.Image):
    """
    Export:
        *_before_pixels.csv
        *_after_pixels.csv
        *_before_binary.csv
        *_after_binary.csv

    Pixels CSV format:
        header "R,G,B"
        rows of integer 0..255 RGB per pixel

    Binary CSV format:
        header "R,G,B"
        rows of 8-bit binary '01010101' per channel.

    One line per pixel in row-major order.
    """
    if before_img is None or after_img is None:
        raise RuntimeError("Images not ready for export.")

    b_np = pil_to_numpy_uint8(before_img)
    a_np = pil_to_numpy_uint8(after_img)

    # Force same dims for export by cropping to min
    h_b, w_b, _ = b_np.shape
    h_a, w_a, _ = a_np.shape
    h = min(h_b, h_a)
    w = min(w_b, w_a)
    if (h != h_b) or (w != w_b) or (h != h_a) or (w != w_a):
        b_np = b_np[:h, :w, :]
        a_np = a_np[:h, :w, :]

    b_flat = b_np.reshape(-1, 3)
    a_flat = a_np.reshape(-1, 3)

    before_pixels_path = base_path_no_ext + "_before_pixels.csv"
    after_pixels_path  = base_path_no_ext + "_after_pixels.csv"
    before_bin_path    = base_path_no_ext + "_before_binary.csv"
    after_bin_path     = base_path_no_ext + "_after_binary.csv"

    # BEFORE pixels
    with open(before_pixels_path, "w", encoding="utf-8", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["R","G","B"])
        for (r,g,b) in b_flat:
            wri.writerow([int(r), int(g), int(b)])

    # AFTER pixels
    with open(after_pixels_path, "w", encoding="utf-8", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["R","G","B"])
        for (r,g,b) in a_flat:
            wri.writerow([int(r), int(g), int(b)])

    # BEFORE binary
    with open(before_bin_path, "w", encoding="utf-8", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["R","G","B"])
        for (r,g,b) in b_flat:
            wri.writerow([
                _to_8bit_string(int(r)),
                _to_8bit_string(int(g)),
                _to_8bit_string(int(b))
            ])

    # AFTER binary
    with open(after_bin_path, "w", encoding="utf-8", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["R","G","B"])
        for (r,g,b) in a_flat:
            wri.writerow([
                _to_8bit_string(int(r)),
                _to_8bit_string(int(g)),
                _to_8bit_string(int(b))
            ])

    return (
        before_pixels_path,
        after_pixels_path,
        before_bin_path,
        after_bin_path
    )

def pil_resize_keep_aspect(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """
    Downscale PIL image to fit in max_w x max_h, without upscaling.
    """
    w, h = img.size
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    if new_w == w and new_h == h:
        return img
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


# -----------------------------------------------------------------------------
# GUI App
# -----------------------------------------------------------------------------

class ColorToolApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ColorTruncation CVD Simulator (daltonlens)")
        self.root.geometry("1200x700")

        # State
        self.before_img_pil = None   # original
        self.after_img_pil  = None   # simulated
        self.before_tk      = None   # tk preview img
        self.after_tk       = None   # tk preview img
        self.loaded_path    = tk.StringVar(value="")

        # CVD controls
        self.cvd_mode_var = tk.StringVar(value="None")  # "None", "Protan", "Deutan", "Tritan"
        # severity slider 0..100 -> we map /100 to [0..1]
        self.severity_var = tk.IntVar(value=100)

        # ==== Top controls frame ====
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=10)

        # File row
        tk.Label(top, text="Image file:").grid(row=0, column=0, sticky="w")
        tk.Entry(top, textvariable=self.loaded_path, width=70).grid(row=0, column=1, sticky="we", padx=6)
        tk.Button(top, text="Browse…", command=self.pick_image).grid(row=0, column=2, padx=(0,10))
        tk.Button(top, text="Reload", command=self.reload_image).grid(row=0, column=3)

        # CVD row
        tk.Label(top, text="CVD mode:").grid(row=1, column=0, sticky="w", pady=(8,0))

        mode_menu = tk.OptionMenu(top, self.cvd_mode_var,
                                  "None",
                                  "Protan",
                                  "Deutan",
                                  "Tritan")
        mode_menu.grid(row=1, column=1, sticky="w", pady=(8,0))

        tk.Label(top, text="Severity (0-100):").grid(row=1, column=2, sticky="e", pady=(8,0))
        tk.Scale(
            top, from_=0, to=100, orient="horizontal",
            variable=self.severity_var,
            command=lambda _evt=None: self.update_after_image_preview()
        ).grid(row=1, column=3, sticky="we", pady=(8,0))

        # Action row
        tk.Button(
            top,
            text="Apply Simulation / Refresh Preview",
            command=self.update_after_image_preview
        ).grid(row=2, column=1, sticky="w", pady=(10,0))

        tk.Button(
            top,
            text="Export CSVs…",
            command=self.export_csvs
        ).grid(row=2, column=2, sticky="w", pady=(10,0))

        tk.Button(
            top,
            text="Quit",
            command=self.root.destroy
        ).grid(row=2, column=3, sticky="e", pady=(10,0))

        # Status label
        self.status_var = tk.StringVar(value="No image loaded.")
        tk.Label(self.root, textvariable=self.status_var, anchor="w")\
            .pack(fill="x", padx=10, pady=(0,10))

        # Preview frame (left=Before, right=After)
        preview = tk.Frame(self.root)
        preview.pack(fill="both", expand=True, padx=10, pady=10)

        left_frame  = tk.LabelFrame(preview, text="BEFORE (original)")
        right_frame = tk.LabelFrame(preview, text="AFTER (CVD simulated)")
        left_frame.pack(side="left", fill="both", expand=True, padx=(0,5))
        right_frame.pack(side="left", fill="both", expand=True, padx=(5,0))

        self.before_canvas = tk.Label(left_frame, bg="#222222")
        self.after_canvas  = tk.Label(right_frame, bg="#222222")
        self.before_canvas.pack(fill="both", expand=True)
        self.after_canvas.pack(fill="both", expand=True)

        # footer info
        footer = tk.Frame(self.root)
        footer.pack(fill="x", padx=10, pady=(0,10))
        self.dim_var = tk.StringVar(value="")
        tk.Label(footer, textvariable=self.dim_var, anchor="w")\
            .pack(side="left", fill="x", expand=True)

        # daltonlens availability warning
        if not DALTONLENS_AVAILABLE:
            messagebox.showwarning(
                "daltonlens not found",
                "daltonlens is not installed or not importable.\n"
                "Simulation will fall back to 'None'.\n"
                "Install with: pip install daltonlens"
            )

        self.root.mainloop()

    # -------------------------------------------------------------------------
    # Image loading / updating
    # -------------------------------------------------------------------------

    def pick_image(self):
        p = filedialog.askopenfilename(
            title="Choose image",
            filetypes=[
                ("Images","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                ("All files","*.*")
            ]
        )
        if not p:
            return
        self.loaded_path.set(p)
        self.reload_image()

    def reload_image(self):
        path = self.loaded_path.get().strip()
        if not path:
            messagebox.showwarning("No file","Please pick an image.")
            return
        try:
            before = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image:\n{e}")
            return

        self.before_img_pil = before
        self.after_img_pil = self._make_after_from_ui(before)

        self._update_previews()
        self._update_status_counters()

    def _make_after_from_ui(self, src_img: Image.Image) -> Image.Image:
        """Return simulated image using UI mode + severity."""
        mode_ui = self.cvd_mode_var.get().strip().lower()
        sev01 = float(self.severity_var.get()) / 100.0

        # normalize to keys we accept in simulate_cvd_with_daltonlens()
        if "prot" in mode_ui:
            key = "protan"
        elif "deut" in mode_ui:
            key = "deutan"
        elif "trit" in mode_ui:
            key = "tritan"
        elif "none" in mode_ui or "normal" in mode_ui or "off" in mode_ui:
            key = "none"
        else:
            key = "none"

        return simulate_cvd_with_daltonlens(src_img, key, sev01)

    def update_after_image_preview(self):
        """Recompute 'after' using current slider + mode, then refresh canvases."""
        if self.before_img_pil is None:
            return
        self.after_img_pil = self._make_after_from_ui(self.before_img_pil)
        self._update_previews()
        self._update_status_counters()

    def _update_previews(self):
        """Update the tk.PhotoImage previews for before/after."""
        if self.before_img_pil is None:
            self.before_canvas.config(image="", text="(no image)")
            self.after_canvas.config(image="", text="(no image)")
            return

        b_disp = pil_resize_keep_aspect(self.before_img_pil, 500, 500)
        a_disp = pil_resize_keep_aspect(
            self.after_img_pil if self.after_img_pil else self.before_img_pil,
            500, 500
        )

        self.before_tk = ImageTk.PhotoImage(b_disp)
        self.after_tk  = ImageTk.PhotoImage(a_disp)

        self.before_canvas.configure(image=self.before_tk)
        self.after_canvas.configure(image=self.after_tk)

    def _update_status_counters(self):
        """Update status line + footer info."""
        if self.before_img_pil is None:
            self.status_var.set("No image loaded.")
            self.dim_var.set("")
            return

        w, h = self.before_img_pil.size
        self.dim_var.set(f"Image size: {w}x{h} | Total pixels: {w*h}")

        mode_ui = self.cvd_mode_var.get().strip()
        sev_ui  = self.severity_var.get()
        self.status_var.set(
            f"Mode={mode_ui}  Severity={sev_ui}/100  |  "
            f"daltonlens={'OK' if DALTONLENS_AVAILABLE else 'NOT AVAILABLE'}"
        )

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    def export_csvs(self):
        """
        Choose a base name, then dump:
            *_before_pixels.csv
            *_after_pixels.csv
            *_before_binary.csv
            *_after_binary.csv
        """
        if self.before_img_pil is None or self.after_img_pil is None:
            messagebox.showwarning("No image", "Load and simulate first.")
            return

        initial_name = os.path.splitext(
            os.path.basename(self.loaded_path.get().strip() or "output")
        )[0]

        save_path = filedialog.asksaveasfilename(
            title="Choose base name for CSV export",
            initialfile=initial_name,
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"), ("All files","*.*")]
        )
        if not save_path:
            return

        base_no_ext = save_path
        if base_no_ext.lower().endswith(".csv"):
            base_no_ext = base_no_ext[:-4]

        try:
            paths = export_pixels_and_binary_csvs(
                base_no_ext,
                self.before_img_pil,
                self.after_img_pil
            )
            msg = "Exported:\n" + "\n".join(paths)
            messagebox.showinfo("Export complete", msg)
        except Exception as e:
            messagebox.showerror("Export error", str(e))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    ColorToolApp()
