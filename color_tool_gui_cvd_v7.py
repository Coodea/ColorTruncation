# color_tool_gui_cvd_v7.py
# Adjustable GUI (Color Vision & Highlight) + automatic data export.
# AUTOMATIC DATA EXPORT:
#   • On image open  -> saves BEFORE data beside the source image:
#       <srcdir>/<name>_before_bits.txt          (rows of 0/1 for BGR bytes)
#       <srcdir>/<name>_before_pixels.csv        (B,G,R per row)
#   • On Save (S)    -> saves AFTER data beside the source image:
#       <srcdir>/<name>_<mode>_after#<n>_bits.txt
#       <srcdir>/<name>_<mode>_after#<n>_pixels.csv
#     (# increments each time you save during the session)
#
# NOTE: Quit never saves anything new. Only opening an image (BEFORE) and pressing S (AFTER).

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from typing import Tuple

VERSION_TAG = "CVD GUI v7 (auto BEFORE+AFTER data)"

# ---------- Core helpers: image<->pixels<->binary ----------

def image_to_pixels(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("Input must be a color image (H x W x 3).")
    return img_bgr.astype(np.uint8, copy=False)

def pixels_to_binary(pixels: np.ndarray) -> bytes:
    return pixels.tobytes(order="C")

def binary_to_pixels(buf: bytes, shape: Tuple[int,int,int], dtype=np.uint8) -> np.ndarray:
    h, w, c = shape
    arr = np.frombuffer(buf, dtype=dtype)
    expect = h*w*c
    if arr.size != expect:
        raise ValueError(f"Binary buffer has {arr.size} elements; expected {expect}.")
    return arr.reshape((h, w, c))

# ---------- Data export (bits + pixels) ----------

def write_bits_per_row_txt(u8_bgr: np.ndarray, path_txt: str) -> None:
    """
    Writes 0/1 text of the image bytes, one image row per line.
    Each line length = width * 3 channels * 8 bits.
    """
    h, w, c = u8_bgr.shape
    assert c == 3
    with open(path_txt, "w", encoding="utf-8") as f:
        row_bytes = w * 3
        flat = u8_bgr.reshape(-1).view(np.uint8)
        for y in range(h):
            start = y * row_bytes
            end   = start + row_bytes
            row = flat[start:end]
            bits = np.unpackbits(row)  # big-endian bit order
            # join is much faster than str(list(...))
            f.write("".join('1' if b else '0' for b in bits.tolist()) + "\n")

def write_pixels_csv(u8_bgr: np.ndarray, path_csv: str) -> None:
    """
    Writes one pixel per row as "B,G,R" integers.
    """
    h, w, c = u8_bgr.shape
    flat = u8_bgr.reshape(-1, c)
    np.savetxt(path_csv, flat, fmt="%d", delimiter=",")

# ---------- Highlight (HSV) ----------

def make_mask_hsv(hsv: np.ndarray, hc:int, hr:int, smin:int, smax:int, vmin:int, vmax:int,
                  morph:int=0, soft:int=2, invert:int=0) -> np.ndarray:
    H,S,V = cv2.split(hsv)
    low = (hc - hr) % 180
    high= (hc + hr) % 180
    if low <= high:
        hmask = cv2.inRange(H, low, high)
    else:
        hmask = cv2.inRange(H, low, 179) | cv2.inRange(H, 0, high)
    smask = cv2.inRange(S, smin, smax)
    vmask = cv2.inRange(V, vmin, vmax)
    mask = hmask & smask & vmask
    if invert == 1:
        mask = 255 - mask
    if morph > 0:
        k = 2*morph + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if soft > 0:
        k = 2*soft + 1
        mask = cv2.GaussianBlur(mask, (k,k), 0)
    return (mask.astype(np.float32)/255.0)

def overlay_style(bgr: np.ndarray, mask01: np.ndarray, style:int, bg_dim:float,
                  alpha:float, hue_for_overlay:int, show_mask:bool=False) -> np.ndarray:
    # style: 0=Keep Only, 1=Dim BG, 2=Overlay (complementary hue)
    if show_mask:
        m8 = np.clip(mask01*255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(m8, cv2.COLOR_GRAY2BGR)
    if style == 0:
        out = (bgr.astype(np.float32) * mask01[...,None])
        return np.clip(out, 0, 255).astype(np.uint8)
    if style == 1:
        out = bgr.astype(np.float32) * (mask01[...,None] + (1.0 - mask01[...,None])*bg_dim)
        return np.clip(out, 0, 255).astype(np.uint8)
    # style 2: overlay complementary color
    comp_h = (hue_for_overlay + 90) % 180
    overlay_col = cv2.cvtColor(np.uint8([[[comp_h,255,255]]]), cv2.COLOR_HSV2BGR)[0,0,:].astype(np.float32)
    base = bgr.astype(np.float32)
    out = base*(1.0 - alpha*mask01[...,None]) + overlay_col[None,None,:]*(alpha*mask01[...,None])
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------- Color-Vision (LMS-based) ----------

RGB_to_LMS = np.array([[0.31399022, 0.63951294, 0.04649755],
                       [0.15537241, 0.75789446, 0.08670142],
                       [0.01775239, 0.10944209, 0.87256922]], dtype=np.float32)
LMS_to_RGB = np.linalg.inv(RGB_to_LMS).astype(np.float32)

SIM_PROTAN = np.array([[0.0, 1.05118294, -0.05116099],
                       [0.0, 1.0,         0.0       ],
                       [0.0, 0.0,         1.0       ]], dtype=np.float32)
SIM_DEUTAN = np.array([[1.0, 0.0,         0.0      ],
                       [0.9513092, 0.0,   0.04866992],
                       [0.0, 0.0,         1.0      ]], dtype=np.float32)
SIM_TRITAN = np.array([[1.0, 0.0,         0.0      ],
                       [0.0, 1.0,         0.0      ],
                       [-0.86744736, 1.86727089, 0.0]], dtype=np.float32)

def bgr_to_rgb01(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

def rgb01_to_bgr(rgb01: np.ndarray) -> np.ndarray:
    rgb8 = np.clip(np.round(rgb01 * 255.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)

def simulate_cvd_rgb(rgb01: np.ndarray, kind: str) -> np.ndarray:
    h, w, _ = rgb01.shape
    flat = rgb01.reshape(-1, 3).astype(np.float32)
    lms = flat @ RGB_to_LMS.T
    if kind == "protan":
        lms2 = lms @ SIM_PROTAN.T
    elif kind == "deutan":
        lms2 = lms @ SIM_DEUTAN.T
    elif kind == "tritan":
        lms2 = lms @ SIM_TRITAN.T
    else:
        raise ValueError("Unknown CVD kind")
    rgb2 = lms2 @ LMS_to_RGB.T
    return np.clip(rgb2.reshape(h, w, 3), 0.0, 1.0)

def simulate_achromatopsia(rgb01: np.ndarray) -> np.ndarray:
    Y = (0.2126*rgb01[...,0] + 0.7152*rgb01[...,1] + 0.0722*rgb01[...,2])[..., None]
    return np.repeat(Y, 3, axis=-1)

def daltonize_rgb(rgb01: np.ndarray, kind: str, strength: float=0.8) -> np.ndarray:
    if kind == "achroma":
        sim = simulate_achromatopsia(rgb01)
        err = rgb01 - sim
        return np.clip(sim + strength * err, 0.0, 1.0)
    sim = simulate_cvd_rgb(rgb01, kind)
    err = rgb01 - sim
    if kind == "protan":
        corr = np.stack([0.0*err[...,0], 0.7*err[...,0], 0.7*err[...,0]], axis=-1)
    elif kind == "deutan":
        corr = np.stack([0.7*err[...,1], 0.0*err[...,1], 0.7*err[...,1]], axis=-1)
    else:  # tritan
        corr = np.stack([0.7*err[...,2], 0.7*err[...,2], 0.0*err[...,2]], axis=-1)
    return np.clip(rgb01 + strength * corr, 0.0, 1.0)

def apply_cvd_pipeline(bgr: np.ndarray, cvd_type_ui: str, view_mode: str,
                       severity01: float, assist_strength01: float,
                       rg_subtype: str = "deutan", tri_variant: str = "deutan") -> np.ndarray:
    rgb = bgr_to_rgb01(bgr)
    label = (cvd_type_ui or "").lower().replace("–", "-")

    if "deuteranopia" in label or "deuteranomaly" in label:
        base = "deutan"
    elif "protanopia" in label or "protanomaly" in label:
        base = "protan"
    elif "tritanopia" in label or "tritanomaly" in label or "blue-yellow" in label:
        base = "tritan"
    elif "achromatopsia" in label or "complete colorblindness" in label:
        base = "achroma"
    elif "red-green" in label:
        base = "deutan" if rg_subtype == "deutan" else "protan"
    elif "anomalous trichromacy" in label:
        base = tri_variant
    else:
        base = "deutan"

    sim = simulate_achromatopsia(rgb) if base == "achroma" else simulate_cvd_rgb(rgb, base)
    rgb_sim = (1.0 - severity01) * rgb + severity01 * sim
    if view_mode == "simulate":
        return rgb01_to_bgr(rgb_sim)
    out = daltonize_rgb(rgb, base if base != "achroma" else "achroma", assist_strength01)
    out = (1.0 - severity01) * rgb + severity01 * out
    return rgb01_to_bgr(out)

# ---------- App ----------

CVD_TYPES = [
    "Deuteranopia", "Protanopia", "Tritanopia", "Blue-Yellow colorblindness",
    "Complete colorblindness", "Deuteranomaly", "Protanomaly",
    "Red–green color blindness", "Achromatopsia", "Tritanomaly", "Anomalous Trichromacy",
]

class ColorToolApp:
    def __init__(self):
        print(VERSION_TAG)
        self.root = tk.Tk()
        self.root.title(f"Color Tool — Launcher [{VERSION_TAG}]")
        self.root.geometry("560x260")
        self.root.resizable(False, False)

        # PREDEFINE Color-Vision vars (prevents early-callback issues)
        self.var_cvd_type    = tk.StringVar(value=CVD_TYPES[0])
        self.var_view        = tk.StringVar(value="simulate")  # "simulate" or "assist"
        self.var_sev         = tk.IntVar(value=100)
        self.var_strength    = tk.IntVar(value=80)
        self.var_rg_sub      = tk.StringVar(value="deutan")
        self.var_tri_variant = tk.StringVar(value="deutan")

        self.mode = tk.StringVar(value="cvd")  # "cvd" or "highlight"
        self.image_path = tk.StringVar(value="")

        frm = tk.Frame(self.root)
        frm.pack(fill="both", expand=True, padx=14, pady=14)

        tk.Label(frm, text="Mode", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        tk.Radiobutton(frm, text="Color Vision (choose type; simulate or assist)", variable=self.mode, value="cvd").grid(row=1, column=0, sticky="w")
        tk.Radiobutton(frm, text="Highlight (HSV isolate / overlay)",               variable=self.mode, value="highlight").grid(row=2, column=0, sticky="w")

        tk.Label(frm, text="Image", font=("Segoe UI", 11, "bold")).grid(row=3, column=0, sticky="w", pady=10)
        tk.Entry(frm, textvariable=self.image_path, width=54).grid(row=4, column=0, sticky="we")
        tk.Button(frm, text="Browse…", command=self.browse).grid(row=4, column=1, padx=8)

        tk.Button(frm, text="Open", width=12, command=self.open_session).grid(row=5, column=0, pady=16, sticky="w")
        tk.Button(frm, text="Quit", width=12, command=self.root.destroy).grid(row=5, column=1, pady=16, sticky="e")

        self.root.mainloop()

    def browse(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files","*.*")]
        )
        if path:
            self.image_path.set(path)

    # ---------- Session ----------
    def open_session(self):
        path = self.image_path.get()
        if not path:
            messagebox.showwarning("No image", "Please select an image.")
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", f"Could not open: {path}")
            return

        self.img_full = img
        self.h_full, self.w_full = img.shape[:2]
        self.hsv_full = cv2.cvtColor(self.img_full, cv2.COLOR_BGR2HSV)

        # Where to auto-save data (same folder as source image)
        self.src_dir  = os.path.dirname(path)
        self.src_name = os.path.splitext(os.path.basename(path))[0]
        self.save_count = 0  # increments for each AFTER save

        # --- AUTO-SAVE BEFORE data immediately ---
        try:
            before_bits   = os.path.join(self.src_dir, f"{self.src_name}_before_bits.txt")
            before_pixels = os.path.join(self.src_dir, f"{self.src_name}_before_pixels.csv")
            write_bits_per_row_txt(self.img_full, before_bits)
            write_pixels_csv(self.img_full, before_pixels)
        except Exception as e:
            messagebox.showwarning("Before data not saved", f"Could not write BEFORE data:\n{e}")

        self.show_original = False
        self.show_mask = False
        self.last_pick_hsv = None

        # Image viewer (Tk Canvas — resizable, fit-to-window)
        self.viewer = tk.Toplevel(self.root)
        self.viewer.title(f"Image — Fit to Window [{VERSION_TAG}]  (S=Save  O=Original  Space=Mask  Q/Esc=Quit)")
        self.viewer.geometry("1000x640")
        self.viewer.minsize(480, 360)

        self.canvas = tk.Canvas(self.viewer, background="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.viewer.bind("<Key-s>", self.on_save)   # also triggers AFTER data export
        self.viewer.bind("<Key-S>", self.on_save)
        self.viewer.bind("<Key-o>", self.on_toggle_orig)
        self.viewer.bind("<Key-O>", self.on_toggle_orig)
        self.viewer.bind("<space>", self.on_toggle_mask)
        self.viewer.bind("<Key-q>", self.on_quit)
        self.viewer.bind("<Key-Escape>", self.on_quit)

        if self.mode.get() == "cvd":
            self.build_cvd_controls()
        else:
            self.build_highlight_controls()

        # First draw after windows exist
        self.viewer.after(50, self.update_preview)

    # ---------- Color Vision controls ----------
    def build_cvd_controls(self):
        self.ctrl = tk.Toplevel(self.root)
        self.ctrl.title("Color Vision — Type & Options")
        self.ctrl.resizable(False, False)

        tk.Label(self.ctrl, text="Color-blindness Type:", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10, pady=8)
        self.type_combo = ttk.Combobox(self.ctrl, textvariable=self.var_cvd_type, values=CVD_TYPES, state="readonly", width=34)
        self.type_combo.pack(anchor="w", padx=10)
        self.type_combo.bind("<<ComboboxSelected>>", lambda e: (self._update_subrows(), self.update_preview()))

        # Subtype rows holder
        self.sub_frame = tk.Frame(self.ctrl)
        self.sub_frame.pack(fill="x", padx=10, pady=6)

        # Red–Green subtype row
        self.rg_row = tk.Frame(self.sub_frame)
        tk.Label(self.rg_row, text="Red–Green subtype:").pack(side="left")
        ttk.Radiobutton(self.rg_row, text="Deutan-like", variable=self.var_rg_sub, value="deutan",
                        command=self.update_preview).pack(side="left", padx=6)
        ttk.Radiobutton(self.rg_row, text="Protan-like", variable=self.var_rg_sub, value="protan",
                        command=self.update_preview).pack(side="left", padx=6)

        # Anomalous variant row
        self.tri_row = tk.Frame(self.sub_frame)
        tk.Label(self.tri_row, text="Anomalous variant:").pack(side="left")
        ttk.Radiobutton(self.tri_row, text="Deuteranomaly", variable=self.var_tri_variant, value="deutan",
                        command=self.update_preview).pack(side="left", padx=6)
        ttk.Radiobutton(self.tri_row, text="Protanomaly",   variable=self.var_tri_variant, value="protan",
                        command=self.update_preview).pack(side="left", padx=6)
        ttk.Radiobutton(self.tri_row, text="Tritanomaly",   variable=self.var_tri_variant, value="tritan",
                        command=self.update_preview).pack(side="left", padx=6)

        # View mode row
        view_row = tk.Frame(self.ctrl)
        view_row.pack(fill="x", padx=10, pady=6)
        tk.Label(view_row, text="View Mode:", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Radiobutton(view_row, text="Simulate (how it looks)", variable=self.var_view, value="simulate",
                        command=self.update_preview).pack(side="left", padx=6)
        ttk.Radiobutton(view_row, text="Assist (daltonize)",       variable=self.var_view, value="assist",
                        command=self.update_preview).pack(side="left", padx=6)

        # Severity / Strength
        sliders = tk.Frame(self.ctrl)
        sliders.pack(fill="x", padx=10, pady=6)
        tk.Label(sliders, text="Severity (%) — 0=mild anomaly   100=full -opia / complete",
                 anchor="w").pack(fill="x")
        tk.Scale(sliders, variable=self.var_sev, from_=0, to=100, orient="horizontal", length=360,
                 command=lambda _=None: self.update_preview()).pack(anchor="w")

        self.str_row = tk.Frame(self.ctrl)
        self.str_row.pack(fill="x", padx=10, pady=6)
        tk.Label(self.str_row, text="Assist strength (%) — only for Assist:", anchor="w").pack(side="left")
        tk.Scale(self.str_row, variable=self.var_strength, from_=0, to=100, orient="horizontal", length=240,
                 command=lambda _=None: self.update_preview()).pack(side="left")

        btns = tk.Frame(self.ctrl)
        btns.pack(pady=8)
        tk.Button(btns, text="Save (S)", command=self.save_as_dialog).pack(side="left", padx=6)
        tk.Button(btns, text="Quit (Q/Esc)", command=self.on_quit).pack(side="left", padx=6)

        self._update_subrows()

    def _update_subrows(self):
        t = self.var_cvd_type.get()
        for r in (getattr(self, "rg_row", None), getattr(self, "tri_row", None)):
            if r is not None:
                r.pack_forget()
        if t == "Red–green color blindness":
            self.rg_row.pack(anchor="w", pady=4)
        if t == "Anomalous Trichromacy":
            self.tri_row.pack(anchor="w", pady=4)

    # ---------- Highlight controls ----------
    def build_highlight_controls(self):
        self.ctrl = tk.Toplevel(self.root)
        self.ctrl.title("Highlight — HSV Selection & Style")
        self.ctrl.resizable(False, False)

        self.var_hc   = tk.IntVar(value=15)
        self.var_hr   = tk.IntVar(value=18)
        self.var_smin = tk.IntVar(value=40)
        self.var_smax = tk.IntVar(value=255)
        self.var_vmin = tk.IntVar(value=40)
        self.var_vmax = tk.IntVar(value=255)
        self.var_style= tk.IntVar(value=1)  # 0 keep, 1 dim, 2 overlay
        self.var_bgdim= tk.IntVar(value=25) # %
        self.var_alpha= tk.IntVar(value=70) # %
        self.var_soft = tk.IntVar(value=2)
        self.var_morph= tk.IntVar(value=0)
        self.var_invert=tk.IntVar(value=0)

        def row(label, var, a, b):
            fr = tk.Frame(self.ctrl)
            fr.pack(fill="x", padx=10, pady=6)
            tk.Label(fr, text=label, width=22, anchor="w").pack(side="left")
            tk.Scale(fr, variable=var, from_=a, to=b, orient="horizontal", length=300,
                     command=lambda _=None: self.update_preview()).pack(side="left")
            tk.Label(fr, textvariable=var, width=5).pack(side="left")

        row("Hue Center (0–179)", self.var_hc, 0, 179)
        row("Hue Range (±0–90)",  self.var_hr, 0, 90)
        row("S min (0–255)",      self.var_smin, 0, 255)
        row("S max (0–255)",      self.var_smax, 0, 255)
        row("V min (0–255)",      self.var_vmin, 0, 255)
        row("V max (0–255)",      self.var_vmax, 0, 255)

        fr_style = tk.Frame(self.ctrl)
        fr_style.pack(fill="x", padx=10, pady=6)
        tk.Label(fr_style, text="Style", width=22, anchor="w").pack(side="left")
        ttk.Radiobutton(fr_style, text="Keep Only",      variable=self.var_style, value=0,
                        command=self.update_preview).pack(side="left", padx=4)
        ttk.Radiobutton(fr_style, text="Dim Background", variable=self.var_style, value=1,
                        command=self.update_preview).pack(side="left", padx=4)
        ttk.Radiobutton(fr_style, text="Overlay",        variable=self.var_style, value=2,
                        command=self.update_preview).pack(side="left", padx=4)

        row("BG Dim % (0–100)", self.var_bgdim, 0, 100)
        row("Overlay Alpha %",  self.var_alpha, 0, 100)
        row("Softness (0–20)",  self.var_soft,  0, 20)
        row("Morph Open (0–10)",self.var_morph, 0, 10)

        fr_inv = tk.Frame(self.ctrl)
        fr_inv.pack(fill="x", padx=10, pady=6)
        tk.Checkbutton(fr_inv, text="Invert selection (everything BUT chosen color)",
                       variable=self.var_invert, onvalue=1, offvalue=0,
                       command=self.update_preview).pack(side="left")

        info = tk.Label(self.ctrl, fg="#444",
                        text="Click the image to pick a color (centers Hue; suggests S/V mins).")
        info.pack(anchor="w", padx=10, pady=6)

        btns = tk.Frame(self.ctrl)
        btns.pack(pady=8)
        tk.Button(btns, text="Save (S)", command=self.save_as_dialog).pack(side="left", padx=6)
        tk.Button(btns, text="Save Mask", command=self.on_save_mask).pack(side="left", padx=6)
        tk.Button(btns, text="Quit (Q/Esc)", command=self.on_quit).pack(side="left", padx=6)

    # ---------- Rendering / Preview ----------

    def current_processed_full(self) -> np.ndarray:
        if self.mode.get() == "cvd":
            cvdt = self.var_cvd_type.get()
            view = self.var_view.get()
            sev  = self.var_sev.get() / 100.0
            strength = self.var_strength.get() / 100.0
            rg_sub = self.var_rg_sub.get()
            tri_var= self.var_tri_variant.get()
            return apply_cvd_pipeline(self.img_full, cvdt, view, sev, strength,
                                      rg_subtype=rg_sub, tri_variant=tri_var)
        # highlight
        hc,hr   = self.var_hc.get(), self.var_hr.get()
        smin    = self.var_smin.get(); smax = self.var_smax.get()
        vmin    = self.var_vmin.get(); vmax = self.var_vmax.get()
        style   = self.var_style.get()
        bgdim   = self.var_bgdim.get()/100.0
        alpha   = self.var_alpha.get()/100.0
        soft    = self.var_soft.get()
        morph   = self.var_morph.get()
        invert  = self.var_invert.get()
        mask01  = make_mask_hsv(self.hsv_full, hc, hr, smin, smax, vmin, vmax, morph, soft, invert)
        if self.show_original:
            return self.img_full.copy()
        return overlay_style(self.img_full, mask01, style, bgdim, alpha, hc, show_mask=self.show_mask)

    def update_preview(self, *_):
        if not hasattr(self, "canvas"):
            return
        W = self.canvas.winfo_width()
        H = self.canvas.winfo_height()
        if W < 2 or H < 2:
            return
        try:
            frame = self.current_processed_full()
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(10, 10, anchor="nw", fill="white",
                                    text=f"Error in preview:\n{e}")
            return

        # Fit-to-window (keep aspect)
        scale = min(W / self.w_full, H / self.h_full)
        vw = max(1, int(self.w_full * scale))
        vh = max(1, int(self.h_full * scale))
        x0 = (W - vw) // 2
        y0 = (H - vh) // 2
        self.view_scale = scale
        self.view_offset = (x0, y0)

        disp = frame if scale == 1.0 else cv2.resize(frame, (vw, vh), interpolation=cv2.INTER_AREA)

        # HUD
        hud = disp.copy()
        lines = ["S=Save (also writes AFTER data)", "O=Original  Space=Mask  Q/Esc=Quit"]
        y = 8
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(hud, (6, y), (6+tw+12, y+th+12), (0,0,0), -1)
            cv2.putText(hud, line, (12, y+th+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            y += th + 14

        rgb = cv2.cvtColor(hud, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.image = img_tk
        self.canvas.delete("all")
        self.canvas.create_image(x0, y0, anchor="nw", image=img_tk)

    # ---------- Events ----------
    def on_canvas_resize(self, event):
        self.update_preview()

    def on_canvas_click(self, event):
        if self.mode.get() != "highlight":
            return
        if not hasattr(self, "view_scale"):
            return
        x_off, y_off = self.view_offset
        x = int((event.x - x_off) / self.view_scale)
        y = int((event.y - y_off) / self.view_scale)
        if x < 0 or y < 0 or x >= self.w_full or y >= self.h_full:
            return
        h,s,v = self.hsv_full[y, x]
        self.last_pick_hsv = (int(h), int(s), int(v))
        if self.var_hr.get() < 12:
            self.var_hr.set(14)
        self.var_hc.set(int(h))
        self.var_smin.set(max(0, int(s) - 45))
        self.var_vmin.set(max(0, int(v) - 45))
        self.update_preview()

    def on_toggle_orig(self, _evt=None):
        self.show_original = not self.show_original
        self.update_preview()

    def on_toggle_mask(self, _evt=None):
        if self.mode.get() == "highlight":
            self.show_mask = not self.show_mask
            self.update_preview()

    def on_quit(self, _evt=None):
        for w in ("ctrl", "viewer"):
            try:
                getattr(self, w).destroy()
            except Exception:
                pass

    # ---------- Save (JPEG) + AUTO AFTER data ----------
    def on_save(self, _evt=None):
        self.save_as_dialog()

    def save_as_dialog(self):
        if not hasattr(self, "img_full"):
            return
        # 1) Ask where to save the EDITED JPEG (user may cancel — AFTER data will still be saved)
        in_dir  = self.src_dir
        in_base = self.src_name
        suffix  = "cvd" if self.mode.get()=="cvd" else "highlight"
        default_name = f"{in_base}_{suffix}_edited.jpg"
        path = filedialog.asksaveasfilename(
            title="Save processed image (optional — AFTER data will be saved regardless)",
            initialdir=in_dir,
            initialfile=default_name,
            defaultextension=".jpg",
            filetypes=[("JPEG image","*.jpg"), ("All files","*.*")]
        )

        # 2) Compute AFTER frame
        after_full = self.current_processed_full()
        # (Round-trip binary -> pixels to honor your original spec)
        buf_after  = pixels_to_binary(after_full)
        after_full = binary_to_pixels(buf_after, after_full.shape, dtype=np.uint8)

        # 3) AUTO-SAVE AFTER data (bits + pixels) beside the SOURCE image
        try:
            self.save_count += 1
            tag = f"{suffix}_after#{self.save_count}"
            after_bits   = os.path.join(self.src_dir, f"{self.src_name}_{tag}_bits.txt")
            after_pixels = os.path.join(self.src_dir, f"{self.src_name}_{tag}_pixels.csv")
            write_bits_per_row_txt(after_full, after_bits)
            write_pixels_csv(after_full, after_pixels)
        except Exception as e:
            messagebox.showwarning("After data not fully saved", f"Error writing AFTER data:\n{e}")

        # 4) If user chose a JPEG path, save the edited image too
        if path:
            ok = cv2.imwrite(path, after_full, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if ok:
                messagebox.showinfo("Saved",
                    f"Edited image:\n{path}\n\n"
                    f"AFTER data written beside source image:\n{after_bits}\n{after_pixels}")
            else:
                messagebox.showerror("Error", f"Could not write image:\n{path}")
        else:
            messagebox.showinfo("AFTER data saved",
                f"AFTER data written beside source image:\n{after_bits}\n{after_pixels}\n\n"
                "No JPEG saved (you canceled the dialog).")

    # ---------- Optional: Save mask (highlight only) ----------
    def on_save_mask(self, _evt=None):
        if self.mode.get() != "highlight":
            return
        in_dir  = self.src_dir
        default_name = f"{self.src_name}_highlight_mask.png"
        path = filedialog.asksaveasfilename(
            title="Save selection mask (PNG)",
            initialdir=in_dir,
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG image","*.png"), ("All files","*.*")]
        )
        if not path:
            return
        hc,hr   = self.var_hc.get(), self.var_hr.get()
        smin    = self.var_smin.get(); smax = self.var_smax.get()
        vmin    = self.var_vmin.get(); vmax = self.var_vmax.get()
        soft    = self.var_soft.get(); morph = self.var_morph.get()
        invert  = self.var_invert.get()
        mask01  = make_mask_hsv(self.hsv_full, hc, hr, smin, smax, vmin, vmax, morph, soft, invert)
        mask8   = np.clip(mask01*255.0, 0, 255).astype(np.uint8)
        ok = cv2.imwrite(path, mask8)
        if ok:
            messagebox.showinfo("Saved", f"Mask saved to:\n{path}")
        else:
            messagebox.showerror("Error", f"Could not write:\n{path}")

# ---------- Main ----------

if __name__ == "__main__":
    ColorToolApp()
