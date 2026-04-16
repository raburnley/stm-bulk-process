
import os
import sys
import ctypes
import subprocess
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib import cm

import spym as SP
from spym.process.filters import destripe, gauss, median, sharpen, mean as mean_filter


APP_TITLE = "SM4 Batch Processor"
WINDOW_SIZE = "1460x920"

CHANNEL_MAP = [
    ("Forward topography", "Topography_Forward"),
    ("Backward topography", "Topography_Backward"),
    ("LIA forward", "LIA_Current_Forward"),
    ("LIA backward", "LIA_Current_Backward"),
    ("Current forward", "Current_Forward"),
    ("Current backward", "Current_Backward"),
]

CUSTOM_CMAP_NAME = "Gwyddion Copper"
CMAP_CHOICES = [
    CUSTOM_CMAP_NAME,
    "gray",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "bone",
    "hot",
    "afmhot",
    "jet",
]

ALIGN_BASELINES = ["median", "mean", "poly"]
DESTRIPE_SIGNS = ["both", "positive", "negative"]
SCALEBAR_POSITIONS = ["upper left", "upper right", "lower left", "lower right"]


def to_nm(scale, unit):
    return scale * {"m": 1e9, "um": 1e3, "nm": 1.0}.get(unit, 1.0)


def set_channel_data(channel, arr):
    if np.shape(arr) != np.shape(channel.data):
        raise ValueError(
            f"Shape mismatch writing back to channel: arr {np.shape(arr)} vs channel {np.shape(channel.data)}"
        )
    channel.data[...] = arr


def normalize_2d(result):
    arr = result[0] if isinstance(result, tuple) else result
    if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[0] in (1, 2):
        arr = arr[0]
    return arr


def safe_percentiles(arr, low_pct, high_pct):
    img = np.asarray(arr, dtype=float)
    finite = np.isfinite(img)
    if not np.any(finite):
        return 0.0, 1.0

    low = float(low_pct)
    high = float(high_pct)
    low = max(0.0, min(100.0, low))
    high = max(0.0, min(100.0, high))

    if high <= low:
        high = min(100.0, low + 1.0)

    vals = img[finite]
    vmin, vmax = np.nanpercentile(vals, [low, high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if vmax <= vmin:
            vmax = vmin + 1.0
    return vmin, vmax


def flatten_lines(arr, axis=1, degree=1):
    """Subtract a polynomial background from each row or column."""
    img = np.array(arr, dtype=float, copy=True)

    if img.ndim != 2:
        return img

    degree = max(0, int(degree))

    if axis == 0:
        img = img.T

    x = np.arange(img.shape[1], dtype=float)
    out = img.copy()

    for i in range(img.shape[0]):
        row = img[i, :]
        mask = np.isfinite(row)
        if np.count_nonzero(mask) < degree + 1:
            continue
        coeff = np.polyfit(x[mask], row[mask], degree)
        trend = np.polyval(coeff, x)
        trend_offset = np.nanmedian(trend[mask]) if np.any(mask) else 0.0
        out[i, :] = row - trend + trend_offset

    if axis == 0:
        out = out.T
    return out


def apply_orientation_transforms(arr, settings):
    out = np.array(arr, copy=True)
    if settings["flip_lr"]:
        out = np.fliplr(out)
    if settings["flip_ud"]:
        out = np.flipud(out)
    return out


def get_colormap(name, invert=False):
    if name == CUSTOM_CMAP_NAME:
        cmap = LinearSegmentedColormap.from_list(
            "GwyddionCopper", ["#530501", "#F28212", "#FEFB18"]
        )
        return cmap.reversed() if invert else cmap

    cmap = cm.get_cmap(name)
    return cmap.reversed() if invert else cmap


def apply_destripe(arr, settings):
    kwargs = {
        "min_length": settings["destripe_min_len"],
        "hard_threshold": settings["destripe_hard_threshold"],
        "soft_threshold": settings["destripe_soft_threshold"],
        "sign": settings["destripe_sign"],
    }

    try:
        kwargs["rel_threshold"] = settings["destripe_rel_threshold"]
        result = destripe(arr, **kwargs)
    except TypeError:
        kwargs.pop("rel_threshold", None)
        result = destripe(arr, **kwargs)

    return normalize_2d(result)



def build_export_names(channel_stem, base_name):
    file_stem = f"{channel_stem}_{base_name}"
    title_name = f"{channel_stem.replace('_', ' ').title()} {base_name}"
    return title_name, file_stem



def _clean_unit_text(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            value = str(value)
    value = str(value).strip()
    return value or None


def infer_z_unit(channel):
    candidate_keys = [
        "RHK_Zunits",
        "RHK_Zunit",
        "Zunits",
        "Zunit",
        "units",
        "unit",
    ]
    for key in candidate_keys:
        value = _clean_unit_text(channel.attrs.get(key))
        if value:
            return value
    return None




SI_PREFIX_FACTORS = {
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "µ": 1e-6,
    "m": 1e-3,
    "": 1.0,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
}


def parse_si_unit(unit_text):
    unit = _clean_unit_text(unit_text)
    if unit is None:
        return None

    unit = unit.replace("μ", "µ")

    if unit in {"", "px", "arb.", "a.u.", "counts"}:
        return None

    if len(unit) >= 2:
        prefix = unit[0]
        base = unit[1:]
        if prefix in SI_PREFIX_FACTORS and base:
            return SI_PREFIX_FACTORS[prefix], base

    return 1.0, unit


def choose_engineering_prefix(base_values):
    finite = np.asarray(base_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0, ""

    magnitude = float(np.nanpercentile(np.abs(finite), 95))
    if magnitude == 0.0:
        return 1.0, ""

    candidates = [
        ("p", 1e-12),
        ("n", 1e-9),
        ("µ", 1e-6),
        ("m", 1e-3),
        ("", 1.0),
        ("k", 1e3),
        ("M", 1e6),
        ("G", 1e9),
    ]

    valid = []
    for prefix, factor in candidates:
        scaled = magnitude / factor
        if 1.0 <= scaled < 1000.0:
            valid.append((prefix, factor, scaled))

    if valid:
        prefix, factor, _ = min(valid, key=lambda item: abs(np.log10(item[2]) - 1.5))
        return factor, prefix

    prefix, factor = min(
        candidates,
        key=lambda item: abs(np.log10(max(magnitude / item[1], 1e-30)) - 2.0),
    )
    return factor, prefix


def scale_image_and_unit_for_display(img, unit_text):
    parsed = parse_si_unit(unit_text)
    if parsed is None:
        return np.asarray(img, dtype=float), _clean_unit_text(unit_text)

    input_factor, base_unit = parsed
    base_values = np.asarray(img, dtype=float) * input_factor
    display_factor, display_prefix = choose_engineering_prefix(base_values)
    scaled_img = base_values / display_factor
    display_unit = f"{display_prefix}{base_unit}"
    return scaled_img, display_unit

def get_plot_metadata(channel):
    try:
        x_scale = to_nm(channel.attrs.get("RHK_Xscale", 1), channel.attrs.get("RHK_Xunits", "m"))
        y_scale = to_nm(channel.attrs.get("RHK_Yscale", 1), channel.attrs.get("RHK_Yunits", "m"))
        x_size = channel.attrs.get("RHK_Xsize", channel.data.shape[1])
        y_size = channel.attrs.get("RHK_Ysize", channel.data.shape[0])
        extent = [0, x_scale * x_size, 0, y_scale * y_size]
        x_span_nm = float(extent[1] - extent[0])
        y_span_nm = float(extent[3] - extent[2])
        return {
            "extent": extent,
            "xlabel": "X (nm)",
            "ylabel": "Y (nm)",
            "x_span_nm": x_span_nm,
            "y_span_nm": y_span_nm,
            "units": "nm",
            "z_unit": infer_z_unit(channel),
        }
    except Exception:
        return {
            "extent": None,
            "xlabel": "X (px)",
            "ylabel": "Y (px)",
            "x_span_nm": None,
            "y_span_nm": None,
            "units": "px",
            "z_unit": infer_z_unit(channel),
        }


def round_to_nearest_five(value):
    return int(5 * np.floor((float(value) / 5.0) + 0.5))


def compute_scalebar_length_nm(x_span_nm):
    if x_span_nm is None or x_span_nm <= 0:
        return None
    return max(5, round_to_nearest_five(x_span_nm / 3.0))



def add_scale_bar(ax, plot_meta, position):
    extent = plot_meta.get("extent")
    if extent is None:
        return False

    x_min, x_max, y_min, y_max = [float(v) for v in extent]
    x_span_nm = abs(x_max - x_min)
    y_span_nm = abs(y_max - y_min)
    if x_span_nm <= 0 or y_span_nm <= 0:
        return False

    bar_length_nm = compute_scalebar_length_nm(x_span_nm)
    if bar_length_nm is None:
        return False

    # Anchor the bar to the visible plot area (axes box), not to image/data corners.
    # Width is still tied to the real x-span in nm.
    width_frac = max(0.02, min(0.90, bar_length_nm / x_span_nm))

    # Keep the bar 5:1 in displayed shape. Convert the height to an axes fraction
    # using the physical y-span so the displayed size stays reasonable.
    bar_height_nm = max(bar_length_nm / 5.0, y_span_nm * 0.015)
    height_frac = max(0.01, min(0.25, bar_height_nm / y_span_nm))

    margin_x = 0.05
    margin_y = 0.05

    pos = (position or "lower right").strip().lower()

    if "left" in pos:
        rect_x = margin_x
    else:
        rect_x = 1.0 - margin_x - width_frac

    if "upper" in pos:
        rect_y = 1.0 - margin_y - height_frac
    else:
        rect_y = margin_y

    rect = Rectangle(
        (rect_x, rect_y),
        width_frac,
        height_frac,
        transform=ax.transAxes,
        facecolor="#d9d9d9",
        edgecolor="#d9d9d9",
        linewidth=1.0,
        zorder=5,
        clip_on=False,
    )
    ax.add_patch(rect)
    ax.text(
        rect_x + width_frac / 2.0,
        rect_y + height_frac / 2.0,
        f"{int(bar_length_nm)} nm",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="black",
        fontsize=10,
        fontweight="semibold",
        zorder=6,
        clip_on=False,
    )
    return True



def process_channel_for_preview(channel, settings):

    arr = np.array(channel.data, copy=True)

    if settings["plane"]:
        channel.spym.plane()
        arr = np.array(channel.data, copy=True)

    if settings["flatten_rows"]:
        arr = flatten_lines(arr, axis=1, degree=settings["flatten_degree"])
        set_channel_data(channel, arr)

    if settings["flatten_cols"]:
        arr = flatten_lines(arr, axis=0, degree=settings["flatten_degree"])
        set_channel_data(channel, arr)

    if settings["alignment"]:
        for _ in range(2):
            align_kwargs = {
                "baseline": settings["align_baseline"],
                "axis": settings["align_axis"],
            }
            if settings["align_baseline"] == "poly":
                align_kwargs["poly_degree"] = settings["align_poly_degree"]

            try:
                channel.spym.align(**align_kwargs)
            except TypeError:
                align_kwargs.pop("poly_degree", None)
                channel.spym.align(**align_kwargs)

            arr = np.array(channel.data, copy=True)

    if settings["destripe"]:
        arr = apply_destripe(arr, settings)
        set_channel_data(channel, arr)

    if settings["fix_zero"]:
        try:
            channel.spym.fixzero(to_mean=settings["fixzero_to_mean"])
        except TypeError:
            channel.spym.fixzero()
        arr = np.array(channel.data, copy=True)

    if settings["gaussian"]:
        arr = gauss(arr, size=max(1, settings["gauss_size"]))

    if settings["median"]:
        arr = median(arr, size=max(1, settings["median_size"]))

    if settings["mean_filter"]:
        arr = mean_filter(arr, size=max(1, settings["mean_size"]))

    if settings["sharpen"]:
        arr = sharpen(arr, size=max(1, settings["sharpen_size"]), alpha=settings["sharpen_alpha"])

    arr = apply_orientation_transforms(arr, settings)
    return arr


def process_and_save_channel(channel, title_name, file_stem, folder, settings, cmap, log_fn):
    plot_meta = get_plot_metadata(channel)

    ch_copy = channel.copy(deep=True)
    try:
        img = process_channel_for_preview(ch_copy, settings)
    except Exception as e:
        log_fn(f"  Warning: processing failed for {title_name} ({e})")
        img = np.array(ch_copy.data, copy=True)
        img = img - np.nanmin(img)

    display_img, display_z_unit = scale_image_and_unit_for_display(img, plot_meta.get("z_unit"))
    vmin, vmax = safe_percentiles(display_img, settings["contrast_low_pct"], settings["contrast_high_pct"])

    fig = Figure(figsize=(6, 6), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        display_img,
        cmap=cmap,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        extent=plot_meta["extent"],
        aspect="equal",
    )
    ax.set_title(title_name, fontsize=10)

    if settings["show_scale_bar"]:
        if not add_scale_bar(ax, plot_meta, settings["scalebar_position"]):
            log_fn(f"  Scale bar skipped for {title_name} (missing physical scale metadata)")
        ax.axis("off")
    else:
        ax.set_xlabel(plot_meta["xlabel"])
        ax.set_ylabel(plot_meta["ylabel"])
        ax.axis("on")

    if settings["show_colorbar"]:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if display_z_unit:
            cbar.ax.set_title(display_z_unit, pad=6, fontsize=9)
        try:
            cbar.formatter.set_useOffset(False)
            cbar.formatter.set_scientific(False)
        except Exception:
            pass
        cbar.update_ticks()

    save_path = os.path.join(folder, f"{file_stem}.png")
    fig.savefig(save_path, dpi=1200, bbox_inches="tight", pad_inches=0.05)
    log_fn(f"  Saved {save_path}")


def save_stitched_side_by_side(fwd_png, bwd_png, out_path, log_fn):
    import matplotlib.pyplot as plt

    try:
        img_f = plt.imread(fwd_png)
        img_b = plt.imread(bwd_png)
    except Exception as e:
        log_fn(f"  Composite read failed: {e}")
        return

    if img_f.shape != img_b.shape:
        log_fn("  Skipped composite (shape mismatch)")
        return

    stitched = np.hstack((img_f, img_b))
    fig = Figure(figsize=(6, 6), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.imshow(stitched)
    ax.axis("off")
    fig.savefig(out_path, dpi=1200, bbox_inches="tight", pad_inches=0.05)
    log_fn(f"  Saved composite: {out_path}")


def open_folder_in_os(folder_path):
    if not folder_path:
        return

    if sys.platform.startswith("win"):
        os.startfile(folder_path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", folder_path])
    else:
        subprocess.Popen(["xdg-open", folder_path])


class App(tk.Tk):
    CONFIG_FIELDS = [
        ("plane", "int"),
        ("fix_zero", "int"),
        ("fixzero_to_mean", "int"),
        ("alignment", "int"),
        ("destripe", "int"),
        ("gaussian", "int"),
        ("median", "int"),
        ("mean_filter", "int"),
        ("sharpen", "int"),
        ("flatten_rows", "int"),
        ("flatten_cols", "int"),
        ("flip_lr", "int"),
        ("flip_ud", "int"),
        ("align_axis", "int"),
        ("align_baseline", "str"),
        ("align_poly_degree", "int"),
        ("destripe_min_len", "int"),
        ("destripe_hard_threshold", "float"),
        ("destripe_soft_threshold", "float"),
        ("destripe_rel_threshold", "float"),
        ("destripe_sign", "str"),
        ("gauss_size", "int"),
        ("median_size", "int"),
        ("mean_size", "int"),
        ("sharpen_size", "int"),
        ("sharpen_alpha", "float"),
        ("flatten_degree", "int"),
        ("cmap_name", "str"),
        ("invert_cmap", "int"),
        ("contrast_low_pct", "float"),
        ("contrast_high_pct", "float"),
        ("show_colorbar", "int"),
        ("show_scale_bar", "int"),
        ("scalebar_position", "str"),
    ]

    COLORS = {
        "bg": "#f3f6fb",
        "panel": "#ffffff",
        "soft": "#e8eef7",
        "border": "#d4deeb",
        "text": "#1f2937",
        "muted": "#667085",
        "accent": "#2b6cb0",
        "accent_hover": "#245a93",
        "danger": "#c0392b",
        "danger_hover": "#a93226",
        "success": "#2f855a",
        "success_hover": "#276749",
        "selected": "#dbeafe",
        "tab_bg": "#edf2f7",
    }

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(WINDOW_SIZE)
        self.minsize(1320, 840)
        self.configure(bg=self.COLORS["bg"])

        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self._configure_styles()

        self.file_paths = []
        self.file_display_map = {}
        self.dataset_cache = {}
        self.current_ds = None
        self.current_file = None
        self.channel_map = CHANNEL_MAP

        self._build_variables()
        self._build_layout()
        self._bind_shortcuts()
        self._update_navigation_state()

    def _configure_styles(self):
        c = self.COLORS
        self.option_add("*Font", ("Segoe UI", 10))

        self.style.configure(
            ".",
            background=c["bg"],
            foreground=c["text"],
            fieldbackground=c["panel"],
        )
        self.style.configure("App.TFrame", background=c["bg"])
        self.style.configure("Card.TFrame", background=c["panel"], relief="flat")
        self.style.configure("Toolbar.TFrame", background=c["panel"])
        self.style.configure("Muted.TLabel", background=c["bg"], foreground=c["muted"], font=("Segoe UI", 9))
        self.style.configure("Header.TLabel", background=c["bg"], foreground=c["text"], font=("Segoe UI Semibold", 20))
        self.style.configure("SubHeader.TLabel", background=c["bg"], foreground=c["muted"], font=("Segoe UI", 10))
        self.style.configure("Section.TLabel", background=c["panel"], foreground=c["text"], font=("Segoe UI Semibold", 11))
        self.style.configure("Value.TLabel", background=c["soft"], foreground=c["text"], font=("Segoe UI Semibold", 9))
        self.style.configure("Panel.TLabelframe", background=c["panel"], bordercolor=c["border"], relief="solid")
        self.style.configure("Panel.TLabelframe.Label", background=c["panel"], foreground=c["text"], font=("Segoe UI Semibold", 10))
        self.style.configure("TNotebook", background=c["bg"], borderwidth=0, tabmargins=[0, 0, 0, 0])
        self.style.configure(
            "TNotebook.Tab",
            background=c["tab_bg"],
            foreground=c["text"],
            padding=(14, 8),
            font=("Segoe UI Semibold", 10),
            borderwidth=0,
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", c["panel"]), ("active", c["soft"])],
            foreground=[("selected", c["accent"])],
        )
        self.style.configure(
            "TButton",
            padding=(10, 7),
            relief="flat",
            borderwidth=0,
            background=c["soft"],
            foreground=c["text"],
        )
        self.style.map(
            "TButton",
            background=[("active", c["selected"]), ("disabled", c["soft"])],
            foreground=[("disabled", "#9aa4b2")],
        )
        self.style.configure("Accent.TButton", background=c["accent"], foreground="white")
        self.style.map("Accent.TButton", background=[("active", c["accent_hover"]), ("disabled", "#8db2dc")])
        self.style.configure("Success.TButton", background=c["success"], foreground="white")
        self.style.map("Success.TButton", background=[("active", c["success_hover"]), ("disabled", "#90c3ab")])
        self.style.configure("Danger.TButton", background=c["danger"], foreground="white")
        self.style.map("Danger.TButton", background=[("active", c["danger_hover"]), ("disabled", "#dda49e")])
        self.style.configure("Tool.TButton", padding=(8, 6))
        self.style.configure("TEntry", padding=5)
        self.style.configure("TCombobox", padding=4)
        self.style.configure("TCheckbutton", background=c["panel"], foreground=c["text"])
        self.style.map("TCheckbutton", background=[("active", c["panel"])])
        self.style.configure("Vertical.TScrollbar", background=c["soft"], troughcolor=c["bg"], borderwidth=0, arrowsize=12)

    def _build_variables(self):
        self.file_var = tk.StringVar()
        self.channel_var = tk.StringVar(value=self.channel_map[0][0])
        self.cmap_name = tk.StringVar(value=CUSTOM_CMAP_NAME)
        self.invert_cmap = tk.IntVar(value=0)
        self.file_position_var = tk.StringVar(value="File 0 / 0")
        self.channel_position_var = tk.StringVar(value="Channel 0 / 0")
        self.path_var = tk.StringVar(value="No file loaded.")
        self.status_bar_var = tk.StringVar(value="Ready.")
        self.preview_title_var = tk.StringVar(value="No image loaded")
        self.file_name_var = tk.StringVar(value="—")
        self.channel_name_var = tk.StringVar(value=self.channel_map[0][0])

        self.plane = tk.IntVar(value=1)
        self.fix_zero = tk.IntVar(value=1)
        self.fixzero_to_mean = tk.IntVar(value=0)
        self.alignment = tk.IntVar(value=1)
        self.destripe = tk.IntVar(value=1)
        self.gaussian = tk.IntVar(value=0)
        self.median_filter = tk.IntVar(value=0)
        self.mean_filter_var = tk.IntVar(value=0)
        self.sharpen_filter = tk.IntVar(value=0)
        self.flatten_rows = tk.IntVar(value=0)
        self.flatten_cols = tk.IntVar(value=0)
        self.flip_lr = tk.IntVar(value=0)
        self.flip_ud = tk.IntVar(value=0)

        self.align_axis = tk.IntVar(value=1)
        self.align_baseline = tk.StringVar(value="median")
        self.align_poly_degree = tk.IntVar(value=1)

        self.destripe_min = tk.IntVar(value=10)
        self.destripe_hard = tk.DoubleVar(value=0.10)
        self.destripe_soft = tk.DoubleVar(value=0.05)
        self.destripe_rel = tk.DoubleVar(value=0.50)
        self.destripe_sign = tk.StringVar(value="both")

        self.gauss_size = tk.IntVar(value=5)
        self.median_size = tk.IntVar(value=5)
        self.mean_size = tk.IntVar(value=5)
        self.sharpen_size = tk.IntVar(value=5)
        self.sharpen_alpha = tk.DoubleVar(value=30.0)
        self.flatten_degree = tk.IntVar(value=1)

        self.contrast_low_pct = tk.DoubleVar(value=5.0)
        self.contrast_high_pct = tk.DoubleVar(value=93.0)
        self.show_colorbar = tk.IntVar(value=0)
        self.show_scale_bar = tk.IntVar(value=0)
        self.scalebar_position = tk.StringVar(value="lower right")

    def _variable_map(self):
        return {
            "plane": self.plane,
            "fix_zero": self.fix_zero,
            "fixzero_to_mean": self.fixzero_to_mean,
            "alignment": self.alignment,
            "destripe": self.destripe,
            "gaussian": self.gaussian,
            "median": self.median_filter,
            "mean_filter": self.mean_filter_var,
            "sharpen": self.sharpen_filter,
            "flatten_rows": self.flatten_rows,
            "flatten_cols": self.flatten_cols,
            "flip_lr": self.flip_lr,
            "flip_ud": self.flip_ud,
            "align_axis": self.align_axis,
            "align_baseline": self.align_baseline,
            "align_poly_degree": self.align_poly_degree,
            "destripe_min_len": self.destripe_min,
            "destripe_hard_threshold": self.destripe_hard,
            "destripe_soft_threshold": self.destripe_soft,
            "destripe_rel_threshold": self.destripe_rel,
            "destripe_sign": self.destripe_sign,
            "gauss_size": self.gauss_size,
            "median_size": self.median_size,
            "mean_size": self.mean_size,
            "sharpen_size": self.sharpen_size,
            "sharpen_alpha": self.sharpen_alpha,
            "flatten_degree": self.flatten_degree,
            "cmap_name": self.cmap_name,
            "invert_cmap": self.invert_cmap,
            "contrast_low_pct": self.contrast_low_pct,
            "contrast_high_pct": self.contrast_high_pct,
            "show_colorbar": self.show_colorbar,
            "show_scale_bar": self.show_scale_bar,
            "scalebar_position": self.scalebar_position,
        }

    def _build_layout(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, style="App.TFrame", padding=(18, 16, 18, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text=APP_TITLE, style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Batch preview, cleanup, browsing, and export for RHK .sm4 data",
            style="SubHeader.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        content = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        content.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 12))

        sidebar = ttk.Frame(content, style="App.TFrame", padding=(0, 0, 10, 0))
        workspace = ttk.Frame(content, style="App.TFrame", padding=(10, 0, 0, 0))
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(0, weight=1)
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(1, weight=1)

        content.add(sidebar, weight=0)
        content.add(workspace, weight=1)

        self._build_sidebar(sidebar)
        self._build_workspace(workspace)

        footer = ttk.Frame(self, style="App.TFrame", padding=(18, 2, 18, 12))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(1, weight=1)

        ttk.Label(footer, text="Status", style="Muted.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Label(footer, textvariable=self.status_bar_var, style="Muted.TLabel").grid(row=0, column=1, sticky="w")

    def _build_sidebar(self, parent):
        shell = ttk.Frame(parent, style="Card.TFrame", padding=10)
        shell.grid(row=0, column=0, sticky="nsew")
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(shell)
        notebook.grid(row=0, column=0, sticky="nsew")

        processing_tab = ttk.Frame(notebook, style="Card.TFrame", padding=12)
        view_tab = ttk.Frame(notebook, style="Card.TFrame", padding=12)
        session_tab = ttk.Frame(notebook, style="Card.TFrame", padding=12)

        notebook.add(processing_tab, text="Processing")
        notebook.add(view_tab, text="Display")
        notebook.add(session_tab, text="Session")

        self._build_processing_tab(processing_tab)
        self._build_view_tab(view_tab)
        self._build_session_tab(session_tab)

    def _build_processing_tab(self, parent):
        parent.columnconfigure(0, weight=1)

        leveling = ttk.LabelFrame(parent, text="Leveling and cleanup", style="Panel.TLabelframe", padding=10)
        leveling.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self._check(leveling, "Plane", self.plane, 0, 0)
        self._check(leveling, "Fix zero", self.fix_zero, 1, 0)
        self._check(leveling, "Fix zero to mean", self.fixzero_to_mean, 1, 1)
        self._check(leveling, "Alignment", self.alignment, 2, 0)
        self._check(leveling, "Destripe", self.destripe, 5, 0)
        self._check(leveling, "Flatten rows", self.flatten_rows, 10, 0)
        self._check(leveling, "Flatten columns", self.flatten_cols, 11, 0)

        self._labeled_entry(leveling, "Axis", self.align_axis, 2, width=8)
        self._labeled_combo(leveling, "Baseline", self.align_baseline, ALIGN_BASELINES, 3, width=10)
        self._labeled_entry(leveling, "Poly degree", self.align_poly_degree, 4, width=8)
        self._labeled_entry(leveling, "Min length", self.destripe_min, 5, width=8)
        self._labeled_entry(leveling, "Hard thr", self.destripe_hard, 6, width=8)
        self._labeled_entry(leveling, "Soft thr", self.destripe_soft, 7, width=8)
        self._labeled_entry(leveling, "Rel thr", self.destripe_rel, 8, width=8)
        self._labeled_combo(leveling, "Sign", self.destripe_sign, DESTRIPE_SIGNS, 9, width=10)
        self._labeled_entry(leveling, "Flatten degree", self.flatten_degree, 10, width=8)

        filters = ttk.LabelFrame(parent, text="Spatial filters", style="Panel.TLabelframe", padding=10)
        filters.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self._check(filters, "Gaussian", self.gaussian, 0, 0)
        self._check(filters, "Median", self.median_filter, 1, 0)
        self._check(filters, "Mean", self.mean_filter_var, 2, 0)
        self._check(filters, "Sharpen", self.sharpen_filter, 3, 0)

        self._labeled_entry(filters, "Gaussian size", self.gauss_size, 0, width=8)
        self._labeled_entry(filters, "Median size", self.median_size, 1, width=8)
        self._labeled_entry(filters, "Mean size", self.mean_size, 2, width=8)
        self._labeled_entry(filters, "Sharpen size", self.sharpen_size, 3, width=8)
        self._labeled_entry(filters, "Sharpen alpha", self.sharpen_alpha, 4, width=8)

        orientation = ttk.LabelFrame(parent, text="Orientation", style="Panel.TLabelframe", padding=10)
        orientation.grid(row=2, column=0, sticky="ew")
        self._check(orientation, "Flip left-right", self.flip_lr, 0, 0)
        self._check(orientation, "Flip up-down", self.flip_ud, 1, 0)

    def _build_view_tab(self, parent):
        parent.columnconfigure(0, weight=1)

        display = ttk.LabelFrame(parent, text="Preview display", style="Panel.TLabelframe", padding=10)
        display.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(display, text="Colormap", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Combobox(
            display,
            textvariable=self.cmap_name,
            values=CMAP_CHOICES,
            state="readonly",
            width=22,
        ).grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=(0, 6))
        self._check(display, "Invert colormap", self.invert_cmap, 1, 0)
        self._labeled_entry(display, "Low percentile", self.contrast_low_pct, 2, width=8)
        self._labeled_entry(display, "High percentile", self.contrast_high_pct, 3, width=8)
        self._check(display, "Export colorbar", self.show_colorbar, 4, 0)

        scalebar_row = ttk.Frame(display, style="Toolbar.TFrame")
        scalebar_row.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Checkbutton(scalebar_row, text="Scale bar", variable=self.show_scale_bar).grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            scalebar_row,
            textvariable=self.scalebar_position,
            values=SCALEBAR_POSITIONS,
            state="readonly",
            width=14,
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))

        display.columnconfigure(1, weight=1)

        help_box = ttk.LabelFrame(parent, text="Tips", style="Panel.TLabelframe", padding=10)
        help_box.grid(row=1, column=0, sticky="nsew")
        help_box.columnconfigure(0, weight=1)
        tips = (
            "• Use Apply after changing parameters.\n"
            "• Ctrl+Left / Ctrl+Right: previous or next file.\n"
            "• Ctrl+Up / Ctrl+Down: previous or next channel.\n"
            "• Scale bar appears in preview and export; colorbar is export only.\n"
            "• Save Config stores the current control values as defaults."
        )
        ttk.Label(help_box, text=tips, justify="left", background=self.COLORS["panel"], foreground=self.COLORS["muted"]).grid(
            row=0, column=0, sticky="w"
        )

    def _build_session_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        actions = ttk.LabelFrame(parent, text="Batch and configuration", style="Panel.TLabelframe", padding=10)
        actions.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

        ttk.Button(actions, text="Load .sm4 Files", style="Accent.TButton", command=self.load_files).grid(
            row=0, column=0, sticky="ew", padx=(0, 6), pady=(0, 8)
        )
        ttk.Button(actions, text="Run Batch Export", style="Success.TButton", command=self.bulk_process).grid(
            row=0, column=1, sticky="ew", padx=(6, 0), pady=(0, 8)
        )
        ttk.Button(actions, text="Load Config", command=self.load_config).grid(
            row=1, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(actions, text="Save Config", command=self.save_config).grid(
            row=1, column=1, sticky="ew", padx=(6, 0)
        )

        summary = ttk.LabelFrame(parent, text="Current item", style="Panel.TLabelframe", padding=10)
        summary.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        summary.columnconfigure(1, weight=1)

        ttk.Label(summary, text="File", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(summary, textvariable=self.file_name_var, background=self.COLORS["panel"], foreground=self.COLORS["text"]).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Label(summary, text="Channel", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(summary, textvariable=self.channel_name_var, background=self.COLORS["panel"], foreground=self.COLORS["text"]).grid(
            row=1, column=1, sticky="w", pady=(6, 0)
        )

        status = ttk.LabelFrame(parent, text="Status log", style="Panel.TLabelframe", padding=8)
        status.grid(row=2, column=0, sticky="nsew")
        status.columnconfigure(0, weight=1)
        status.rowconfigure(0, weight=1)

        text_shell = ttk.Frame(status, style="Card.TFrame")
        text_shell.grid(row=0, column=0, sticky="nsew")
        text_shell.columnconfigure(0, weight=1)
        text_shell.rowconfigure(0, weight=1)

        self.log = tk.Text(
            text_shell,
            height=16,
            wrap="word",
            relief="flat",
            borderwidth=0,
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
            insertbackground=self.COLORS["text"],
            selectbackground=self.COLORS["selected"],
            font=("Consolas", 10),
            padx=8,
            pady=8,
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        self.log.configure(state=tk.DISABLED)

        scroll = ttk.Scrollbar(text_shell, orient="vertical", command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)

    def _build_workspace(self, parent):
        top_card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        top_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        top_card.columnconfigure(0, weight=1)

        ttk.Label(top_card, text="Preview controls", style="Section.TLabel").grid(row=0, column=0, sticky="w")

        file_row = ttk.Frame(top_card, style="Toolbar.TFrame")
        file_row.grid(row=1, column=0, sticky="ew", pady=(12, 8))
        file_row.columnconfigure(2, weight=1)

        ttk.Label(file_row, text="File", background=self.COLORS["panel"], foreground=self.COLORS["muted"]).grid(row=0, column=0, sticky="w")
        self.prev_file_btn = ttk.Button(file_row, text="◀ Prev", style="Tool.TButton", command=self.prev_file, width=9)
        self.prev_file_btn.grid(row=0, column=1, sticky="w", padx=(10, 6))

        self.file_menu = ttk.Combobox(file_row, textvariable=self.file_var, state="readonly", width=52)
        self.file_menu.grid(row=0, column=2, sticky="ew", padx=(0, 6))
        self.file_menu.bind("<<ComboboxSelected>>", lambda e: self.change_file())

        self.next_file_btn = ttk.Button(file_row, text="Next ▶", style="Tool.TButton", command=self.next_file, width=9)
        self.next_file_btn.grid(row=0, column=3, sticky="w", padx=(0, 12))

        self.remove_file_btn = ttk.Button(file_row, text="Remove from Batch", command=self.remove_current_from_batch)
        self.remove_file_btn.grid(row=0, column=4, sticky="e", padx=(0, 6))

        self.delete_file_btn = ttk.Button(file_row, text="Delete File", style="Danger.TButton", command=self.delete_current_file)
        self.delete_file_btn.grid(row=0, column=5, sticky="e", padx=(0, 6))

        self.open_folder_btn = ttk.Button(file_row, text="Open Folder", command=self.open_containing_folder)
        self.open_folder_btn.grid(row=0, column=6, sticky="e")

        channel_row = ttk.Frame(top_card, style="Toolbar.TFrame")
        channel_row.grid(row=2, column=0, sticky="ew")
        channel_row.columnconfigure(2, weight=1)

        ttk.Label(channel_row, text="Channel", background=self.COLORS["panel"], foreground=self.COLORS["muted"]).grid(row=0, column=0, sticky="w")
        self.prev_channel_btn = ttk.Button(channel_row, text="▲ Prev", style="Tool.TButton", command=self.prev_channel, width=9)
        self.prev_channel_btn.grid(row=0, column=1, sticky="w", padx=(10, 6))

        self.channel_menu = ttk.Combobox(
            channel_row,
            textvariable=self.channel_var,
            values=[c[0] for c in self.channel_map],
            state="readonly",
            width=38,
        )
        self.channel_menu.grid(row=0, column=2, sticky="w", padx=(0, 6))
        self.channel_menu.bind("<<ComboboxSelected>>", lambda e: self.on_channel_selected())

        self.next_channel_btn = ttk.Button(channel_row, text="Next ▼", style="Tool.TButton", command=self.next_channel, width=9)
        self.next_channel_btn.grid(row=0, column=3, sticky="w", padx=(0, 12))

        self.apply_btn = ttk.Button(channel_row, text="Apply", style="Accent.TButton", command=self.apply_preview, width=12)
        self.apply_btn.grid(row=0, column=4, sticky="e")

        preview_card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        preview_card.grid(row=1, column=0, sticky="nsew")
        preview_card.columnconfigure(0, weight=1)
        preview_card.rowconfigure(1, weight=1)

        meta_row = ttk.Frame(preview_card, style="Toolbar.TFrame")
        meta_row.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        meta_row.columnconfigure(1, weight=1)

        ttk.Label(meta_row, textvariable=self.preview_title_var, style="Section.TLabel").grid(row=0, column=0, sticky="w")
        badges = ttk.Frame(meta_row, style="Toolbar.TFrame")
        badges.grid(row=0, column=1, sticky="e")
        ttk.Label(badges, textvariable=self.file_position_var, style="Value.TLabel", padding=(10, 4)).grid(row=0, column=0, padx=(0, 8))
        ttk.Label(badges, textvariable=self.channel_position_var, style="Value.TLabel", padding=(10, 4)).grid(row=0, column=1)

        self.fig = Figure(figsize=(7.4, 7.4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=preview_card)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        details = ttk.Frame(preview_card, style="Toolbar.TFrame")
        details.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        details.columnconfigure(1, weight=1)

        ttk.Label(details, text="Path", style="Muted.TLabel").grid(row=0, column=0, sticky="nw", padx=(0, 8))
        ttk.Label(
            details,
            textvariable=self.path_var,
            background=self.COLORS["panel"],
            foreground=self.COLORS["text"],
            justify="left",
            wraplength=900,
        ).grid(row=0, column=1, sticky="ew")

    def _bind_shortcuts(self):
        self.bind("<Control-Left>", lambda e: self.prev_file())
        self.bind("<Control-Right>", lambda e: self.next_file())
        self.bind("<Control-Up>", lambda e: self.prev_channel())
        self.bind("<Control-Down>", lambda e: self.next_channel())

    def _check(self, parent, text, variable, row, column):
        ttk.Checkbutton(parent, text=text, variable=variable).grid(row=row, column=column, sticky="w", pady=3)

    def _labeled_entry(self, parent, label, variable, row, width=10, label_col=1, entry_col=2):
        ttk.Label(parent, text=label, background=self.COLORS["panel"], foreground=self.COLORS["muted"]).grid(
            row=row, column=label_col, sticky="e", padx=(12, 6), pady=(3, 0)
        )
        ttk.Entry(parent, textvariable=variable, width=width, justify="center").grid(
            row=row, column=entry_col, sticky="w", pady=(3, 0)
        )

    def _labeled_combo(self, parent, label, variable, values, row, width=10, label_col=1, combo_col=2):
        ttk.Label(parent, text=label, background=self.COLORS["panel"], foreground=self.COLORS["muted"]).grid(
            row=row, column=label_col, sticky="e", padx=(12, 6), pady=(3, 0)
        )
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=width).grid(
            row=row, column=combo_col, sticky="w", pady=(3, 0)
        )

    def log_line(self, msg):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)
        self.status_bar_var.set(msg)
        self.update_idletasks()

    def current_settings(self):
        return {
            "plane": self.plane.get() == 1,
            "fix_zero": self.fix_zero.get() == 1,
            "fixzero_to_mean": self.fixzero_to_mean.get() == 1,
            "alignment": self.alignment.get() == 1,
            "destripe": self.destripe.get() == 1,
            "gaussian": self.gaussian.get() == 1,
            "median": self.median_filter.get() == 1,
            "mean_filter": self.mean_filter_var.get() == 1,
            "sharpen": self.sharpen_filter.get() == 1,
            "flatten_rows": self.flatten_rows.get() == 1,
            "flatten_cols": self.flatten_cols.get() == 1,
            "flip_lr": self.flip_lr.get() == 1,
            "flip_ud": self.flip_ud.get() == 1,
            "align_axis": max(0, int(self.align_axis.get())),
            "align_baseline": self.align_baseline.get(),
            "align_poly_degree": max(0, int(self.align_poly_degree.get())),
            "destripe_min_len": max(1, int(self.destripe_min.get())),
            "destripe_hard_threshold": max(0.0, float(self.destripe_hard.get())),
            "destripe_soft_threshold": max(0.0, float(self.destripe_soft.get())),
            "destripe_rel_threshold": max(0.0, float(self.destripe_rel.get())),
            "destripe_sign": self.destripe_sign.get() or "both",
            "gauss_size": max(1, int(self.gauss_size.get())),
            "median_size": max(1, int(self.median_size.get())),
            "mean_size": max(1, int(self.mean_size.get())),
            "sharpen_size": max(1, int(self.sharpen_size.get())),
            "sharpen_alpha": max(0.0, float(self.sharpen_alpha.get())),
            "flatten_degree": max(0, int(self.flatten_degree.get())),
            "cmap_name": self.cmap_name.get(),
            "invert_cmap": self.invert_cmap.get() == 1,
            "contrast_low_pct": float(self.contrast_low_pct.get()),
            "contrast_high_pct": float(self.contrast_high_pct.get()),
            "show_colorbar": self.show_colorbar.get() == 1,
            "show_scale_bar": self.show_scale_bar.get() == 1,
            "scalebar_position": (self.scalebar_position.get() or "lower right").strip().lower(),
        }

    def current_colormap(self):
        return get_colormap(self.cmap_name.get(), invert=self.invert_cmap.get() == 1)

    def bulk_process(self):
        if not self.file_paths:
            messagebox.showinfo("Info", "Load .sm4 files first.")
            return

        save_root = filedialog.askdirectory(parent=self, title="Choose a folder to save outputs")
        if not save_root:
            self.log_line("Bulk process canceled (no folder chosen).")
            return

        os.makedirs(save_root, exist_ok=True)
        self.run_batch(save_root=save_root)

    def _load_dataset(self, path):
        if path not in self.dataset_cache:
            self.dataset_cache[path] = SP.load(path)
        return self.dataset_cache[path]

    def _get_selected_channel(self, ds):
        label = self.channel_var.get()
        key = dict(self.channel_map).get(label)
        if key not in ds.data_vars:
            key = list(ds.data_vars.keys())[0]
        return key

    def _channel_labels(self):
        return [label for label, _ in self.channel_map]

    def _get_current_file_index(self):
        if self.current_file in self.file_paths:
            return self.file_paths.index(self.current_file)
        return -1

    def _get_current_channel_index(self):
        labels = self._channel_labels()
        current = self.channel_var.get()
        if current in labels:
            return labels.index(current)
        return 0

    def _format_file_label(self, path, index):
        base = os.path.basename(path)
        parent = os.path.basename(os.path.dirname(path)) or os.path.dirname(path) or "."
        return f"{index + 1:03d} | {base} [{parent}]"

    def _refresh_file_menu(self, selected_path=None):
        self.file_display_map = {}
        values = []

        for idx, path in enumerate(self.file_paths):
            label = self._format_file_label(path, idx)
            self.file_display_map[label] = path
            values.append(label)

        self.file_menu["values"] = values

        if not self.file_paths:
            self.file_var.set("")
            return

        if selected_path is None:
            selected_path = self.current_file if self.current_file in self.file_paths else self.file_paths[0]

        for label, path in self.file_display_map.items():
            if path == selected_path:
                self.file_var.set(label)
                return

        self.file_var.set(values[0])

    def _set_current_file(self, path, apply_preview=True):
        if path not in self.file_paths:
            return

        self.current_file = path
        self.current_ds = self._load_dataset(path)
        self._refresh_file_menu(selected_path=path)
        self.path_var.set(path)
        self.file_name_var.set(os.path.basename(path))
        self._update_navigation_state()

        if apply_preview:
            self.apply_preview()

    def _clear_current_file(self):
        self.current_file = None
        self.current_ds = None
        self.file_var.set("")
        self.file_menu["values"] = []
        self.path_var.set("No file loaded.")
        self.file_name_var.set("—")
        self.preview_title_var.set("No image loaded")
        self._clear_preview()
        self._update_navigation_state()

    def _clear_preview(self):
        self.ax.clear()
        self.ax.axis("off")
        self.canvas.draw()

    def _update_navigation_state(self):
        file_count = len(self.file_paths)
        file_idx = self._get_current_file_index()
        channel_labels = self._channel_labels()
        channel_count = len(channel_labels)
        channel_idx = self._get_current_channel_index()
        current_channel = self.channel_var.get() or self.channel_map[0][0]
        self.channel_name_var.set(current_channel)

        if file_count > 0 and file_idx >= 0:
            self.file_position_var.set(f"File {file_idx + 1} / {file_count}")
        else:
            self.file_position_var.set("File 0 / 0")

        self.channel_position_var.set(f"Channel {channel_idx + 1} / {channel_count}")

        file_loaded = file_count > 0 and file_idx >= 0
        self.prev_file_btn.configure(state=("normal" if file_loaded and file_idx > 0 else "disabled"))
        self.next_file_btn.configure(state=("normal" if file_loaded and file_idx < file_count - 1 else "disabled"))
        self.remove_file_btn.configure(state=("normal" if file_loaded else "disabled"))
        self.delete_file_btn.configure(state=("normal" if file_loaded else "disabled"))
        self.open_folder_btn.configure(state=("normal" if file_loaded else "disabled"))
        self.prev_channel_btn.configure(state=("normal" if channel_idx > 0 else "disabled"))
        self.next_channel_btn.configure(state=("normal" if channel_idx < channel_count - 1 else "disabled"))
        self.apply_btn.configure(state=("normal" if file_loaded else "disabled"))

    def _current_config_dict(self):
        var_map = self._variable_map()
        config = {}
        for key, _value_type in self.CONFIG_FIELDS:
            config[key] = var_map[key].get()
        return config

    def save_config(self):
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save configuration",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return

        config = self._current_config_dict()
        with open(path, "w", encoding="utf-8") as f:
            f.write("# SM4 Batch Processor configuration\n")
            f.write("# key=value\n")
            for key, value_type in self.CONFIG_FIELDS:
                value = config[key]
                if value_type == "str" and key == "destripe_sign" and not value:
                    value = "both"
                f.write(f"{key}={value}\n")

        self.log_line(f"Saved config: {path}")

    def load_config(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Load configuration",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return

        var_map = self._variable_map()
        schema = {key: value_type for key, value_type in self.CONFIG_FIELDS}
        loaded = 0

        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    if key not in schema or key not in var_map:
                        continue

                    parsed = self._parse_config_value(schema[key], value)
                    if key == "destripe_sign" and parsed == "":
                        parsed = "both"
                    var_map[key].set(parsed)
                    loaded += 1
        except Exception as e:
            messagebox.showerror("Config error", f"Failed to load config:\n{e}")
            return

        if not self.destripe_sign.get():
            self.destripe_sign.set("both")
        if not self.scalebar_position.get():
            self.scalebar_position.set("lower right")

        self.log_line(f"Loaded config: {path} ({loaded} fields)")
        self.apply_preview()
        self._update_navigation_state()

    @staticmethod
    def _parse_config_value(value_type, text_value):
        if value_type == "int":
            lowered = text_value.strip().lower()
            if lowered in ("true", "yes", "on"):
                return 1
            if lowered in ("false", "no", "off"):
                return 0
            return int(float(text_value))
        if value_type == "float":
            return float(text_value)
        return text_value

    def load_files(self):
        try:
            ctypes.windll.user32.SetForegroundWindow(ctypes.windll.kernel32.GetConsoleWindow())
        except Exception:
            pass

        paths = filedialog.askopenfilenames(
            parent=self,
            title="Select one or more SM4 files",
            filetypes=[("SM4 files", "*.sm4"), ("All files", "*.*")],
        )
        if not paths:
            return

        self.file_paths = list(paths)
        self.current_file = self.file_paths[0]
        self.current_ds = self._load_dataset(self.current_file)
        self._refresh_file_menu(selected_path=self.current_file)
        self.path_var.set(self.current_file)
        self.file_name_var.set(os.path.basename(self.current_file))
        self.log_line(f"Loaded {len(self.file_paths)} file(s). Showing: {os.path.basename(self.current_file)}")
        self._update_navigation_state()
        self.apply_preview()

    def change_file(self):
        selection = self.file_var.get()
        path = self.file_display_map.get(selection)
        if not path:
            return
        self._set_current_file(path, apply_preview=True)

    def on_channel_selected(self):
        self._update_navigation_state()
        self.apply_preview()

    def prev_file(self):
        idx = self._get_current_file_index()
        if idx > 0:
            self._set_current_file(self.file_paths[idx - 1], apply_preview=True)

    def next_file(self):
        idx = self._get_current_file_index()
        if 0 <= idx < len(self.file_paths) - 1:
            self._set_current_file(self.file_paths[idx + 1], apply_preview=True)

    def prev_channel(self):
        idx = self._get_current_channel_index()
        if idx > 0:
            self.channel_var.set(self._channel_labels()[idx - 1])
            self._update_navigation_state()
            self.apply_preview()

    def next_channel(self):
        idx = self._get_current_channel_index()
        labels = self._channel_labels()
        if idx < len(labels) - 1:
            self.channel_var.set(labels[idx + 1])
            self._update_navigation_state()
            self.apply_preview()

    def _remove_path_from_batch(self, path):
        if path not in self.file_paths:
            return False

        idx = self.file_paths.index(path)
        self.file_paths.pop(idx)
        self.dataset_cache.pop(path, None)

        if not self.file_paths:
            self._clear_current_file()
            return True

        new_idx = min(idx, len(self.file_paths) - 1)
        self._set_current_file(self.file_paths[new_idx], apply_preview=True)
        return True

    def remove_current_from_batch(self):
        if not self.current_file:
            return

        removed_name = os.path.basename(self.current_file)
        if self._remove_path_from_batch(self.current_file):
            self.log_line(f"Removed from batch: {removed_name}")

    def delete_current_file(self):
        if not self.current_file:
            return

        path = self.current_file
        filename = os.path.basename(path)
        confirmed = messagebox.askyesno(
            "Delete file",
            f"Delete this SM4 file from disk?\n\n{path}\n\nThis cannot be undone by the app.",
            icon="warning",
        )
        if not confirmed:
            return

        try:
            os.remove(path)
        except Exception as e:
            messagebox.showerror("Delete failed", f"Could not delete file:\n{e}")
            return

        self.log_line(f"Deleted file: {filename}")
        self._remove_path_from_batch(path)

    def open_containing_folder(self):
        if not self.current_file:
            return

        folder = os.path.dirname(self.current_file)
        try:
            open_folder_in_os(folder)
            self.log_line(f"Opened folder: {folder}")
        except Exception as e:
            messagebox.showerror("Open folder failed", f"Could not open folder:\n{e}")

    def apply_preview(self):
        if self.current_ds is None:
            self._clear_preview()
            return

        settings = self.current_settings()
        key = self._get_selected_channel(self.current_ds)
        ch = self.current_ds[key].copy(deep=True)
        plot_meta = get_plot_metadata(ch)

        try:
            img = process_channel_for_preview(ch, settings)
        except Exception as e:
            self.log_line(f"Preview failed: {e}")
            img = np.array(ch.data, copy=True)
            img = img - np.nanmin(img)

        vmin, vmax = safe_percentiles(img, settings["contrast_low_pct"], settings["contrast_high_pct"])
        self.ax.clear()
        self.ax.imshow(
            img,
            cmap=self.current_colormap(),
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            extent=plot_meta["extent"],
            aspect="equal",
        )
        self.ax.axis("off")

        if settings["show_scale_bar"]:
            if not add_scale_bar(self.ax, plot_meta, settings["scalebar_position"]):
                self.log_line("Scale bar skipped in preview (missing physical scale metadata)")

        self.canvas.draw()

        current_channel = self.channel_var.get() or "Preview"
        self.preview_title_var.set(current_channel)
        self._update_navigation_state()
        self.status_bar_var.set(f"Preview updated: {current_channel}")

    def run_batch(self, save_root=None):
        if not self.file_paths:
            messagebox.showinfo("Info", "Load .sm4 files first.")
            return

        settings = self.current_settings()
        cmap = self.current_colormap()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        if save_root is None or not str(save_root).strip():
            save_root = os.getcwd()
        os.makedirs(save_root, exist_ok=True)

        composite_parent = os.path.join(save_root, f"{timestamp} Topography")
        os.makedirs(composite_parent, exist_ok=True)

        for file_path in self.file_paths:
            self.log_line(f"Processing: {file_path}")
            ds = self._load_dataset(file_path)
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            output_folder = os.path.join(save_root, base_name)
            os.makedirs(output_folder, exist_ok=True)

            channels = {
                "lia_forward": ds.get("LIA_Current_Forward", None),
                "lia_backward": ds.get("LIA_Current_Backward", None),
                "current_forward": ds.get("Current_Forward", None),
                "current_backward": ds.get("Current_Backward", None),
                "topography_forward": ds.get("Topography_Forward", None),
                "topography_backward": ds.get("Topography_Backward", None),
            }

            for stem, ch in channels.items():
                if ch is None:
                    self.log_line(f"  Skipping missing channel: {stem}")
                    continue
                title_name, file_stem = build_export_names(stem, base_name)
                process_and_save_channel(ch, title_name, file_stem, output_folder, settings, cmap, self.log_line)

            _fwd_title, fwd_stem = build_export_names("topography_forward", base_name)
            _bwd_title, bwd_stem = build_export_names("topography_backward", base_name)
            fwd_png = os.path.join(output_folder, f"{fwd_stem}.png")
            bwd_png = os.path.join(output_folder, f"{bwd_stem}.png")
            if os.path.exists(fwd_png) and os.path.exists(bwd_png):
                out_path = os.path.join(composite_parent, f"{base_name}_topography.png")
                save_stitched_side_by_side(fwd_png, bwd_png, out_path, self.log_line)

            try:
                with open(os.path.join(output_folder, "metadata.txt"), "w", encoding="utf-8") as f:
                    for channel_name in ds.data_vars:
                        f.write(f"[{channel_name}]\n")
                        for key, value in ds[channel_name].attrs.items():
                            f.write(f"{key}: {value}\n")
                        f.write("\n")
                self.log_line("  Wrote metadata.")
            except Exception as e:
                self.log_line(f"  Metadata write failed: {e}")

        self.log_line("Batch processing complete.")

if __name__ == "__main__":
    app = App()
    app.mainloop()
