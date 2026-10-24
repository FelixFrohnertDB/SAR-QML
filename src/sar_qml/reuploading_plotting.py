"""Plotting helpers for the combined query-encoding / query-value figure.

Used by ``fig_01_02_combined.ipynb``, which reads the curves written by
``fig_01_vary_encoding.ipynb`` and ``fig_02_vary_query.ipynb``. The module
holds the figure style (rcParams, one colour, dash pattern and label per
series), the loader for the saved curves and the functions that draw one row
of three panels.
"""

import numpy as np

#: Series in drawing order; the same order is used in the legend.
SERIES = ("local", "local_narrow", "nonlocal")
QUANTITIES = (
    "shape_change",
    "nonseparable_weight",
    "prior_distance",
    "prior_distance_error",
)

COLOR = {
    "local": "#0072B2",
    "local_narrow": "#009E73",
    "nonlocal": "#D55E00",
}
STYLE = {
    "local": (0, (4, 2)),
    "local_narrow": (0, (5, 1.5, 1, 1.5)),
    "nonlocal": "-",
}
BAND_ALPHA = {"local": 0.12, "local_narrow": 0.13, "nonlocal": 0.14}
LEGEND_ORDER = ("local", "local_narrow", "nonlocal")
LABEL = {
    "local": r"(I) Absorbable, Haar",
    "local_narrow": r"(II) Absorbable, narrow Gaussian",
    "nonlocal": r"(III) Nonlocal, Haar",
}


def figure_rc_params(use_tex):
    """Matplotlib rcParams of the figure: serif text at 8 pt, LaTeX if ``use_tex``."""
    return {
        "text.usetex": use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 6.9,
        "legend.handlelength": 2.1,
        "legend.columnspacing": 1.2,
        "lines.linewidth": 1.3,
        "savefig.bbox": None,
    }


def load_sweep_curves(path, abscissa_key):
    """
    Read one saved sweep into ``{quantity: {series: array}}``.

    The result also holds the abscissa, the Haar noise ceiling, the block
    type and, under ``"extra"``, the fixed query vector or the query
    direction, whichever the file contains.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the corresponding notebook once to write it."
        )
    with np.load(path) as data:
        results = {
            quantity: {series: data[f"{quantity}__{series}"] for series in SERIES}
            for quantity in QUANTITIES
        }
        results["abscissa"] = data[abscissa_key]
        results["noise_ceiling"] = float(data["noise_ceiling"])
        results["trainable_blocks"] = str(data["trainable_blocks"])
        results["extra"] = {
            key: data[key]
            for key in ("fixed_query_vector", "query_direction")
            if key in data
        }
    return results


def plot_series(axis, abscissa, values, errors=None):
    """
    Draw every series, with an optional 95 percent band clipped at zero.

    Returns the upper end of the plotted data (including the bands), so
    that axis limits can follow it.
    """
    upper = -np.inf
    for series in SERIES:
        axis.plot(abscissa, values[series], color=COLOR[series], ls=STYLE[series])
        if errors is None:
            upper = max(upper, np.max(values[series]))
            continue

        half_width = 1.96 * errors[series]
        axis.fill_between(
            abscissa,
            np.maximum(0.0, values[series] - half_width),
            values[series] + half_width,
            color=COLOR[series],
            alpha=BAND_ALPHA[series],
            linewidth=0,
        )
        upper = max(upper, np.max(values[series] + half_width))
    return upper


def query_vector_label(vector):
    """A LaTeX label f=(...) for the fixed query, recognising simple fractions of pi."""
    named = {
        np.pi / 2.0: r"\pi/2",
        np.pi / 3.0: r"\pi/3",
        np.pi / 4.0: r"\pi/4",
        np.pi / 6.0: r"\pi/6",
    }
    components = []
    for value in vector:
        for reference, text in named.items():
            if np.isclose(value, reference):
                components.append(text)
                break
        else:
            components.append(f"{value:.3f}")
    return r"$f=(" + ",".join(components) + r")$"


def plot_row(axes, results, variable, configure_axis, labels, descriptions=None):
    """
    Fill the three panels of one sweep: shape change, class witness, prior distance.

    ``variable`` is the LaTeX name of the abscissa, ``configure_axis`` sets
    its limits, ticks and label, ``labels`` are the panel letters and
    ``descriptions`` optional right-hand panel titles. The prior-distance
    axis is linear inside the Haar null and logarithmic above it. Returns
    the upper end of the class-witness data.
    """
    abscissa = results["abscissa"]
    noise_ceiling = results["noise_ceiling"]
    x_mid = 0.5 * (abscissa[0] + abscissa[-1])

    # naive test: coefficient direction at one fixed parameter vector
    ax = axes[0]
    plot_series(ax, abscissa, results["shape_change"])
    configure_axis(ax)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel(rf"Shape change $\Delta({variable})$")
    ax.set_title(labels[0], loc="left", pad=4, fontweight="bold")

    # class test: existence of a nonseparable function
    ax = axes[1]
    upper_weight = plot_series(ax, abscissa, results["nonseparable_weight"])
    configure_axis(ax)
    ax.set_ylim(-0.045 * upper_weight, 1.12 * upper_weight)
    ax.axhline(0.0, color="0.55", linewidth=0.5, zorder=0)
    ax.set_ylabel(rf"Class witness $S({variable})$")
    ax.set_title(labels[1], loc="left", pad=4, fontweight="bold")
    if results["trainable_blocks"] == "product":
        ax.annotate(
            rf"(I), (II): $S({variable})=0$ exactly",
            xy=(x_mid, 0.0),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.6,
            color="0.25",
        )

    # prior test: change in the parameter-induced spectral second moment
    ax = axes[2]
    upper_distance = plot_series(
        ax,
        abscissa,
        results["prior_distance"],
        results["prior_distance_error"],
    )
    configure_axis(ax)
    ax.axhspan(0.0, noise_ceiling, color="0.80", alpha=0.55, linewidth=0, zorder=0)
    # linear inside the sampling null, logarithmic above it
    ax.set_yscale("symlog", linthresh=noise_ceiling, linscale=0.35)
    ax.set_ylim(0.0, max(1.0, 1.6 * upper_distance))
    ax.annotate(
        "Haar null",
        xy=(abscissa[0] + 0.05 * (abscissa[-1] - abscissa[0]), noise_ceiling),
        xytext=(0, -1),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=6.6,
        color="0.25",
    )
    ax.set_ylabel(rf"Spectral-prior distance $D_{{\Sigma}}({variable})$")
    ax.set_title(labels[2], loc="left", pad=4, fontweight="bold")
    if descriptions is not None:
        for ax, description in zip(axes, descriptions):
            ax.set_title(description, loc="right", pad=4, fontsize=7.0, color="0.25")
    return upper_weight


def configure_encoding_axis(axis):
    """Abscissa of the encoding sweep: alpha in [0, pi/2]."""
    axis.set_xlim(0.0, np.pi / 2.0)
    axis.set_xticks([0.0, np.pi / 4.0, np.pi / 2.0])
    axis.set_xticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$"])
    axis.set_xlabel(r"Encoding interpolation $\alpha$")


def configure_query_axis(axis):
    """Abscissa of the query sweep: t in [0, 2 pi]."""
    axis.set_xlim(0.0, 2.0 * np.pi)
    axis.set_xticks([0.0, np.pi, 2.0 * np.pi])
    axis.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    axis.set_xlabel(r"Query path parameter $t$")
