"""Plotting helpers shared by the figure notebooks."""

import shutil
import subprocess

import matplotlib.ticker as mticker


def latex_is_usable():
    """True if this installation can serve matplotlib's usetex."""
    if shutil.which("latex") is None or shutil.which("dvipng") is None:
        return False
    try:
        completed = subprocess.run(
            ["kpsewhich", "latex.ltx"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and bool(completed.stdout.strip())


def panel_title(axis, letter, description):
    """Bold panel letter on the left, description on the right."""
    axis.set_title(letter, loc="left", pad=4, fontweight="bold")
    axis.set_title(description, loc="right", pad=4, fontsize=7.0, color="0.25")


def finish_axis(axis, x_minor=True, y_minor=True, grid_axis="both"):
    """Gridlines, minor ticks and spines, applied identically to every panel."""
    axis.set_axisbelow(True)
    axis.minorticks_on()
    if not x_minor:
        axis.xaxis.set_minor_locator(mticker.NullLocator())
    if not y_minor:
        axis.yaxis.set_minor_locator(mticker.NullLocator())
    axis.tick_params(which="both", top=False, right=False)
    axis.grid(True, which="major", axis=grid_axis, linestyle="-", alpha=0.30)
    axis.grid(True, which="minor", axis=grid_axis, linestyle=":", alpha=0.15)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
