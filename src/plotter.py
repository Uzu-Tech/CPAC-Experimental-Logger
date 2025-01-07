from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


def create_plot(x: int, y: int) -> Tuple[plt.Figure, plt.Axes]:
    """Create a plot with specified dimensions."""
    return plt.subplots(figsize=(x, y))


def origin_line(x, a):
    return a * x


def add_best_fit_line(
    graph: plt.Axes,
    x: Iterable,
    y: Iterable,
    line_style="solid",
    color: str = "#ffffff",
    line_width: float = 1.5,
    alpha: float = 1,
    is_origin_intercept: bool = False,
    hide_origin: bool = False,
) -> Tuple[float, float]:
    """Add a best-fit line to the plot."""
    x, y = np.array(x), np.array(y)
    x_min = x.min() if hide_origin else 0
    x_seq = np.linspace(x_min, x.max())

    if not is_origin_intercept:
        x = np.array(x).reshape((-1, 1))
        best_fit_line = LinearRegression()
        best_fit_line.fit(x, y)

        gradient = best_fit_line.coef_[0]
        intercept = best_fit_line.intercept_
        line_eq = intercept + gradient * x_seq
    else:
        gradient, _ = curve_fit(origin_line, x, y)
        intercept = 0
        line_eq = gradient * x_seq

    graph.plot(
        x_seq,
        line_eq,
        color=color,
        lw=line_width,
        ls=line_style,
        alpha=alpha,
        zorder=1,
    )

    return gradient, intercept


def plot_scatter_graph(
    graph: plt.Axes,
    x: Iterable,
    y: Iterable,
    title: Tuple[str, float],
    axis_labels: Tuple[str, str] = ("", ""),
    size: float = 25,
    marker_color: str = "#ffffff",
    line_color: str = "#ffffff",
    line_width: float = 1.5,
    is_origin_intercept: bool = False,
    hide_origin: bool = False,
) -> Tuple:
    """Plot a scatter graph with a title and best-fit line."""
    x, y = np.array(x), np.array(y)
    gradient, intercept = add_best_fit_line(
        graph,
        x,
        y,
        color=line_color,
        line_width=line_width,
        is_origin_intercept=is_origin_intercept,
        hide_origin=hide_origin,
    )
    graph.scatter(x, y, s=size, marker="x", zorder=3, c=marker_color)
    graph.set_title(title[0], fontsize=title[1])
    graph.set_xlabel(axis_labels[0])
    graph.set_ylabel(axis_labels[1])
    return gradient, intercept


def add_error_bars(
    graph: plt.Axes,
    x: Iterable,
    y: Iterable,
    errors_x: Iterable,
    errors_y: Iterable,
    color: str = "#ffffff",
    capsize: float = 2,
    alpha: float = 1,
) -> None:
    """Add error bars to the plot."""
    x, y = np.array(x), np.array(y)
    graph.errorbar(
        x,
        y,
        xerr=errors_x,
        yerr=errors_y,
        ls="None",
        capsize=capsize,
        zorder=2,
        ecolor=color,
        alpha=alpha,
    )


def plot_text(
    graph: plt.Axes, text: str, pos: Tuple[float, float], font_size: float, color: str
):
    graph.text(
        x=pos[0],
        y=pos[1],
        s=text,
        fontsize=font_size,
        c=color,
        horizontalalignment="center",
        verticalalignment="center",
        transform=graph.transAxes,
    )
