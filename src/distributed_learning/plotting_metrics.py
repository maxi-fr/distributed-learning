from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from matplotlib.figure import Figure


def plot_metric(metric: pd.Series, ax: plt.Axes, label: str) -> None:
    """Plot one metric series and label its axis entry.

    Parameters
    ----------
    metric : pandas.Series
        Metric values indexed by training step or epoch.
    ax : matplotlib.axes.Axes
        Axis on which to draw the series.
    label : str
        Legend label for the series.
    """
    ax.plot(range(len(metric)), metric, label=label)
    ax.set_xlabel("Epochs")
    ax.grid(visible=True)
    ax.legend()


def plot_metrics(
    df: pd.DataFrame,
    fname: str | Path | None = None,
) -> tuple[Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot loss and accuracy columns in separate panels.

    Parameters
    ----------
    df : pandas.DataFrame
        Metrics with column names identifying loss or accuracy values.
    fname : str or pathlib.Path, optional
        Destination path for saving the figure.

    Returns
    -------
    tuple
        Figure and the loss and accuracy axes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for col in df.columns:
        if "loss" in col:
            plot_metric(df[col], ax1, col)
            ax1.set_ylabel("loss")
        else:
            plot_metric(df[col], ax2, col)
            ax2.set_ylabel("accuracy")

    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname)
    return fig, (ax1, ax2)
