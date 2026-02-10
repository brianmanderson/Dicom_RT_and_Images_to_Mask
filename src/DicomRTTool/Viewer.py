"""Simple scrollable image viewer built on matplotlib.

Provides a convenience function for interactively browsing 3-D image
stacks in a Jupyter or interactive matplotlib session.
"""
from __future__ import annotations

import copy
from typing import Tuple

import numpy as np


def plot_scroll_Image(x: np.ndarray) -> Tuple:
    """Display a scrollable 3-D image stack using matplotlib.

    Args:
        x: Input array.  Accepted shapes are ``(slices, rows, cols)`` or
           ``(rows, cols, slices)``.  The function will attempt to
           transpose so slices are on axis-2 for display.

    Returns:
        ``(fig, tracker)`` – the matplotlib figure and the
        :class:`IndexTracker` instance so the caller can keep a reference
        (required for the scroll callback to remain active).
    """
    import matplotlib.pyplot as plt  # Late import – optional dependency

    if x.dtype not in (np.float32, np.float64):
        x = copy.deepcopy(x).astype(np.float32)

    x = np.squeeze(x)
    if x.ndim == 2:
        x = np.expand_dims(x, axis=-1)
    elif x.ndim == 3:
        # Heuristic: put the "slices" dimension last for display
        if x.shape[0] != x.shape[1]:
            x = np.transpose(x, (1, 2, 0))
        elif x.shape[0] == x.shape[2]:
            x = np.transpose(x, (1, 2, 0))

    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, x)
    fig.canvas.mpl_connect("scroll_event", tracker.onscroll)
    return fig, tracker


class IndexTracker:
    """Matplotlib callback helper for scrolling through image slices."""

    def __init__(self, ax, X: np.ndarray) -> None:
        self.ax = ax
        ax.set_title("Use scroll wheel to navigate images")
        self.X = X
        _rows, _cols, self.slices = X.shape

        # Start at a slice that has some contrast
        nonuniform = np.where(
            np.min(self.X, axis=(0, 1)) != np.max(self.X, axis=(0, 1))
        )[-1]
        if len(nonuniform) > 0:
            self.ind = int(nonuniform[len(nonuniform) // 2])
        else:
            self.ind = self.slices // 2

        self.im = ax.imshow(self.X[:, :, self.ind], cmap="gray")
        self.update()

    def onscroll(self, event) -> None:  # noqa: ANN001
        if event.button == "up":
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self) -> None:
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel("slice %s" % self.ind)
        self.im.axes.figure.canvas.draw()
