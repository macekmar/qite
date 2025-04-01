import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


def transform_axis(inds, N, steps=[[20, 0.5], [100, 0.9]]):
    "Stretches and shrinks parts of the axis."
    # 0-20 values maps into first 50% of the axis
    # 20-100 maps into 50-90 % of the axis
    lim1, wid1 = steps[0]
    lim2, wid2 = steps[1]
    inds = np.array(inds).astype(np.float64)
    transformed = np.piecewise(
        inds,
        [inds <= lim1, inds > lim1, inds > lim2],
        [
            lambda x: x * wid1 / lim1,
            lambda x: wid1 + (x - lim1) * (wid2 - wid1) / (lim2 - lim1),
            lambda x: wid2 + (x - lim2) * (1 - wid2) / (2**N - lim2),
        ],
    )
    return transformed


def plot_sorted_psi(
    psi, filename: str = None, transform_axis: Callable[[int | float], float] = None
):
    """Plot |ψ|² sorted by magnitude."""
    if transform_axis is None:
        transform_axis = lambda i: i

    fig, ax = plt.subplots()

    abs_psi_sq = np.abs(psi) ** 2
    inds = np.argsort(abs_psi_sq)[::-1]
    abs_psi_sq = abs_psi_sq[inds]
    x_data = transform_axis(np.arange(psi.shape[0], dtype=float))

    ax.semilogy(x_data, abs_psi_sq, ".-", label=r"$|\psi|^2$")
    ax.semilogy(
        x_data[1:],
        (abs_psi_sq[:-1] - abs_psi_sq[1:]) / abs_psi_sq[:-1],
        ".-",
        label=r"$1-|\psi_{i+1}|^2/|\psi_i|^2$",
    )

    ax.set_xticks(np.concatenate((x_data[:22][::2], x_data[30:100:10])))
    ax.set_xticklabels(np.concatenate((np.arange(0, 22, 2), np.arange(30, 100, 10))))
    ax.grid(True)

    ax.set_ylim(1e-10, None)
    ax.legend()

    if filename is not None:
        fig.savefig(filename, dpi=300)

    return fig, ax


def plot_sampling_process(
    psi_0, data, filename: str, save_separate=True
):
    N = int(np.log2(psi_0.shape[0]))

    fig, ax = plt.subplots()

    inds = np.argsort(np.abs(psi_0))[::-1]
    transformed = transform_axis(np.arange(2**N), N)
    ax.semilogy(transformed, np.abs(psi_0)[inds], ".-", label=r"$|\psi|$")

    itr = 0
    (all_dots,) = ax.semilogy(
        transformed,
        np.abs(data["exp_A_psi"][itr])[inds],
        ".",
        label=r"$|e^{-iA_kt}\psi\rangle|$",
    )
    (cross,) = ax.semilogy(
        [],
        [],
        "kx",
        ms=10,
        zorder=-20,
        label="Should be removed",
    )
    (circle,) = ax.semilogy(
        [],
        [],
        "ko",
        ms=10,
        markerfacecolor=None,
        fillstyle="none",
        markeredgecolor="k",
        zorder=-20,
        label="Actually sampled",
    )
    (gdot,) = ax.semilogy([], [], "go", zorder=-10, label="Removed")

    ax.legend(loc="lower left")
    ax.set_ylim(1e-4, 1.1)
    ax.set_xticks(np.concatenate((transformed[:22][::2], transformed[30:100:10])))
    ax.set_xticklabels(np.concatenate((np.arange(0, 22, 2), np.arange(30, 100, 10))))
    ax.grid(True)

    def update(i):
        if i == 0:
            psi_k = np.abs(psi_0)
        else:
            psi_k = np.abs(data["exp_A_psi"][i - 1])

        all_dots.set_ydata(psi_k[inds])

        cross.set_xdata([transform_axis(i, N)])
        cross.set_ydata([psi_k[inds][i]])

        circle.set_xdata([transform_axis(data["rank"][i], N)])
        circle.set_ydata([psi_k[data["removed"][i]]])

        t = transform_axis(np.array(data["rank"][:i]), N)
        gdot.set_xdata(t)
        gdot.set_ydata(np.abs(data["exp_A_psi"][i - 1][data["removed"][:i]]))

        return all_dots, cross, gdot

    if save_separate:
        for i in np.arange(0, len(data["rank"])):
            update(i)
            fig.savefig(filename + f"{i:02d}.png", dpi=300)
    else:
        from matplotlib.animation import FuncAnimation, PillowWriter
        ani = FuncAnimation(fig, update, frames=np.arange(0, len(data["rank"])), interval=1500, blit=True)
        ani.save(filename + ".gif", writer=PillowWriter(fps=1), dpi=300)
