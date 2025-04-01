# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: NK-Dev2
#     language: python
#     name: nk-dev
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

from utils import generate_all_pauli_matrices, load_psi, comm
from qite_functions import (
    get_largest_bistrings,
    gen_cost_fun_fidelity_per_string,
)

from plotting import transform_axis, plot_sorted_psi, plot_sampling_process

# %%
N = 7
pauli_matrices = generate_all_pauli_matrices(N)

# %%
random_psi = True
psi_file = "fe3_ground_state.npy"
if not random_psi:
    psi_0 = load_psi(psi_file, N)
else:
    psi_0 = np.exp(5 * (np.random.randn(2**N)))
    psi_0 /= np.linalg.norm(psi_0)

# %%
# ## Plot |psi|^2
plot_sorted_psi(psi_0, transform_axis=lambda i: transform_axis(i, N))

# %%
removed = []
n_runs = 1
for run in range(n_runs):
    # noise_paulis = lambda ln: 1 + 0.1 * (np.random.rand(ln) - 0.5)
    data = get_largest_bistrings(
        psi_0,
        M=10,
        print_level=1,
        pauli_matrices=pauli_matrices,
        tau=10,
        n_samples_psi=100,
        L=32,  # psi_0: works with 1, psi_k: L > 1
        n_samples=0,
        noise=None,
        gen_cost_fun=gen_cost_fun_fidelity_per_string,
        mode="psi_k",  # psi_0 | psi_k
    )

    M = len(data["removed"])
    nw = np.sum(np.array(data["rank"])>M)
    print(f"Removed out of first {M}: {nw}")

    removed.append(data["removed"])

# %%
removed = np.array(removed)
common_strings = reduce(lambda x,y: np.intersect1d(x,y),removed[1:], removed[0])
all_strings = reduce(lambda x,y: np.union1d(x,y),removed[1:], removed[0])
inds_sort = np.argsort(np.abs(psi_0))[::-1]
order_common = np.sort(np.array([np.argmax(inds_sort==c) for c in common_strings]))
order_all = np.sort(np.array([np.argmax(inds_sort==c) for c in all_strings]))

# %%
times_counted = np.array([np.sum(removed == inds_sort[i]) for i in range(40)])

fig, ax = plt.subplots()
ax.set_title("Number of counts per bit string")
ax.set_xlabel("Rank in magnitude")

ax.plot(times_counted, "o:")

xlim = np.where(times_counted != 0)[0][-1]
ax.set_xlim(0, xlim+2)

from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# %%
fig, ax = plt.subplots()

ax.set_title("Magnitude squared of the removed bitstring")
ax.set_xlabel("Step")

sorted_psi_sq = np.sort(np.abs(psi_0)**2)[::-1]
mag_removed = np.abs(psi_0[data["removed"]]) ** 2
ax.semilogy(mag_removed, "k", zorder=-1)

colors = []
for i in range(len(mag_removed)):
    if np.any(mag_removed[i] < sorted_psi_sq[i:]):
        colors.append("r")
    else:
        colors.append("b")

ax.scatter(np.arange(len(mag_removed)), mag_removed,  c=colors)


from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# %%
plot_sampling_process(psi_0, data, filename="am", save_separate=False)
