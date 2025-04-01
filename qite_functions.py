from typing import Callable, List, Tuple
from collections import defaultdict

import numpy as np
import scipy as sc

from utils import fidelity, generate_all_pauli_matrices, sample_pdf


def calc_O_b(psi, phi, pauli_matrices):
    """Calculates the O_αi = σ^α_{ij}ψ_j matrix and b_α = ⟨ψ|σ_α·H|ψ⟩ vector."""
    O = np.array([p @ psi for p in pauli_matrices]).T  # noqa:E741
    b = 2 * (O.conj().T @ phi).imag
    return O, b


def solve_system(O, b, return_S=False):  # noqa:E741
    """Solves linear system O⁺O x = -O⁺f = -b.

    If O is an isometry OO⁺=I, then x = -b."""
    S = 2 * (O.conj().T @ O).real
    invS = np.linalg.pinv(S, rcond=1e-10)
    sol = invS @ (-b)
    if return_S:
        return sol, S, invS
    else:
        return sol


def generate_projector(strings, size, tau=5):
    """Generates projector removing the supplied strings.

    Operator H projects onto bit strings.
    We do imaginary time evolution:
        P = exp(-Hτ) = 1-H(1-exp(-τ))
    which for τ→∞ reduces to P = 1-H, thus removing the bit strings.
    """
    diag = np.zeros(size)
    diag[strings] = 1
    H = np.diag(diag)
    P = np.diag(np.exp(-diag * tau))
    return H, P


def gen_cost_fun_fidelity_per_string(expAt, psi, target_psi):
    """Returns a cost function ∑_s F(e^-iAt |ψ⟩,|s⟩)."""

    def fid_sep(t):
        inds = np.where(target_psi != 0)[0]
        tot = 0
        for i in inds:
            phi = np.zeros(psi.shape)
            phi[i] = 1
            tot += fidelity(phi, expAt(t) @ psi).real
        return tot

    return fid_sep


def gen_cost_fun_fidelity(expAt, psi, target_psi):
    """Returns a cost function F(e^-iAt |ψ⟩,|ϕ⟩)."""
    return lambda t: fidelity(target_psi, expAt(t) @ psi).real


def sample_next_bitstring(psi, removed_bitstrings, n_samples):
    """Samples |ψ|² until we get a bitstring not in removed_bitstrings."""
    pdf = np.abs(psi) ** 2
    cond = True
    stats = defaultdict(lambda: 0)
    while cond:
        new_stats = sample_pdf(pdf, n_samples)
        for key, val in new_stats.items():
            stats[key] += val

        max_count = 0
        for bs, counts in stats.items():
            if bs in removed_bitstrings:
                continue

            cond = False
            if counts > max_count:
                bitstring_to_remove = int(bs)
                max_count = counts

        if cond is True:
            print("Taking additional samples of psi.")

    return bitstring_to_remove, stats


def get_unitary_operator(
    target_psi: np.ndarray,
    psi: np.ndarray,
    coeff: np.ndarray,
    pauli_matrices: List[sc.sparse.sparray],
    L: int = 10,
    n_samples: int = 0,
    noise: Callable = None,
    gen_cost_fun: Callable[
        [Callable, np.ndarray, np.ndarray], Callable[[float], float]
    ] = None,
) -> Tuple[np.ndarray, sc.optimize.OptimizeResult]:
    """Gets the optimal unitary operator.

    We want to find the optimal unitary operator
        exp_A(t) = exp(-i·A·t)
    parametrized with time t and A is defined by coefficients
        A = ∑_α coeff_α Pauli_α.

    The optimality is defined by the cost functions, the default one is
    fidelity between time evolved state and projected state:
        F(exp_A |ψ⟩, P|ψ⟩).
    The user should provide a generator for a cost function taking as
    arguments a callable for exp_A(t), |ψ⟩, and |ψ_target⟩.

    Since coeff is exponentially long, we can define L: the number of
    largest coefficients we take. We can sample the largest coefficients.

    Parameters:
    - P: Projector matrix
    - psi: Wave function
    - coeff: Coefficients in the expansion of the operator into Pauli strings.
    - pauli_matrices: A list of matrices corresponding to the Pauli strings.
    - L: The number of largest coefficients to consider.
    - n_samples_pauli: The number of samples used to determine the largest coefficients.
                 If set to 0, take L largest coefficients
    - noise: If measurement of coefficients is noisy, see code
    - gen_cost_fun: A generator function for the cost function.
    """

    if L is None:
        L = len(coeff)
    if n_samples == 0:
        inds = np.argsort(-np.abs(coeff))
        il = inds[:L]
    else:
        if L > n_samples:
            raise ValueError("Number of samples is smaller than the number of demanded Paulis.")
        # Sample bitstrings (HOW?)
        stats = np.array(list(sample_pdf(np.abs(coeff), n_samples).keys()))
        # Because the distribution is flat, measure the coefficients
        # and find L largest ones
        if noise is not None:
            rel_noise = noise(coeff.shape[0])
        else:
            rel_noise = 1.0
        inds_subset = np.argsort(-np.abs(coeff[stats] * rel_noise))
        il = stats[inds_subset][:L]

    # # This actually kind of works this is because of
    # # optimization below
    # il = np.random.randint(low=0, high=len(coeff), size=L)

    used_coeff = coeff[il]
    used_coeff /= np.linalg.norm(used_coeff)
    A = np.sum(
        [used_coeff[i] * pauli_matrices[j].todense() for i, j in enumerate(il)], axis=0
    )

    expAt = lambda t: sc.linalg.expm(-1j * A * t)  # noqa:E731

    if gen_cost_fun is None:
        gen_cost_fun = gen_cost_fun_fidelity

    opt = sc.optimize.minimize(gen_cost_fun(expAt, psi, target_psi), 0)

    return sc.linalg.expm(-1j * A * opt.x), np.vstack((il, used_coeff)).T, opt


def remove_largest_bitstring(
    psi_0: np.ndarray,
    psi_k: np.ndarray,
    tau: float,
    pauli_matrices: List[sc.sparse.sparray],
    removed: List[int] = [],
    n_samples_psi: int = 100,
    mode: str = "psi_k",
    **kwargs,
):
    """Removes the largest bitstring via a unitary operator.

    Parameters:
    - psi_0: The starting wave function.
    - psi_k: The wave function from the previous iteration.
    - tau: Imaginary time which determines the strength of projection.
    - pauli_matrices: ...
    - removed: List of already sampled/removed string.
    - n_samples_psi: Number of bitstring samples from psi_k.
    - kwargs: arguments for `get_unitary_operator`
    """

    assert mode == "psi_k" or mode == "psi_0"

    # Sample from psi_k and get the next bitstring to remove
    bitstring_to_remove, stats = sample_next_bitstring(psi_k, removed, n_samples_psi)

    # TODO: this could be two functions?
    # I leave it here for experimenting. One could try different combinations of psi_k, psi_0
    if mode == "psi_k":
        # "Solve" QITE linear system
        Hk, Pk = generate_projector([bitstring_to_remove], psi_0.shape[0], tau=tau)
        O, b = calc_O_b(psi_k, Hk @ psi_k, pauli_matrices)  # noqa: E741
        coeff = -b

        # Get the best unitary
        U, used_coeff, sol = get_unitary_operator(
            np.diag(Hk), psi_k, coeff, pauli_matrices, **kwargs
        )
        out_psi = U @ psi_k
    if mode == "psi_0":
        # "Solve" QITE linear system
        H, P = generate_projector(removed + [bitstring_to_remove], psi_0.shape[0], tau=tau)
        O, b = calc_O_b(psi_0, H@psi_0, pauli_matrices)  # noqa: E741
        coeff = -b

        # Get the best unitary
        U, used_coeff, sol = get_unitary_operator(
            np.diag(H), psi_0, coeff, pauli_matrices, **kwargs
        )
        out_psi = U @ psi_0

    output = {
        "exp_A_psi": out_psi,
        "n_samples_psi": np.sum(list(stats.values())),
        "removed": bitstring_to_remove,
        "stats_psi": stats,
        "coeff": coeff,
        "used_coeff": used_coeff,
        "opt": sol,
    }

    return output


def get_largest_bistrings(
    psi_0: np.ndarray,
    M: int,
    print_level: int = 0,
    pauli_matrices: List[sc.sparse.sparray] = None,
    **kwargs,
):
    """Get M-largest bitstrings in psi_0.

    The algorithm should be precise for L=4**N considered Paulis, where
    N is the size of the system, and large tau (about > 5).
    For L = 1, we get usable algorithm but it is not guaranteed that M
    bitstrings are the M largest ones.

    Parameters:
    - psi_0: The wave function to be sampled
    - M: The number of bitstrings to return.
    - print_level: Controls the level of printing messages:
                     0 - no printing,
                     1 - iter number with obtained bitstring
                     2 - print extra/missing bitstrings.
    - pauli_matrices: A list of Pauli matrices that should span the entire
                      non-null space. For a real wave function, only matrices
                      with an odd number of Y can be considered.
    """
    N = int(np.log2(psi_0.shape[0]))

    if pauli_matrices is None:
        pauli_matrices = generate_all_pauli_matrices(N)

    data = {
        "exp_A_psi": [],  # e^-iAt |ψ⟩
        "n_samples_psi": [],
        "removed": [],  # removed bitstrings
        "rank": [],  # ranking of the removed bitstrings
        "stats_psi": [],  # sampled bitstrings of |ψ⟩ and their counts
        "coeff": [],  # Coefficients for A(t) = ∑c_α Pauli_α
        "used_coeff": [],  # Actually used coefficients in the expansion
        "opt": [],  # optimization results for t in e^-iAt
    }

    bitstring_order = np.argsort(-(np.abs(psi_0) ** 2))
    for k in range(1, M + 1):
        data_k = remove_largest_bitstring(
            psi_0,
            psi_k=data["exp_A_psi"][-1] if len(data["exp_A_psi"]) > 0 else psi_0,
            removed=data["removed"] if len(data["removed"]) > 0 else [],
            pauli_matrices=pauli_matrices,
            **kwargs,
        )

        for key, val in data_k.items():
            data[key].append(val)
        removed_k = data["removed"]
        rank = int(np.argmax(bitstring_order == removed_k[-1]))
        data["rank"].append(rank)

        # Check which indices we have found
        missing = [i for i in bitstring_order[: len(removed_k)] if i not in removed_k]
        found = [i for i in bitstring_order[: len(removed_k)] if i in removed_k]
        extra = [i for i in removed_k if i not in bitstring_order[: len(removed_k)]]
        assert len(extra) == len(missing)
        assert len(extra) + len(found) == k
        missing_with_order = [(i, np.argmax(bitstring_order == i)) for i in missing]
        extra_with_order = [(i, np.argmax(bitstring_order == i)) for i in extra]

        if print_level > 0:
            print(
                f"{k: 3d}/{M:d}, new bitstring: {removed_k[-1]: 4d} which is {rank}-th in order"
            )
            if len(missing) > 0 and print_level > 1:
                _print_missing(missing_with_order, extra_with_order)

    return data


def _print_missing(miss_inds, extra_inds):
    print("| Missing | Pos | Extra | Pos |")
    print("|--------:|----:|------:|----:|")
    for i in range(len(miss_inds)):
        print(
            f"| {miss_inds[i][0]: 7d} | {miss_inds[i][1]: 3d} | {extra_inds[i][0]: 5d} | {extra_inds[i][1] + 1: 3d} |"
        )
