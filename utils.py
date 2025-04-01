from functools import reduce
from typing import List

import numpy as np
import scipy as sc

def fidelity(p: np.ndarray, f: np.ndarray):
    """Calculate fidelity of two states in numpy array format."""
    res = p.conj().T @ f * (f.conj().T @ p) / (p.conj().T @ p) / (f.conj().T @ f)
    assert res.size == 1
    if len(res.shape) == 2:
        res = res[0, 0]
    assert np.abs(res.imag) < 1e-14  # Usually ~ 1e-24
    return float(res.real)


def comm(A, B):
    """Commutator of matrices A and B."""
    return A @ B - B @ A


def expand_in_basis(number: int | np.ndarray, b: int):
    """Expands the number in basis b."""
    max_num = np.max(number)
    min_num = np.max(number)
    assert min_num >= 0

    is_one_num = np.isscalar(number) or number.size == 1
    number = np.atleast_1d(number)

    num_digits = int(np.log2(max_num) / np.log2(b)) if max_num > 0 else 0
    res = number[..., np.newaxis] // b ** np.arange(num_digits, -1, -1) % b
    return res[0] if is_one_num else res


def sample_pdf(
    pdf: np.ndarray, n_samples: int = 10, replace: bool = True
) -> dict:
    """Takes bitstring samples σ from pdf |ψ(σ)|².

    It returns statistics of sampled strings: list of lists where the
    first element is the integer representation of a bitstring and the
    second is the number of occurences.
    """
    assert np.all(pdf >= 0)
    pdf /= np.sum(pdf)
    sampled_strings = np.random.choice(
        pdf.shape[0], size=(n_samples), replace=replace, p=pdf
    )
    # Pointless if replace=False
    bitstrings, counts = np.unique(sampled_strings, return_counts=True)

    return dict(zip(bitstrings, counts))


_str_to_mat = {
    "I": sc.sparse.coo_array((np.eye(2)) / np.sqrt(2)),
    "X": sc.sparse.coo_array((np.array([[0, 1], [1, 0]])) / np.sqrt(2)),
    "Y": sc.sparse.coo_array((np.array([[0, -1j], [1j, 0]])) / np.sqrt(2)),
    "Z": sc.sparse.coo_array((np.array([[1, 0], [0, -1]])) / np.sqrt(2)),
}


# TODO: I guess one could do some D&C algorithm?
def generate_all_pauli_matrices(
    N: int, paulis: List[str] = ["I", "X", "Y", "Z"], _max_N: int = 10
) -> List[sc.sparse.coo_array]:
    """Generates all possible Kronecker product of Paulis up to including order N.

    This is the fast approach to generate all Pauli matrices. User can supply a
    subset of available Paulis."""
    if N > _max_N:
        raise ValueError(
            "N is too large. You can raise _max_N but the algorithm is still exponential."
        )
    p = [_str_to_mat[i] for i in paulis]
    for itr in range(N - 1):
        p = [sc.sparse.kron(i, _str_to_mat[j], format="coo") for i in p for j in paulis]
    return p


def gen_mat_from_str(string: str) -> sc.sparse.coo_array:
    """Generate a matrix from a Pauli string"""
    pauli_matrices = [_str_to_mat[i] for i in string]
    one = sc.sparse.coo_array([[1]])
    return reduce(lambda a, b: sc.sparse.kron(a, b, format="coo"), pauli_matrices, one)


def generate_matrices_from_strings(strings: List[str]) -> List[sc.sparse.coo_array]:
    """Generate matrices from a list of Pauli strings."""
    return [gen_mat_from_str(s) for s in strings]


def load_psi(filename, N):
    """Loads psi from a numpy file and maps into a N-qubit space.

    We assume psi comes from N' > N space but it only has M < 2**N
    non-zero values.
    """
    phi = np.load(filename)
    non_zero = np.abs(phi) > 0
    M = np.sum(non_zero)
    assert np.linalg.norm(phi.imag) < 1e-14
    phi = phi.real

    # Map to smaller N
    assert 2**N > M
    psi = np.zeros((2**N))
    inds = np.random.choice(np.arange(0, 2**N), size=M, replace=False)
    psi[inds] = phi[non_zero]

    return psi / np.linalg.norm(psi)
