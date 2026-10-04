import numpy as np
from functools import reduce
from scipy.linalg import null_space

PAULI = {
    'I': np.array([[1,0],[0,1]], dtype=complex),
    'X': np.array([[0,1],[1,0]], dtype=complex),
    'Y': np.array([[0,-1j],[1j,0]], dtype=complex),
    'Z': np.array([[1,0],[0,-1]], dtype=complex)
}

def kron_all(*operators):
    if len(operators) == 1 and isinstance(operators[0], (list, tuple)):
        operators = tuple(operators[0])

    if not operators:
        raise ValueError("At least one matrix is required.")

    return reduce(np.kron, operators)


def tensor_equation_solver(constraints, rng=None):
    """Sample a normalized tensor satisfying all product-operator constraints.

    For each constraint ``(A_1, ..., A_n)``, solve
    ``kron_all(A_1, ..., A_n) @ R.ravel() = R.ravel()``. The common
    solution space is the null space of the vertically stacked matrices
    ``kron_all(A_1, ..., A_n) - I``. A random complex linear combination
    of its orthonormal basis vectors is reshaped and normalized to obtain R.

    Parameters
    ----------
    constraints : sequence of sequences of numpy.ndarray
        Nonempty collection of constraints, each containing one square
        operator per tensor axis. Every constraint must have the same
        operator shapes. The operator dimensions determine the tensor shape.
    atol : float, optional
        Currently unused. The null space is computed using SciPy's default
        singular-value cutoff.
    rng : optional
        Seed or random number generator accepted by ``np.random.default_rng``.
        Defaults to None, which creates a generator with fresh entropy.

    Returns
    -------
    R : numpy.ndarray
        Complex tensor with unit Frobenius norm, satisfying each constraint
        to numerical precision. Its shape is inferred from the operators.
    solution_space : numpy.ndarray
        Orthonormal basis for the common solution space, with shape
        ``(R.size, nullity)``. Each column represents a flattened tensor
        in NumPy's default C order.

    Raises
    ------
    ValueError
        If constraints is empty or operator shapes differ from the square
        shapes inferred from the first constraint.
    RuntimeError
        If the constraints have no nonzero common solution.
    """
    if not constraints:
        raise ValueError('At least one constraint is required.')

    # Infer tensor shape from the constraints
    tensor_shape = tuple(operator.shape[0] for operator in constraints[0])
    expected_shapes = tuple((dimension, dimension) for dimension in tensor_shape)
    for constraint_number, operators in enumerate(constraints, start=1):
        operator_shapes = tuple(operator.shape for operator in operators)
        if operator_shapes != expected_shapes:
            raise ValueError(
                f'Constraint {constraint_number} has operator shapes '
                f'{operator_shapes}; expected {expected_shapes}.'
            )

    tensor_size = int(np.prod(tensor_shape))
    tensor_identity = np.eye(tensor_size, dtype=complex)
    constraint_blocks = [
        kron_all(operators) - tensor_identity
        for operators in constraints
    ]
    constraint_matrix = np.vstack(constraint_blocks)
    solution_space = null_space(constraint_matrix)

    if solution_space.shape[1] == 0:
        raise RuntimeError(
            f'The constraints have no nonzero common solution.'
        )


    rng = np.random.default_rng(rng)
    coefficients = (
        rng.standard_normal(solution_space.shape[1])
        + 1j * rng.standard_normal(solution_space.shape[1])
    )
    coefficients /= np.linalg.norm(coefficients)
    R = (solution_space @ coefficients).reshape(tensor_shape)
    R = R/np.linalg.norm(R)

    return R, solution_space

def dominant_eigenvals(M, round_off = True):
    eigenvalues = np.linalg.eigvals(M)
    spectral_radius = np.abs(eigenvalues).max()
    dominant_tolerance = 1e-10
    dominant_eigenvalues = eigenvalues[
        np.isclose(
            np.abs(eigenvalues),
            spectral_radius,
            rtol=dominant_tolerance,
            atol=0.0,
        )
    ]
    
    # Dividing by a reference eigenvalue fixes both its modulus and phase.
    dominant_eigenvalues /= dominant_eigenvalues[0]
    if round_off:
        dominant_eigenvalues = np.round(dominant_eigenvalues)
        return dominant_eigenvalues
    else: return dominant_eigenvalues 
    
