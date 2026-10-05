# Boundary MPS with VUMPS

Approximate the dominant fixed point of a uniform transfer MPO using
[`vumpsMPO`](vumpsTransferMatrix/vumps.py), for example to obtain the boundary MPS of a 2D PEPS.

## Install

From the repository root (requires NumPy and SciPy):

```sh
python3 -m pip install -e .
```

## Use `vumpsMPO`

Supply a NumPy MPO tensor of shape `(D, d, d, D)`, with axes
**(left bond, bra physical, ket physical, right bond)**.

```python
AC, C, AL, AR, leftenv, rightenv, mpo_lam = vumpsMPO(
    mpoTensor=mpo,
    chi=16,          # MPS bond dimension; must be >= 2
    tol=1e-8,       # convergence threshold
    eigs_tol=1e-12,  # canonicalization / eigensolver accuracy
    maxIter=200,
    info=True,      # print iteration diagnostics
)
```

- `AC`, `AL`, `AR`: center, left-canonical, and right-canonical MPS tensors, shape `(chi, d, chi)`.
- `C`: center matrix, shape `(chi, chi)`; at convergence, `AC ≈ AL @ C ≈ C @ AR`, contracting bond indices.
- `leftenv`, `rightenv`: normalized environments, shape `(chi, D, chi)`.
- `mpo_lam`: dominant transfer eigenvalue per site at the MPS fixed point.

Initialization is random by default. Pass `A0=initial_mps` with shape
`(chi, d, chi)` to supply an initial injective MPS; its bond dimension overrides `chi`.
Both the projected residual and canonical error must fall below `tol`.
Failure to converge within `maxIter` raises `RuntimeError`.
Defaults: `tol=eigs_tol=1e-14`, `maxIter=100`, `info=True`.

For an MPO construction example, see [`ising_model_test/order_parameter.py`](ising_model_test/order_parameter.py).

## Use `tensor_equation_solver`

[`tensor_equation_solver`](wen_plq_model/tensor_equation_solver.py) finds a normalized
tensor `R` satisfying `(A1 ⊗ ... ⊗ An) @ R.ravel() = R.ravel()` for every constraint.
Pass the individual operators directly; the solver forms their Kronecker products internally.

```python
R, solution_space = tensor_equation_solver(
    [
        [X, X], 
        [Z, Z]
    ]
)
```

- `R`: random complex solution with unit Frobenius norm; operator dimensions determine its shape.
- `solution_space`: orthonormal basis of all solutions, shape `(R.size, number_of_basis_vectors)`; each column is a flattened tensor.


## References

- https://arxiv.org/pdf/1810.07006
- https://tensortrack.readthedocs.io/en/latest/examples/examples.html
