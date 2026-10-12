import numpy as np
from functools import reduce
from scipy.sparse import kron, vstack, csr_array, eye_array
from scipy.sparse.linalg import aslinearoperator, eigsh

def sparse_kron_all(*operators):
    return reduce(
        lambda A, B: kron(A, B, format="csr"),
        (csr_array(op) for op in operators)
    )



if __name__ == "__main__":
    I = np.array([[1, 0],[0, 1]],dtype=complex)
    X = np.array([[0, 1],[1, 0]],dtype=complex)
    Y = np.array([[0, -1.0j],[1.0j, 0]],dtype=complex)
    Z = np.array([[1, 0],[0, -1] ],dtype=complex)

    A = sparse_kron_all(X, Y)



