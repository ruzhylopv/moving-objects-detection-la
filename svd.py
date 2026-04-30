import numpy as np


def power_iteration(A, tol=1e-15, max_iterations=5000, initial_vector=None):
    n = A.shape[0]
    b = initial_vector if initial_vector is not None else np.random.rand(n)
    b = b / np.linalg.norm(b)

    for _ in range(max_iterations):
        b_new = A @ b
        norm = np.linalg.norm(b_new)
        if norm == 0:
            break
        b_new = b_new / norm
        if np.linalg.norm(b_new - b) < tol:
            break
        b = b_new

    eigenvalue = (b @ A @ b) / (b @ b)
    return eigenvalue, b


def eigen_decomposition(A, tol=1e-15, max_iterations=5000):
    A = A.copy().astype(float)
    n = A.shape[0]
    eigenvalues = []
    eigenvectors = []

    for _ in range(n):
        val, vec = power_iteration(A, tol, max_iterations)
        eigenvalues.append(val)
        eigenvectors.append(vec)
        A = A - val * np.outer(vec, vec)  # deflate

    return eigenvalues, eigenvectors

def gram_schmidt(V):
    """Orthogonalize rows of V using modified Gram-Schmidt."""
    Q = []
    for v in V:
        for q in Q:
            v = v - np.dot(v, q) * q  # subtract projection
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            Q.append(v / norm)
    return np.array(Q)

def svd(A, tol=1e-12):
    A = np.array(A, dtype=float)
    B = A.T @ A

    eigenvalues, eigenvectors = eigen_decomposition(B, tol)

    eigenvalues = np.array(eigenvalues)
    V_T = np.array(eigenvectors)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V_T = V_T[idx]

    singular_values = np.sqrt(np.maximum(eigenvalues, 0))

    mask = singular_values > tol
    singular_values = singular_values[mask]
    V_T = V_T[mask]

    # Re-orthogonalize V with QR to fight deflation drift
    # V, _ = np.linalg.qr(V_T.T)
    # V_T = V.T
    V_T = gram_schmidt(V_T)
    V = V_T.T

    U = A @ V
    U = U / singular_values

    return U, singular_values, V_T





if __name__ == "__main__":
    A1 = np.array([
    [3, -1,  4],
    [2,  5, -2],
    [7,  0,  1],
    [-4, 6,  3]
], dtype=float)
    A2 = np.array([
    [12, -7,  3,  5],
    [-4,  9, 11, -2],
    [6,   1, -8,  7],
    [10, -3,  4, 13],
    [2,  14, -5,  6]
], dtype=float)
    A3 = np.array([
    [17, -4,  9,  2],
    [6,  13, -7,  5],
    [-3,  8, 11, -6],
    [10, -2,  4, 15]
], dtype=float)
    
    A4 = np.array([
    [1, 1, 2],
    [0, 1, 1],
    [1,  0, 1]
], dtype=float)

    X, Y, Z = svd(A1)
    print(X @ np.diag(Y) @ Z)

    
