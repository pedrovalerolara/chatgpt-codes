import cupy as cp

def jacobi(A, num_iterations):
    m, n, p = A.shape
    B = cp.empty_like(A)

    for k in range(num_iterations):
        B[1:-1, 1:-1, 1:-1] = 0.125 * (
            A[:-2, 1:-1, 1:-1] + A[2:, 1:-1, 1:-1] +
            A[1:-1, :-2, 1:-1] + A[1:-1, 2:, 1:-1] +
            A[1:-1, 1:-1, :-2] + A[1:-1, 1:-1, 2:]
        )
        A[...] = B

    return A
