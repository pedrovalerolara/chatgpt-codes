from numba import njit, prange

@njit(parallel=True)
def jacobi_parallel(A, B, num_iterations):
    m, n, p = A.shape
    for k in range(num_iterations):
        for i in prange(1, m - 1):
            for j in prange(1, n - 1):
                for l in prange(1, p - 1):
                    B[i, j, l] = 0.125 * (A[i - 1, j, l] + A[i + 1, j, l] + A[i, j - 1, l] + A[i, j + 1, l] + A[i, j, l - 1] + A[i, j, l + 1])
        A, B = B, A
    return A
