import numba as nb

@nb.njit(parallel=True)
def spmv(n, nnz, val, row, col, x, y):
    for i in nb.prange(n):
        for j in range(row[i] - 1, row[i + 1] - 1):
            y[i] += val[j] * x[col[j]]
