import cupy as cp

def spmv(n, nnz, val, row, col, x, y):
    val_cp = cp.array(val)
    row_cp = cp.array(row)
    col_cp = cp.array(col)
    x_cp = cp.array(x)
    y_cp = cp.array(y)

    for i in range(n):
        start = row_cp[i] - 1
        end = row_cp[i + 1] - 1 if i + 1 < n else nnz
        indices = cp.arange(start, end)
        selected_val = val_cp[indices]
        selected_col = col_cp[indices]

        y_cp[i] += cp.sum(selected_val * x_cp[selected_col])

    cp.copyto(y, y_cp)  # Copy back the result to the original 'y' array
