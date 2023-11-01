#include <omp.h>

void csr_spmv(int n, int *row_ptr, int *col_idx, double *values, double *x, double *y)
{
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];
        for (int j = row_start; j < row_end; j++)
        {
            y[i] += values[j] * x[col_idx[j]];
        }
    }
}
