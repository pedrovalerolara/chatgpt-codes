#include <omp.h>

void sgemv(int m, int n, float alpha, float *a, float *x, float beta, float *y)
{
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        float sum = 0.0;
        for (int j = 0; j < n; j++)
        {
            sum += a[i * n + j] * x[j];
        }
        y[i] = alpha * sum + beta * y[i];
    }
}
