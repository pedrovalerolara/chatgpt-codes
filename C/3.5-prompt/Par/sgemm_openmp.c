#include <omp.h>

void sgemm(int m, int n, int k, float *a, float *b, float *c)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int l = 0; l < k; l++)
            {
                c[i * n + j] += a[i * k + l] * b[l * n + j];
            }
        }
    }
}
