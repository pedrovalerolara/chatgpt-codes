#include <omp.h>

void saxpy(int n, float a, float *x, float *y)
{
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        y[i] = a * x[i] + y[i];
    }
}
