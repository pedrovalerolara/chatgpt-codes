void sgemv(int m, int n, float alpha, float *a, float *x, float beta, float *y)
{
    #pragma acc parallel loop present(a, x, y) independent
    for (int i = 0; i < m; i++)
    {
        float sum = 0.0;
        #pragma acc loop reduction(+:sum)
        for (int j = 0; j < n; j++)
        {
            sum += a[i * n + j] * x[j];
        }
        y[i] = alpha * sum + beta * y[i];
    }
}
