void sgemm(int m, int n, int k, float *a, float *b, float *c)
{
    #pragma acc parallel loop collapse(2) independent
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
