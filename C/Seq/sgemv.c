void sgemv(int m, int n, float alpha, float *a, float *x, float beta, float *y)
{ 
  float sum;
  for (i = 0; i < m; i++) {
    sum = 0.0;
    for (j = 0; j < n; j++) {
      sum += A[i * n + j] * x[j];
    }
    y[i] = alpha * sum + beta * y[i];
  }
}
