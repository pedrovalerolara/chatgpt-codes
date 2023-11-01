void jacobi3D(double ***in, double ***out, int N, int T)
{
  int t, i, j, k;
  double ***temp;

  for (t = 0; t < T; t++)
  {
    for (i = 1; i < N - 1; i++)
    {
      for (j = 1; j < N - 1; j++)
      {
        for (k = 1; k < N - 1; k++)
        {
          out[i][j][k] = (in[i - 1][j][k] + in[i + 1][j][k] + in[i][j - 1][k] + in[i][j + 1][k] + in[i][j][k - 1] + in[i][j][k + 1]) / 6.0;
        }
      }
    }
    temp = out;
    out = in;
    in = temp;
  }
}
