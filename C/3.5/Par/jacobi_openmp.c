#include <omp.h>

void jacobi3D(double ***in, double ***out, int N, int T)
{
    double ***temp;
    
    // Number of threads to use (adjust as needed)
    int num_threads = omp_get_max_threads();
    for (int t = 0; t < T; t++)
    {
      #pragma omp parallel for num_threads(num_threads) collapse(3)
      for (int i = 1; i < N - 1; i++)
      {
        for (int j = 1; j < N - 1; j++)
        {
          for (int k = 1; k < N - 1; k++)
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
