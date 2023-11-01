void jacobi3D(double ***in, double ***out, int N, int T)
{
    double ***temp;

    for (int t = 0; t < T; t++)
    {
        #pragma acc parallel loop collapse(3)
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

        // Swap in and out pointers
        temp = out;
        out = in;
        in = temp;
     }
}
