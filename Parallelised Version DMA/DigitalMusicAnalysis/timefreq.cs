using System;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;

namespace DigitalMusicAnalysis
{
    public class timefreq
    {
        public float[][] timeFreqData;
        public int wSamp;
        public Complex[] twiddles;

        public timefreq(float[] x, int windowSamp, int specifiedThreadCount = 0)
        {
            double pi = Math.PI;
            Complex i = Complex.ImaginaryOne;
            this.wSamp = windowSamp;
            twiddles = new Complex[wSamp];

            ParallelOptions parallelOptions = new ParallelOptions();
            if (specifiedThreadCount > 0)
            {
                parallelOptions.MaxDegreeOfParallelism = specifiedThreadCount;
            }

            // Precompute twiddle factors in parallel
            Parallel.For(0, wSamp, parallelOptions, ii =>
            {
                double a = 2 * pi * ii / (double)wSamp;
                twiddles[ii] = Complex.Exp(-i * a);
            });

            timeFreqData = new float[wSamp / 2][];

            int nearest = (int)Math.Ceiling((double)x.Length / (double)wSamp);
            nearest = nearest * wSamp;

            Complex[] compX = new Complex[nearest];
            Parallel.For(0, nearest, parallelOptions, kk => { compX[kk] = (kk < x.Length) ? x[kk] : Complex.Zero; });

            int cols = 2 * nearest / wSamp;

            Parallel.For(0, wSamp / 2, parallelOptions, jj => { timeFreqData[jj] = new float[cols]; });

            timeFreqData = stft(compX, wSamp, parallelOptions);
        }

        float[][] stft(Complex[] x, int wSamp, ParallelOptions parallelOptions)
        {
            int N = x.Length;
            float[][] Y = new float[wSamp / 2][];

            // Allocate Y array with wSamp/2 rows and appropriate number of columns
            Parallel.For(0, wSamp / 2, parallelOptions, ll => { Y[ll] = new float[2 * (int)Math.Floor((double)N / (double)wSamp)]; });

            int iterations = 2 * (int)Math.Floor((double)N / (double)wSamp) - 1;

            // Create a local max array for reduction to avoid locking
            float[] localMax = new float[iterations];

            Parallel.For(0, iterations, parallelOptions, ii =>
            {
                Complex[] temp = new Complex[wSamp];
                Complex[] tempFFT;

                // Fill the temp array with the appropriate window of data
                for (int jj = 0; jj < wSamp; jj++)
                {
                    int index = ii * (wSamp / 2) + jj;
                    if (index < x.Length)
                    {
                        temp[jj] = x[index];
                    }
                    else
                    {
                        temp[jj] = Complex.Zero; // Handle out-of-bounds case
                    }
                }

                tempFFT = fft(temp);

                float maxVal = 0;
                for (int kk = 0; kk < wSamp / 2; kk++)
                {
                    Y[kk][ii] = (float)Complex.Abs(tempFFT[kk]);
                    if (Y[kk][ii] > maxVal)
                    {
                        maxVal = Y[kk][ii];
                    }
                }

                localMax[ii] = maxVal; // Store the maximum value of this iteration
            });

            // Find the global maximum
            float fftMax = localMax.Max();
            if (fftMax == 0) fftMax = 1; // Prevent division by zero

            // Normalize the Y array using the global maximum value
            Parallel.For(0, iterations, parallelOptions, ii =>
            {
                for (int kk = 0; kk < wSamp / 2; kk++)
                {
                    Y[kk][ii] /= fftMax;
                }
            });

            return Y;
        }

        Complex[] fft(Complex[] x)
        {
            // Placeholder implementation for FFT
            // Ensure this is a thread-safe implementation
            int N = x.Length;
            Complex[] Y = new Complex[N];
            if (N == 1)
            {
                Y[0] = x[0];
                return Y;
            }

            Complex[] even = new Complex[N / 2];
            Complex[] odd = new Complex[N / 2];
            for (int ii = 0; ii < N / 2; ii++)
            {
                even[ii] = x[2 * ii];
                odd[ii] = x[2 * ii + 1];
            }

            Complex[] E = fft(even);
            Complex[] O = fft(odd);

            for (int kk = 0; kk < N / 2; kk++)
            {
                Complex t = twiddles[kk * wSamp / N] * O[kk];
                Y[kk] = E[kk] + t;
                Y[kk + N / 2] = E[kk] - t;
            }

            return Y;
        }
    }
}
