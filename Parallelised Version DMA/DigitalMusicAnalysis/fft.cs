using System;
using System.Numerics;
using System.Threading.Tasks;
using System.Diagnostics;

namespace DigitalMusicAnalysis
{
    class FFT
    {
        // Method to reverse bits of an integer for bit-reversed addressing
        private static int ReverseBits(int bits, int n)
        {
            int rev = 0;
            for (int i = 0; i < bits; i++)
            {
                rev = (rev << 1) | (n & 1);
                n >>= 1;
            }

            return rev;
        }

        // Iterative Cooley-Tukey FFT implementation
        public static Complex[] IterativeCTFFT(Complex[] x, int numThreads)
        {
            int N = x.Length;
            Complex[] Y = new Complex[N];
            int bits = (int)Math.Log(N, 2);

            // Bit-reversed addressing reordering
            Parallel.For(0, N, new ParallelOptions { MaxDegreeOfParallelism = numThreads }, i =>
            {
                int pos = ReverseBits(bits, i);
                Y[i] = x[pos];
            });

            // Iterative FFT computation - layer by layer
            for (int step = 2; step <= N; step <<= 1)
            {
                int halfStep = step / 2;
                double angle = -2 * Math.PI / step;
                Complex wStep = new Complex(Math.Cos(angle), Math.Sin(angle));

                Parallel.For(0, N / step, new ParallelOptions { MaxDegreeOfParallelism = numThreads }, groupIdx =>
                {
                    int groupStart = groupIdx * step;
                    Complex w = Complex.One;

                    for (int pair = 0; pair < halfStep; pair++)
                    {
                        int match = groupStart + pair + halfStep;
                        int group = groupStart + pair;

                        if (match < N && group < N) // Check to avoid out-of-bounds access
                        {
                            Complex temp = w * Y[match];
                            Y[match] = Y[group] - temp;
                            Y[group] += temp;
                        }

                        w *= wStep;
                    }
                });
            }

            return Y;
        }
    }
}
