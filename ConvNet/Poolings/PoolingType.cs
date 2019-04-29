using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvNet.Poolings
{
    interface PoolingType
    {
        double f(Matrix<double> m);

        Matrix<double> df(Matrix<double> m);
        
        string Type();
    }
    public class Max : PoolingType
    {
        public double f(Matrix<double> m) { return m.Enumerate().Max(); }
        public Matrix<double> df(Matrix<double> m)
        {
            double max = -double.MaxValue;
            int maxi = 0, maxj = 0;
            for (int i = 0; i < m.RowCount; i++)
            {
                for (int j = 0; j < m.ColumnCount; j++)
                {
                    if (max < m[i, j]) { max = m[i, j]; maxi = i; maxj = j; }
                }
            }

            return Matrix<double>.Build.Dense(m.RowCount, m.ColumnCount,
                new Func<int, int, double>((i, j) => { return i == maxi && j == maxj ? 1 : 0; })
            );
        }
        public string Type() { return "Max"; }
    }
    public class Average : PoolingType
    {
        public double f(Matrix<double> m) { return m.Enumerate().Average(); }
        public Matrix<double> df(Matrix<double> m)
        {
            double N = m.RowCount * m.ColumnCount;
            return m / N;
        }
        public string Type() { return "Average"; }
    }
}
