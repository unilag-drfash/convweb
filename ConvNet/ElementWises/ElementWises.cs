using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace ConvNet.ElementWises
{
 
    interface IElementWise
    {
        /// <summary>
        /// f(x)
        /// </summary>
        /// <param name="ms"></param>
        /// <param name="y"></param>
        /// <param name="x"></param>
        /// <returns></returns>
        double f(Matrix<double>[] ms, int y, int x);
        /// <summary>
        /// f'(x)
        /// </summary>
        /// <param name="ms"></param>
        /// <returns></returns>
        double[] df(double[] ms);
        /// <summary>
        /// element wise type
        /// </summary>
        /// <returns></returns>
        string Type();
    }
    /// <summary>
    /// Output maximum value from all input
    /// </summary>
    public class MaxOut : IElementWise
    {
        public double f(Matrix<double>[] ms, int y, int x) { return ms.Max(_ => _[y, x]); }
        public double[] df(double[] ms)
        {
            double max = ms[0];
            int _idx = 0;
            for (int i = 1; i < ms.Length; i++)
            {
                if (max < ms[i]) { _idx = i; max = ms[i]; }
            }
            double[] _df = new double[ms.Length];
            _df[_idx] = 1;
            return (double[])_df.Clone();
        }
        public string Type() { return "MaxOut"; }
    }
    /// <summary>
    /// Output average of all input
    /// </summary>
    public class Average : IElementWise
    {
 
        public double f(Matrix<double>[] ms, int y, int x) { return ms.Average(_ => _[y, x]); }
        public double[] df(double[] ms)
        {
            double[] _df = new double[ms.Length];
            for (int i = 0; i < _df.Length; i++) { _df[i] = 1.0 / _df.Length; }
            return (double[])_df.Clone();
        }
        public string Type() { return "Average"; }
    }

}
