using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using OpenCV = Emgu.CV;




namespace ConvNet.Utilities
{
    static class Converters
    {
        /// <summary>
        /// Matrix[] → Vector
        /// </summary>
        /// <param name="matrices"></param>
        /// <returns></returns>
        public static MathNet.Numerics.LinearAlgebra.Vector<double> ToVector(MathNet.Numerics.LinearAlgebra.Matrix<double>[] matrices, int h, int w)
        {
            double[] _vec = new double[matrices.Length * h * w];
            int _idx = 0;
            foreach (var _ in matrices)
            {
                for (int y = 0; y < _.RowCount; y++)
                {
                    for (int x = 0; x < _.ColumnCount; x++)
                    {
                        _vec[_idx] = _[y, x];
                        _idx++;
                    }
                }
            }

            return MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(_vec);
        }

        public static MathNet.Numerics.LinearAlgebra.Matrix<double>[] ToBRGMatrix(MathNet.Numerics.LinearAlgebra.Matrix<double>[] matrices)
        {
            return new MathNet.Numerics.LinearAlgebra.Matrix<double>[3] { matrices[2], matrices[1], matrices[0] };

        }


        /// <summary>
        /// Matrix → Vector
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public static MathNet.Numerics.LinearAlgebra.Vector<double> ToVector(MathNet.Numerics.LinearAlgebra.Matrix<double> matrix)
        {
            double[] _vec = new double[matrix.RowCount * matrix.ColumnCount];
            int _idx = 0;
            for (int y = 0; y < matrix.RowCount; y++)
            {
                for (int x = 0; x < matrix.ColumnCount; x++)
                {
                    _vec[_idx] = matrix[y, x];
                    _idx++;
                }
            }
            return MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(_vec);
        }

        /// <summary>
        /// Vector → Matrix[]
        /// </summary>
        /// <param name="vector"></param>
        /// <returns></returns>
        public static MathNet.Numerics.LinearAlgebra.Matrix<double>[] ToMatrices(MathNet.Numerics.LinearAlgebra.Vector<double> vector, int d, int h, int w)
        {
            if (vector.Count != d * h * w) { throw new ArgumentException("vector size != d * h * w"); }
            MathNet.Numerics.LinearAlgebra.Matrix<double>[] _mats = new MathNet.Numerics.LinearAlgebra.Matrix<double>[d];
            double[,] _mat = new double[h, w];
            int _idx = 0;
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    for (int k = 0; k < w; k++)
                    {
                        _mat[j, k] = vector[_idx];
                        _idx++;
                    }
                }
                _mats[i] = MathNet.Numerics.LinearAlgebra.Matrix<double>.Build.DenseOfArray(_mat);
            }
            return (MathNet.Numerics.LinearAlgebra.Matrix<double>[])_mats.Clone();
        }

        /// <summary>
        /// Vector → Matrix
        /// </summary>
        /// <param name="vector"></param>
        /// <returns></returns>
        public static MathNet.Numerics.LinearAlgebra.Matrix<double> ToMatrix(MathNet.Numerics.LinearAlgebra.Vector<double> vector, int h, int w)
        {
            if (vector.Count != h * w) { throw new ArgumentException("vector size != h * w"); }
            double[,] _mat = new double[h, w];
            int _idx = 0;
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    _mat[i, j] = vector[_idx];
                    _idx++;
                }
            }
            return MathNet.Numerics.LinearAlgebra.Matrix<double>.Build.DenseOfArray(_mat);
        }

        /// <summary>
        /// Math.Net -> OpenCV
        /// </summary>
        /// <param name="m">MathNet.Numerics.LinearAlgebra.Matrix</param>q342r
        /// <returns>Emgu.CV.Matrix<double></returns>
        public static Emgu.CV.Matrix<double> ToOpenCV(MathNet.Numerics.LinearAlgebra.Matrix<double> m)
        {

            Emgu.CV.Matrix<double> res = new Emgu.CV.Matrix<double>(m.RowCount, m.ColumnCount);
            for (int r = 0; r < m.RowCount; ++r)
            {
                for (int c = 0; c < m.ColumnCount; ++c)
                {
                    res[r, c] = m[r, c];
                }
            }
            return res;
        }


        /// <summary>
        /// OpenCV -> Math.Net
        /// </summary>
        /// <param name="m"> Emgu.CV.Matrix</param>
        /// <returns>MathNet.Numerics.LinearAlgebra.Matrix</returns>
        public static MathNet.Numerics.LinearAlgebra.Matrix<double> ToMathNet(Emgu.CV.Matrix<double> m)
        {
            return MathNet.Numerics.LinearAlgebra.Matrix<double>.Build.DenseOfArray(m.Data);
        }


        /// <summary>
        /// Vector to double[]
        /// </summary>
        /// <param name="v">MathNet.Numerics.LinearAlgebra.Vector</param>
        /// <returns>double[]</returns>
        public static double[] ToInterop(this MathNet.Numerics.LinearAlgebra.Vector<double> v) { return v.AsArray(); }

        /// <summary>
        /// Matrix to double[,]
        /// </summary>
        /// <param name="v">MathNet.Numerics.LinearAlgebra.Matrix</param>
        /// <returns>double[,]</returns>
        public static double[,] ToInterop(this MathNet.Numerics.LinearAlgebra.Matrix<double> m) { return m.AsArray(); }


    }

}
