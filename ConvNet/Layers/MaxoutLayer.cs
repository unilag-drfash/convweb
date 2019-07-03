using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace ConvNet.Layers
{
    class MaxoutLayer : FullyConnectedLayer<Activations.Identity>
    {

        new Matrix<double> outputs;
        new Matrix<double> biases;

        public MaxoutLayer(int inputSize, int outputSize, string layerName = "", Vector<double> weights = null, Vector<double> biases = null)
            : base(inputSize, outputSize, layerName, weights, null)
        {
            
            outputs = Matrix<double>.Build.Dense(inputSize, outputSize, 0);

            
            if (biases != null && biases.Count == base.inputSize * base.outputSize) {
                this.biases = Utilities.Converters.ToMatrix(biases, base.inputSize, base.outputSize);
            }
            else {
                this.biases = Matrix<double>.Build.Dense(base.inputSize, base.outputSize, 0);
            }
            _db = Matrix<double>.Build.Dense(base.inputSize, base.outputSize, 0);

            LayerType = "MaxoutLayer Ver.FCL";
            GenericsType = "Maxout";
        }

        public override void ForwardPropagation()
        {
            Parallel.For(0, outputSize, osz =>
            {
                for (int isz = 0; isz < inputSize; isz++)
                {
                    outputs[isz, osz] = weights[isz, osz] * inputs[isz] + biases[isz, osz];
                }
            });
        }

        /// <summary>
        ///  Δb
        /// </summary>
        new Matrix<double> _db;

        public override Vector<double> BackPropagation(Vector<double> next_delta)
        {
            var curt_delta = Matrix<double>.Build.Dense(inputSize, outputSize, 0);

            Parallel.For(0, outputSize, osz =>
            {
                curt_delta[outputs.Column(osz).MaximumIndex(), osz] = next_delta[osz];
                for (int isz = 0; isz < inputSize; isz++)
                {
                    _db[isz, osz] += curt_delta[isz, osz];
                    _dw[isz, osz] += curt_delta[isz, osz] * inputs[isz];
                }
            });

            return (curt_delta.PointwiseMultiply(weights)).RowSums();
        }

        public override void WeightUpdate(double eta, double mu, double lambda)
        {
            // 
            // Δw(t) = -η∂E/∂w(t) + μΔw(t-1) - ηλw(t)
            var _dw_ = -eta * _dw + mu * _pre_dw - eta * lambda * weights;
            weights = weights + _dw_;
            biases -= eta * _db;

            
            _pre_dw = _dw_.Clone();

            _dw.Clear();
            _db.Clear();
        }

        public override Vector<double> Outputs
        {
            get
            {
                Vector<double> _tmp_outputs = Vector<double>.Build.Dense(outputSize);
                for (int osz = 0; osz < outputSize; osz++)
                {
                    _tmp_outputs[osz] = outputs.Column(osz).Maximum();
                }
                return _tmp_outputs.Clone();
            }
        }

        public override Vector<double> Biases
        {
            get
            {
                return Utilities.Converters.ToVector(biases);
            }
            protected set
            {
                if (inputSize * outputSize != value.Count) { throw new ArgumentException("Size of biases are different"); }
                biases = Utilities.Converters.ToMatrix(value, inputSize, outputSize);
            }
        }

        public override Vector<double> PredictOutputs
        {
            get { return this.Outputs; }
        }

        public override string ToString(string fmt = "o")
        {
            StringBuilder _res;
            switch (fmt)
            {
                case "b":
                    _res = new StringBuilder("#Biases\n" + weightHeight + "\t" + weightWidth + "\n", weightSize * 8);
                    for (int bh = 0; bh < weightHeight; bh++)
                    {
                        for (int bw = 0; bw < weightWidth; bw++)
                        {
                            _res.Append(biases[bh, bw] + "\t");
                        }
                        _res.Append("\n");
                    }
                    break;
                case "w":
                    _res = new StringBuilder("#Weights\n" + weightHeight + "\t" + weightWidth + "\n", weightSize * 8);
                    for (int wh = 0; wh < weightHeight; wh++)
                    {
                        for (int ww = 0; ww < weightWidth; ww++)
                        {
                            _res.Append(weights[wh, ww] + "\t");
                        }
                        _res.Append("\n");
                    }
                    break;
                case "i":
                    _res = new StringBuilder("#Inputs\n" + inputSize + "\n", inputSize * 8);
                    for (int isz = 0; isz < inputSize; isz++)
                    {
                        _res.Append(inputs[isz] + "\t");
                    }
                    break;
                case "o":
                    var _o = this.Outputs;
                    _res = new StringBuilder("#Output\n" + outputSize + "\n", outputSize * 8);
                    for (int osz = 0; osz < outputSize; osz++)
                    {
                        _res.Append(activation.f(_o[osz]) + "\t");
                    }
                    break;
                case "l":
                    _res = new StringBuilder("Inputs:" + inputSize + ", " + "Outputs:" + outputSize + ", " +
                        "Weights:" + weightHeight + "x" + weightWidth + ", " +
                        "Biases:" + weightHeight + "x" + weightWidth);
                    break;
                default:
                    _res = new StringBuilder("None\n");
                    break;
            }

            return _res.ToString();
        }
    }

}
