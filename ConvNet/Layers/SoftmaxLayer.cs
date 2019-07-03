using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace ConvNet.Layers
{
    class SoftmaxLayer : FullyConnectedLayer<Activations.Identity>
    {

        const double YMIN = 1.0e-10;
        Func<double, double> _obd_f = new Func<double, double>(value => { return value < YMIN ? YMIN : value; });

        public SoftmaxLayer(int inputSize, int outputSize, string layerName = "", Vector<double> weights = null, Vector<double> biases = null)
            : base(inputSize, outputSize, layerName, weights, biases)
        {
            LayerType = "SoftmaxLayer";
            GenericsType = "Softmax";
        }

        public override void ForwardPropagation()
        {
            // _e_[i] = Σ w_[j,i] * x_[i] + b_[i]
            var _e = weights.TransposeThisAndMultiply(inputs) + biases;
            // _e_[i] = exp[(_e_[i] - max(_e))]
            _e = (_e - _e.Maximum()).PointwiseExp().Map(_obd_f);
            // _outputs_[i] = _e_[i] / Σ _e
            outputs = (_e / _e.Sum());
        }

        public override Vector<double> BackPropagation(Vector<double> next_delta)
        {
            var curt_delta = next_delta.Clone();
            Parallel.For(0, outputSize, osz =>
            {
                // δ_[n,j] = curt_delta_[n+1,j] * φ' = curt_delta_[n+1,j] * outputs[j] * (1.0 - outputs[j])
                curt_delta[osz] *= (outputs[osz] * (1.0 - outputs[osz]));
                // Δb_[n,j] = δ_[n,j]
                _db[osz] += curt_delta[osz];
                // Δw_[n] = δ_[n] * _inputs_[n]
                for (int isz = 0; isz < inputSize; isz++)
                {
                    _dw[isz, osz] += curt_delta[osz] * inputs[isz];
                }
            });

            // return : Σ w_[n] * δ_[n]
            return (weights * curt_delta).Clone();
        }

        public override Vector<double> Outputs
        {
            get { return outputs.Clone(); }
        }

        public override string ToString(string format = "o")
        {
            StringBuilder _res;
            switch (format)
            {
                case "b":
                    _res = new StringBuilder("#Biases\n" + biases.Count + "\n", biases.Count * 8);
                    for (int bsz = 0; bsz < biases.Count; bsz++)
                    {
                        _res.Append(biases[bsz] + "\t");
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
                    _res = new StringBuilder("#Output\n" + outputSize + "\n", outputSize * 8);
                    for (int osz = 0; osz < outputSize; osz++)
                    {
                        _res.Append(outputs[osz] + "\t");
                    }
                    break;
                case "l":
                    _res = new StringBuilder("Inputs:" + inputSize + ", " + "Outputs:" + outputSize + ", " +
                        "Weights:" + weightHeight + "x" + weightWidth + ", " +
                        "Biases:" + outputSize);
                    break;
                default:
                    _res = new StringBuilder("None\n");
                    break;
            }

            return _res.ToString();
        }
    }

}
