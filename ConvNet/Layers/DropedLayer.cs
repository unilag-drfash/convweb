using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace ConvNet.Layers
{
    class DropedLayer
    {
    }

    // DropOut and DropConnect

    /// <summary>
    /// DropOut Layer
    /// </summary>
    class DropOutLayer<ActivationType> : FullyConnectedLayer<ActivationType> where ActivationType : Activations.IActivation, new()
    {
        private MathNet.Numerics.Distributions.ContinuousUniform rand;

        /// <summary>
        /// DropOut Probability
        /// </summary>
        private double dropProbability;

        /// <summary>
        /// DropOut Mask
        /// </summary>
        private int[] dropMask;

        public DropOutLayer(
            int inputSize, int outputSize, double dropProbability = 0.5, string layerName = "",
            Vector<double> weights = null, Vector<double> biases = null)
            : base(inputSize, outputSize, layerName, weights, biases)
        {
            // Uniform Random Value
            rand = new ContinuousUniform();

            // DropOut Condition
            if (dropProbability > 1.0 || dropProbability < 0)
            {
                Console.WriteLine("0 < dropProbability < 1.0. dropProbability is set 0.5");
                this.dropProbability = 0.5;
            }
            else { this.dropProbability = dropProbability; }

            
            dropMask = new int[outputSize];
            CreateDropMask();

            LayerType = "DropOutLayer";
            GenericsType = activation.Type();
        }


        public override void ForwardPropagation()
        {
            base.ForwardPropagation();
        }

        public override Vector<double> BackPropagation(Vector<double> next_delta)
        {
            Vector<double> curt_delta = next_delta;
            //for (int osz = 0; osz < outputSize; osz++)
            Parallel.For(0, outputSize, osz =>
            {
                if (dropMask[osz] == 1)
                {
                    // δ_[n,j] = next_delta_[n+1,j] * φ'(_outputs_[n,j]) * dropMask[j]
                    curt_delta[osz] *= (activation.df(outputs[osz]) * dropMask[osz]);
                    // Δb_[n,j] = (Σ w_[n+1] * δ_[n+1]) * φ'(_outputs_[n,j]) = curt_delta
                    _db[osz] += curt_delta[osz];
                    // Δw_[n] = δ_[n] * input_[n]
                    for (int isz = 0; isz < inputSize; isz++) { _dw[isz, osz] += curt_delta[osz] * inputs[isz]; }
                }
            });

            // return : Σ w_[n] * δ_[n]
            return (weights * curt_delta).Clone();
        }

        public override void WeightUpdate(double eta, double mu, double lambda)
        {
            base.WeightUpdate(eta, mu, lambda);

            // Mini-batch size
            CreateDropMask();
        }

        /// <summary>
        /// DropOut mask
        /// </summary>
        private void CreateDropMask()
        {
            for (int i = 0; i < dropMask.Length; i++)
            {
                dropMask[i] = rand.Sample() < dropProbability ? 0 : 1;
            }
        }

 
        public override Vector<double> Outputs
        {
            get
            {
                return Vector<double>.Build.DenseOfEnumerable(
                    outputs.Select((val, idx) => { return dropMask[idx] == 1 ? activation.f(val) : 0; })
                );
            }
        }

        
        public override Vector<double> PredictOutputs
        {
            get
            {
                return Vector<double>.Build.DenseOfEnumerable(
                    outputs.Select(val => activation.f(val) * (1.0 - dropProbability))
                );
            }
        }

        public override string ToString(string fmt = "o")
        {
            StringBuilder _res;
            switch (fmt)
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
                case "lo":
                    _res = new StringBuilder("#Learning Output\n" + outputSize + "\n", outputSize * 8);
                    for (int osz = 0; osz < outputSize; osz++)
                    {
                        if (dropMask[osz] == 0) { _res.Append(0 + "\t"); }
                        else
                        {
                            _res.Append(activation.f(outputs[osz]) + "\t");
                        }
                    }
                    break;
                case "po":
                    _res = new StringBuilder("#Predict Output\n" + outputSize + "\n", outputSize * 8);
                    for (int osz = 0; osz < outputSize; osz++)
                    {
                        _res.Append((activation.f(outputs[osz]) * (1.0 - dropProbability)) + "\t");
                    }
                    break;
                case "l":
                    _res = new StringBuilder("Inputs:" + inputSize + ", " + "Outputs:" + outputSize + ", " +
                        "Weights:" + weightHeight + "x" + weightWidth + ", " +
                        "Biases:" + outputSize + ", " + "DropOutProb:" + dropProbability);
                    break;
                default:
                    _res = new StringBuilder("None\n");
                    break;
            }

            return _res.ToString();
        }
    }


    /// <summary>
    /// DropConnect Layer
    /// </summary>
    class DropConnectLayer<ActivationType> : FullyConnectedLayer<ActivationType> where ActivationType : Activations.IActivation, new()
    {
        private MathNet.Numerics.Distributions.ContinuousUniform rand;

        /// <summary>
        /// DropOut Probability
        /// </summary>
        private double dropProbability;

        /// <summary>
        /// DropConnect する荷重接続のマスク
        /// </summary>
        private Matrix<double> dropMask;

        public DropConnectLayer(
            int inputSize, int outputSize, double dropProbability = 0.5, string layerName = "",
            Vector<double> weights = null, Vector<double> biases = null)
            : base(inputSize, outputSize, layerName, weights, biases)
        {
            rand = new ContinuousUniform();

            // DropConnect Probaibility
            if (dropProbability > 1.0 || dropProbability < 0)
            {
                Console.WriteLine("0 < dropProbability < 1.0. dropProbability is set 0.5");
                this.dropProbability = 0.5;
            }
            else { this.dropProbability = dropProbability; }

            
            dropMask = Matrix<double>.Build.Dense(inputSize, outputSize, 0);
            CreateDropMask();

            LayerType = "DropConnectLayer";
            GenericsType = activation.Type();
        }

        public override void ForwardPropagation()
        {
            // outputs = (Mask*W)^T * inputs
            outputs = (dropMask.PointwiseMultiply(weights)).TransposeThisAndMultiply(inputs) + biases;
        }

        public override Vector<double> BackPropagation(Vector<double> next_delta)
        {
            Vector<double> curt_delta = next_delta;
            //for (int osz = 0; osz < outputSize; osz++)
            Parallel.For(0, outputSize, osz =>
            {
                // δ_[n,j] = next_delta_[n+1,j] * φ'(_outputs_[n,j])
                curt_delta[osz] *= activation.df(outputs[osz]);
                // Δb_[n,j] = (Σ w_[n+1] * δ_[n+1]) * φ'(_outputs_[n,j]) = curt_delta
                _db[osz] += curt_delta[osz];
                // Δw_[n] = δ_[n] * input_[n]
                for (int isz = 0; isz < inputSize; isz++) { _dw[isz, osz] += curt_delta[osz] * inputs[isz]; }
            });

            // return : Σ w_[n] * δ_[n]
            return (weights * curt_delta).Clone();
        }

        public override Vector<double> PredictOutputs
        {
            get
            {
                return Vector<double>.Build.DenseOfEnumerable(
                    (weights.TransposeThisAndMultiply(inputs) + biases).Select(val =>
                    {
                        return activation.f(val) * (1.0 - dropProbability);
                    })
                );
            }
        }

        public override void WeightUpdate(double eta, double mu, double lambda)
        {
            base.WeightUpdate(eta, mu, lambda);

            // Mini-batch size
            CreateDropMask();
        }

        /// <summary>
        /// DropConnect mask
        /// </summary>
        private void CreateDropMask()
        {
            for (int isz = 0; isz < inputSize; isz++)
            {
                for (int osz = 0; osz < outputSize; osz++)
                {
                    dropMask[isz, osz] = rand.Sample() < dropProbability ? 0 : 1;
                }
            }
        }

        public override string ToString(string fmt = "o")
        {
            StringBuilder _res;
            switch (fmt)
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
                case "lo":
                    _res = new StringBuilder("#Learning Output\n" + outputSize + "\n", outputSize * 8);
                    for (int osz = 0; osz < outputSize; osz++)
                    {
                        _res.Append(activation.f(outputs[osz]) + "\t");
                    }
                    break;
                case "po":
                    var _o = this.PredictOutputs;
                    _res = new StringBuilder("#Predict Output\n" + outputSize + "\n", outputSize * 8);
                    for (int osz = 0; osz < outputSize; osz++)
                    {
                        _res.Append(_o[osz] + "\t");
                    }
                    break;
                case "l":
                    _res = new StringBuilder("Inputs:" + inputSize + ", " + "Outputs:" + outputSize + ", " +
                        "Weights:" + weightHeight + "x" + weightWidth + ", " +
                        "Biases:" + outputSize + ", " + "DropConnectProb:" + dropProbability);
                    break;
                default:
                    _res = new StringBuilder("None\n");
                    break;
            }

            return _res.ToString();
        }
    }

}
