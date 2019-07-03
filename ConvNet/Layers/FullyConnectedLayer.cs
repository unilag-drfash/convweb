using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace ConvNet.Layers
{
    
    class FullyConnectedLayer<ActivationType> : Layer
        where ActivationType : Activations.IActivation, new()
    {
   
        protected ActivationType activation = new ActivationType();

        /// input[current] = output[previous] = activationFunction(Σ weight[previous] * input[previous])
        protected Vector<double> inputs;

        public override Vector<double> Inputs
        {
            set
            {
                if (inputSize != value.Count) { throw new ArgumentException("Size of inputs is different"); }
                inputs = value.Clone();
            }
        }
        
        /// output[current] = Σ weight[current] * input[current]
        protected Vector<double> outputs;


        /// Output = activation_f(this.outputs)
        public override Vector<double> Outputs { get { return outputs.Map(activation.f); } }
        protected int weightHeight, weightWidth, weightSize;

        /// weight[inputIndex][outputIndex] at each connection from input to output 
        protected Matrix<double> weights;

        public override Vector<double> Weights
        {
            get { return Utilities.Converters.ToVector(weights); }
            protected set
            {
                if (weightHeight * weightWidth != value.Count) { throw new ArgumentException("Size of weights is different"); }
                weights = Utilities.Converters.ToMatrix(value, weightHeight, weightWidth);
            }
        }


        protected Vector<double> biases;
        public override Vector<double> Biases
        {
            get
            {
                return biases.Clone();
            }
            protected set
            {
                if (outputSize != value.Count) { throw new ArgumentException("Size of biases are different"); }
                biases = value.Clone();
            }
        }
         /// <param name="in_size">Input Size</param>
        /// <param name="out_size">Output Size</param>
        /// <param name="eta">eta</param>
        /// <param name="layer_name">Layer Name</param>
        /// <param name="weights">null or size rand(-1.0,1.0) / sqrt(inputSize) /param>
        /// <param name="biases">null or size 0</param>
        public FullyConnectedLayer(int inputSize, int outputSize, string layerName = "", Vector<double> weights = null, Vector<double> biases = null)
            : base(layerName)
        {
            base.inputSize = inputSize;
            base.outputSize = outputSize;
            inputs = Vector<double>.Build.Dense(base.inputSize);
            outputs = Vector<double>.Build.Dense(base.outputSize);

            
            weightHeight = base.inputSize;
            weightWidth = base.outputSize;
            weightSize = base.inputSize * base.outputSize;
            

            if (weights != null && weights.Count == weightSize) { GenerateWeights(weights); }
            else { var _w_bd = inputSize; GenerateWeights(-1.0 / _w_bd, 1.0 / _w_bd); }
            _dw = Matrix<double>.Build.Dense(weightHeight, weightWidth, 0);
            _pre_dw = Matrix<double>.Build.Dense(weightHeight, weightWidth, 0);

            
            if (biases != null && biases.Count == base.outputSize) { this.biases = biases.Clone(); }
            else { this.biases = Vector<double>.Build.Dense(base.outputSize, 0); }
            _db = Vector<double>.Build.Dense(base.outputSize, 0);

            LayerType = "FullyConnectedLayer";
            GenericsType = activation.Type();
        }

        public override void ForwardPropagation()
        {
            // outputs =  w^T * input
            outputs = weights.TransposeThisAndMultiply(inputs) + biases;
        }

        /// <summary>
        /// Δw
        /// </summary>
        protected Matrix<double> _dw;
        /// <summary>
        ///  Δb
        /// </summary>
        protected Vector<double> _db;
        /// <summary>
        ///  Δw(t-1)
        /// </summary>
        protected Matrix<double> _pre_dw;

        /// <summary>
        /// <para>BP</para>
        /// </summary>
        /// <param name="next_delta">
        /// <para>wdelta_[n+1]= w_[n+1] * δ_[n+1]</para>
        /// </param>
        /// <returns>w_[n] * δ_[n]</returns>
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

    
        public override void WeightUpdate(double eta, double mu, double lambda)
        {
            // Δw(t) = -η∂E/∂w(t) + μΔw(t-1) - ηλw(t)
            var _dw_ = -eta * _dw + mu * _pre_dw - eta * lambda * weights;
            weights = weights + _dw_;

            // b_[n] = b_[n] - ηΔb_[n]
            biases -= eta * _db;

            
            _pre_dw = _dw_.Clone();

            
            _dw.Clear();
            _db.Clear();
        }

        
        public override void GenerateWeights(double lower = -0.1, double upper = 0.1)
        {
            weights = Matrix<double>.Build.Random(weightHeight, weightWidth, new ContinuousUniform(lower, upper));
        }
        
        public override void GenerateWeights(Vector<double> weights) { Weights = weights; }


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
                case "o":
                    _res = new StringBuilder("#Output\n" + outputSize + "\n", outputSize * 8);
                    for (int osz = 0; osz < outputSize; osz++)
                    {
                        _res.Append(activation.f(outputs[osz]) + "\t");
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
