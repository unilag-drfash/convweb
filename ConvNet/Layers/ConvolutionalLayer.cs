using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace ConvNet.Layers
{
    /// </summary>
    class ConvolutionalLayer<ActivationType> : Layer
        where ActivationType : Activations.IActivation, new()
    {

        private int inputHeight, inputWidth, inputDepth;
        protected int inputSizeWithPadding, inputHeightWithPadding, inputWidthWithPadding;
        protected Matrix<double>[] inputs;
        public override Vector<double> Inputs
        {
            set
            {
                if (inputSize != value.Count) { throw new ArgumentException("Size of inputs is different"); }
                int index = 0;
                for (int id = 0; id < inputDepth; id++)
                {
                    for (int ih = 0; ih < inputHeight; ih++)
                    {
                        for (int iw = 0; iw < inputWidth; iw++)
                        {
                            inputs[id][ih + padding, iw + padding] = value[index];
                            index++;
                        }
                    }
                }
            }
        }

        protected int outputHeight, outputWidth, outputDepth;
        protected Matrix<double>[] outputs;
        public override Vector<double> Outputs
        {
            get
            {
                return Utilities.Converters.ToVector(outputs.Select(x => x.Map(activation.f)).ToArray(), outputHeight, outputWidth);
            }
        }



        /// <summary>
        /// <para>Weights and Filter Related Parameters : weight[height,width,depth,size]</para>
        /// <para>height	: Rows</para>
        /// <para>width	: Columns</para>
        /// <para>depth	: Depth</para>
        /// <para>size	: Total Number</para>
        /// </summary>
        protected int weightHeight, weightWidth, weightDepth, weightSize;

        /// <summary>
        /// Convolution Filter or Kernels
        /// </summary>
        protected Matrix<double>[] kernels;

        /// <summary>
        /// Weight ）
        /// </summary>
        public override Vector<double> Weights
        {
            get { return Utilities.Converters.ToVector(kernels, weightHeight, weightWidth); }
            protected set
            {
                if (weightSize != value.Count) { throw new ArgumentException("Size of kernels are different"); }
                kernels = Utilities.Converters.ToMatrices(vector: value, d: weightDepth, h: weightHeight, w: weightWidth);
            }
        }

        /// <summary>
        /// bias
        /// </summary>
        protected Vector<double> biases;

        /// <summary>
        /// bias
        /// </summary>
        public override Vector<double> Biases
        {
            get
            {
                return biases.Clone();
            }
            protected set
            {
                if (outputDepth != value.Count) { throw new ArgumentException("Size of biases are different"); }
                biases = value.Clone();
            }
        }
        
        /// <summary>
        /// Activation Function
        /// </summary>
        protected ActivationType activation = new ActivationType();

        /// <summary>
        /// padding
        /// </summary>
        protected int padding;

        /// <summary>
        /// connectionTable
        /// </summary>
        protected int[,] connectionTable;

        /// <summary>
        /// Convolutional Layer
        /// </summary>
        /// <param name="inputHeight">Input Height</param>
        /// <param name="inputWidth">Input Width</param>
        /// <param name="inputDepth">Input Depth</param>
        /// <param name="kernelSize">Convolution Kernel Size</param>
        /// <param name="outputDepth">Output Depth</param>
        /// <param name="stride">Stride(Kernel Travel Amount)</param>
        /// <param name="padding">padding</param>
        /// <param name="connectionTable">connectionTable[inputDepth,outputDepth] = 0 or 1，null or all if the size is different</param>
        /// <param name="eta">Learning Coeffiecent</param>
        /// <param name="layerName">Layer Name</param>
        /// <param name="kernels"><para>Intial Convolutional Kernel or Filter: null or if the size is different, initialize with rand(-1.0,1.0) / sqrt(inputDepth * kernelSize^2)</para>
        /// </param>
        /// <param name = "biases"> Initial bias, null or initialize to 0 if size is different </ param>
        public ConvolutionalLayer(
            int inputHeight, int inputWidth, int inputDepth, int kernelSize = 3, int outputDepth = 32,
            int stride = 1, int padding = 0, int[] connectionTable = null, string layerName = "",
            Vector<double> kernels = null, Vector<double> biases = null)
            : base(layerName)
        {
            // Input without padding
            this.inputHeight = inputHeight; this.inputWidth = inputWidth; this.inputDepth = inputDepth;
            this.inputSize = inputHeight * inputWidth * inputDepth;

            // Input with padding
            this.inputHeightWithPadding = inputHeight + padding * 2;
            this.inputWidthWithPadding = inputWidth + padding * 2;
            this.inputSizeWithPadding = inputHeightWithPadding * inputWidthWithPadding * this.inputDepth;

            // Intiliazing input considering padding
            inputs = new Matrix<double>[this.inputDepth];
            for (int id = 0; id < this.inputDepth; id++)
            {
                inputs[id] = Matrix<double>.Build.Dense(inputHeightWithPadding, inputWidthWithPadding, 0);
            }

            // Convolution Kernel
            // For depth access id * outputDepth + od
            // id : input depth loop counter
            // od : oututput depth loop counter
            weightHeight = weightWidth = kernelSize;
            weightDepth = inputDepth * outputDepth;
            weightSize = kernelSize * kernelSize * weightDepth;

            // Output
            outputHeight = (inputHeightWithPadding - weightHeight) / stride + 1;
            outputWidth = (inputWidthWithPadding - weightWidth) / stride + 1;
            this.outputDepth = outputDepth;
            outputSize = outputHeight * outputWidth * this.outputDepth;


            LayerType = "ConvolutionalLayer";
            GenericsType = activation.Type();

            base.stride = stride;
            this.padding = padding;

            // Output Intitialization
            outputs = new Matrix<double>[this.outputDepth];
            for (int od = 0; od < this.outputDepth; od++)
            {
                outputs[od] = Matrix<double>.Build.Dense(outputHeight, outputWidth, 0);
            }

            // Connection table initialization
            this.connectionTable = new int[inputDepth, outputDepth];
            if (connectionTable != null && connectionTable.Length == weightDepth)
            {
                for (int id = 0; id < inputDepth; id++)
                {
                    for (int od = 0; od < outputDepth; od++)
                    {
                        this.connectionTable[id, od] = connectionTable[id * outputDepth + od];
                    }
                }
            }
            else
            {
                for (int id = 0; id < inputDepth; id++)
                {
                    for (int od = 0; od < outputDepth; od++)
                    {
                        this.connectionTable[id, od] = 1;
                    }
                }
            }

            // Convolution Kernel or Filter Initialization
            // For Depth Access:  id * outputDepth + od
            // id : Input Depth Loop Counter
            // od : Output Depth Loop Counter
            this.kernels = new Matrix<double>[weightDepth];
            if (kernels != null && kernels.Count == weightSize) { GenerateWeights(kernels); }
            else
            {
                //Initialize Random Weights
                var w_bd = Math.Sqrt(this.inputDepth * weightHeight * weightWidth);
                GenerateWeights(-1.0 / w_bd, 1.0 / w_bd);
            }

            // Bias Initialization
            if (biases != null && biases.Count == outputDepth) { this.biases = biases.Clone(); }
            else { this.biases = Vector<double>.Build.Dense(outputDepth, 0); }

            // Initialize WeightUpdate for filter(kernel) vector Δw
            deltaWeight = new Matrix<double>[weightDepth];
            previousDeltaWeight = new Matrix<double>[weightDepth];
            for (int wd = 0; wd < weightDepth; wd++)
            {
                deltaWeight[wd] = Matrix<double>.Build.Dense(weightHeight, weightWidth, 0);
                previousDeltaWeight[wd] = Matrix<double>.Build.Dense(weightHeight, weightWidth, 0);
            }
            // Initialize Bias update vector Δb
            deltaBias = Vector<double>.Build.Dense(outputDepth, 0);

        }

        /// <summary>
        /// Generate convolution kernel with uniform random numbers
        /// </summary>
        public override void GenerateWeights(double lower = -0.1, double upper = 0.1)
        {
            for (int id = 0; id < inputDepth; id++)
            {
                for (int od = 0; od < outputDepth; od++)
                {
                    int wd = id * outputDepth + od;
                    if (connectionTable[id, od] == 0)
                    {
                        kernels[wd] = Matrix<double>.Build.Dense(weightHeight, weightWidth, 0);
                    }
                    else
                    {
                        kernels[wd] = Matrix<double>.Build.Random(weightHeight, weightWidth, new ContinuousUniform(lower, upper));
                    }
                }
            }
        }
        /// <summary>
        /// Set Weights with arbitary weights
        /// </summary>
        /// <param name="filters"></param>
        /// <returns></returns>
        public override void GenerateWeights(Vector<double> weights)
        {
            Weights = weights;
        }

        /// <summary>
        /// Convolution
        /// </summary>
        /// <returns>Convolution</returns>
        public override void ForwardPropagation() { Convolution(); }

        /// <summary>
        /// deltaWeight Δw
        /// </summary>
        protected Matrix<double>[] deltaWeight;
        /// <summary>
        /// deltaBias Δb
        /// </summary>
        protected Vector<double> deltaBias;

        /// <summary>
        /// <para>Delta Weight for Previous epoch Δw(t-1)</para>
        /// <para>Used in Weight decay and momentum</para>
        /// </summary>
        protected Matrix<double>[] previousDeltaWeight;

        /// <summary>
        /// Error Back Propagation(Conv Layer ver.)
        /// </summary>
        /// <param name="next_delta">
        /// Error Signal from the next layer (Σ δ_[l+1] * w_[l+1])
        /// </param>
        /// <returns>Error Signal propagating to the previous layer (Σ δ_[l] * w_[l])</returns>
        public override Vector<double> BackPropagation(Vector<double> nextDelta)
        {
            // Vector<double>[_outd * _outh * _outw] => Matrix<double>[_outd][_outh,_outw]
            Matrix<double>[] _curt_delta = Utilities.Converters.ToMatrices(nextDelta, outputDepth, outputHeight, outputWidth);

            //Calculation of change in Weight Δw of each kernel(filter)
            // 1. δ[l,j] = next_delta_[k] * φ'(outputs[l,j])
            // 2. Δw[l,j] = Σ δ[l,j] * output[l-1,j] = Σ δ[l,j] * inputs_[l,j]
            //for (int od = 0; od < outputDepth; od++)
            Parallel.For(0, outputDepth, od =>
            {
                // Change In Bias Amount Δb = Σ δ[l,j]
                double dbSum = 0;

                int _ih = 0;
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    int _iw = 0;
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        // 1. δ_[l,j] = next_delta_[k] * φ'(_outputs_[l,j])
                        _curt_delta[od][oh, ow] *= activation.df(outputs[od][oh, ow]);

                        // Δb = Σ δ_[l,j]
                        dbSum += _curt_delta[od][oh, ow];

                        for (int id = 0; id < inputDepth; id++)
                        {
                            // connectionTable
                            if (connectionTable[id, od] == 1)
                            {
                                // 2.  Δw_[l,j] = Σ δ_[l,j] * _output_[l-1,j] = Σ δ_[l,j] * _inputs_[l,j]
                                deltaWeight[od] += _curt_delta[od][oh, ow] * inputs[id].SubMatrix(_ih, weightHeight, _iw, weightWidth);
                            }
                        }
                        _iw += stride;
                    }
                    _ih += stride;
                }

                // Change in Bias Δb = Σ δ_[l,j]
                deltaBias[od] += dbSum;
            });

            //Calculation of(Σ δ[l] * w[l]) used in the error calculation of the previous layer
            // Error signal actually passed to previous layer
            Matrix<double>[] prev_delta = new Matrix<double>[inputDepth];

            //for (int id = 0; id < inputDepth; id++)
            Parallel.For(0, inputDepth, id =>
            {
                prev_delta[id] = Matrix<double>.Build.Dense(inputHeight, inputWidth);

                // calculation with padding
                var tmpDelta = Matrix<double>.Build.Dense(inputHeightWithPadding, inputWidthWithPadding);

                int _ih = 0;
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    int _iw = 0;
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        // curt_delta[l,j] = next_delta_[k] * φ'(_outputs_[l,j])
                        // (Σ δ_[l] * w_[l]) = (Σ curt_delta[l,j] * kernel_[l])
                        // _wo the window size
                        var _wdelta = Matrix<double>.Build.Dense(weightHeight, weightWidth, 0);
                        for (int od = 0; od < outputDepth; od++)
                        {
                            if (connectionTable[id, od] == 1)
                            {
                                _wdelta += kernels[id * outputDepth + od] * _curt_delta[od][oh, ow];
                            }
                        }

                        // Add _wdelta_ [l] to the corresponding prev_delta_padd
                        for (int u = 0; u < weightHeight; u++)
                        {
                            for (int v = 0; v < weightWidth; v++)
                            {
                                tmpDelta[_ih + u, _iw + v] += _wdelta[u, v];
                            }
                        }
                        _iw += stride;
                    }
                    _ih += stride;
                }

                // leave padding out
                prev_delta[id] = tmpDelta.SubMatrix(padding, inputHeight, padding, inputWidth);
            });

            // Error signal transmission to front layer
            return Utilities.Converters.ToVector(prev_delta, inputHeight, inputWidth);
        }

        /// <summary>
        /// Update Weight
        /// </summary>
        public override void WeightUpdate(double eta, double mu, double lambda)
        {
            for (int wd = 0; wd < weightDepth; wd++)
            {
                // Δw(t) = -η∂E/∂w(t) + μΔw(t-1) - ηλw(t)
                var dwdt = -eta * deltaWeight[wd] + mu * previousDeltaWeight[wd] - eta * lambda * kernels[wd];
                // w(t) = w(t-1) + Δw(t)
                kernels[wd] = kernels[wd] + dwdt;

                // Keep Previous Delta Weight
                previousDeltaWeight[wd] = dwdt.Clone();

                deltaWeight[wd].Clear();
            }
            // b(t+1) = b(t) - ηΔb(t+1)
            biases -= eta * deltaBias;

            deltaBias.Clear();
        }
        /// <summary>
        /// <para>Convolution</para>
        /// <para>outputs[k] = Σ_{j∈L_inputs} inputs[j] * kernel[k]</para>
        /// </summary>
        protected virtual void Convolution()
        {
            //for (int od = 0; od < outputDepth; od++)
            Parallel.For(0, outputDepth, od =>
            {
                var tmpOutput = Matrix<double>.Build.Dense(outputHeight, outputWidth, 0);

                for (int id = 0; id < inputDepth; id++)
                {
                    // is there connection or not
                    if (connectionTable[id, od] == 1)
                    {
                        int ih = 0;
                        for (int oh = 0; oh < outputHeight; oh++)
                        {
                            int iw = 0;
                            for (int ow = 0; ow < outputWidth; ow++)
                            {
                                tmpOutput[oh, ow] += inputs[id].SubMatrix(rowIndex: ih, 
                                                                          rowCount: weightHeight, 
                                                                          columnIndex: iw, 
                                                                          columnCount: weightWidth)
                                                                .PointwiseMultiply(kernels[id * outputDepth + od])
                                                                .Enumerate()
                                                                .Sum();
                                iw += stride;
                            }
                            ih += stride;
                        }
                    }
                }
                outputs[od] = tmpOutput + biases[od];
            });
        }

        public override string ToString(string format = "o")
        {
            StringBuilder resultString;
            switch (format)
            {
                case "b":
                    resultString = new StringBuilder("#Biases\n" + biases.Count + "\n", biases.Count * 8);
                    for (int i = 0; i < biases.Count; i++)
                    {
                        resultString.Append(biases[i] + "\t");
                    }
                    break;
                case "w":
                    resultString = new StringBuilder("#Kernels\n" + weightDepth + "\t" + weightHeight + "\t" + weightWidth + "\n", weightSize * 8);
                    for (int wd = 0; wd < weightDepth; wd++)
                    {
                        for (int wh = 0; wh < weightHeight; wh++)
                        {
                            for (int ww = 0; ww < weightWidth; ww++)
                            {
                                resultString.Append(kernels[wd][wh, ww] + "\t");
                            }
                            resultString.Append("\n");
                        }
                        resultString.Append("\n");
                    }
                    break;
                case "i":
                    resultString = new StringBuilder("#Inputs\n" + inputDepth + "\t" + inputHeight + "\t" + inputWidth + "\n", inputSize * 8);
                    for (int id = 0; id < inputDepth; id++)
                    {
                        for (int ih = 0; ih < inputHeight; ih++)
                        {
                            for (int iw = 0; iw < inputWidth; iw++)
                            {
                                resultString.Append(inputs[id][padding + ih, padding + iw] + "\t");
                            }
                            resultString.Append("\n");
                        }
                        resultString.Append("\n");
                    }
                    break;
                case "p":
                    resultString = new StringBuilder(
                        "#Inputs with padding:" + padding + "\n" +
                        inputDepth + "\t" + inputHeightWithPadding + "\t" + inputWidthWithPadding + "\n", inputSizeWithPadding * 8);
                    for (int id = 0; id < inputDepth; id++)
                    {
                        for (int ihp = 0; ihp < inputHeightWithPadding; ihp++)
                        {
                            for (int iwp = 0; iwp < inputWidthWithPadding; iwp++)
                            {
                                resultString.Append(inputs[id][ihp, iwp] + "\t");
                            }
                            resultString.Append("\n");
                        }
                        resultString.Append("\n");
                    }
                    break;
                case "o":
                    resultString = new StringBuilder("#Output\n" + outputDepth + "\t" + outputHeight + "\t" + outputWidth + "\n", outputSize * 8);
                    for (int od = 0; od < outputDepth; od++)
                    {
                        for (int oh = 0; oh < outputHeight; oh++)
                        {
                            for (int ow = 0; ow < outputWidth; ow++)
                            {
                                resultString.Append(activation.f(outputs[od][oh, ow]) + "\t");
                            }
                            resultString.Append("\n");
                        }
                        resultString.Append("\n");
                    }
                    break;
                case "l":
                    resultString = new StringBuilder(
                        "Inputs:" + inputHeight + "x" + inputWidth + "x" + inputDepth + ", " +
                        "Outputs:" + outputHeight + "x" + outputWidth + "x" + outputDepth + ", " +
                        "Kernels:" + inputDepth + "x" + outputDepth + "x" + weightHeight + "x" + weightWidth + ", " +
                        "Biases:" + outputDepth + "," + "Stride:" + stride + ", " + "Padding:" + padding);
                    break;
                default:
                    resultString = new StringBuilder("None\n");
                    break;
            }

            return resultString.ToString();
        }
    }
}
