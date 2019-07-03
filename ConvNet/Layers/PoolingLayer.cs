using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace ConvNet.Layers
{
    class PoolingLayer<PoolingType> : Layer
     where PoolingType : Poolings.IPooling, new()
    {

        private Matrix<double>[] inputs;
        private Matrix<double>[] outputs;
        private int inputHeight, inputWidth, inputDepth;
        protected int outputHeight, outputWidth, outputDepth;
        private int poolSize;
        PoolingType pooling = new PoolingType();

        public PoolingLayer(
            int inputHeight, int inputWidth, int inputDepth,
            int poolingSize = 2, int stride = 2, string layerName = "")
            : base(layerName)
        {
            // Input
            this.inputHeight = inputHeight; this.inputWidth = inputWidth; this.inputDepth = inputDepth;
            inputSize = inputHeight * inputWidth * inputDepth;
            inputs = new Matrix<double>[this.inputDepth];
            for (int i = 0; i < this.inputDepth; i++)
            {
                inputs[i] = Matrix<double>.Build.Dense(this.inputHeight, this.inputWidth, 0);
            }

            poolSize = poolingSize;

            // Output
            outputHeight = (this.inputHeight - poolSize) / stride + 1;
            outputWidth = (this.inputWidth - poolSize) / stride + 1;
            outputDepth = this.inputDepth;
            outputSize = outputHeight * outputWidth * outputDepth;

            outputs = new Matrix<double>[outputDepth];
            for (int i = 0; i < outputDepth; i++)
            {
                outputs[i] = Matrix<double>.Build.Dense(outputHeight, outputWidth, 0);
            }

            LayerType = "PoolingLayer";
            GenericsType = pooling.Type();

            base.stride = stride;
        }


        public override Vector<double> Inputs
        {
            set
            {
                if (inputSize != value.Count) { throw new ArgumentException("Size of inputs is different"); }
                inputs = Utilities.Converters.ToMatrices(vector: value, d: inputDepth, h: inputHeight, w: inputWidth);
            }
        }

        public override Vector<double> Outputs
        {
            get { return Utilities.Converters.ToVector(outputs, outputHeight, outputWidth); }
        }

        public override void ForwardPropagation() { Pooling(); }

        /// <summary>
        /// BackPropagation (Pooling ver.)
        /// </summary>
        /// <param name="next_delta">
        /// Error Signal from the next layer (Σ δ_[l+1] * w_[l+1])
        /// </param>
        /// <returns>Error signal propagating to the front layer (Σ δ_[l] * w_[l])</returns>
        public override Vector<double> BackPropagation(Vector<double> next_delta)
        {
            // Vector<double>[_outd * _outh * _outw] => Matrix<double>[_outd][_outh,_outw]
            Matrix<double>[] _curt_delta = Utilities.Converters.ToMatrices(next_delta, outputDepth, outputHeight, outputWidth);

            // Error propagating to the front layer
            Matrix<double>[] prev_delta = new Matrix<double>[inputDepth];

            //for (int id = 0; id < inputDepth; id++)
            Parallel.For(0, inputDepth, id =>
            {
                prev_delta[id] = Matrix<double>.Build.Dense(inputHeight, inputWidth, 0);

                int ih = 0;
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    int iw = 0;
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        // (Σ δ * w) * φ'(outputs)
                        var _df_delta = pooling.df(inputs[id].SubMatrix(rowIndex: ih, rowCount: poolSize, columnIndex: iw, columnCount: poolSize)) * _curt_delta[id][oh, ow];

                        // inv_f_delta Add to corresponding pre_delta location
                        for (int u = 0; u < poolSize; u++)
                        {
                            for (int v = 0; v < poolSize; v++)
                            {
                                prev_delta[id][ih + u, iw + v] += _df_delta[u, v];
                            }
                        }
                        iw += stride;
                    }
                    ih += stride;
                }
            });

            return Utilities.Converters.ToVector(prev_delta, inputHeight, inputWidth);
        }

        /// <summary>
        /// <para>Pooling</para>
        /// </summary>
        private void Pooling()
        {
            //for (int od = 0; od < outputDepth; id++)
            Parallel.For(0, outputDepth, od =>
            {
                int ih = 0;
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    int iw = 0;
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        outputs[od][oh, ow] = pooling.f(inputs[od].SubMatrix(rowIndex: ih, rowCount: poolSize, columnIndex: iw, columnCount: poolSize));
                        iw += stride;
                    }
                    ih += stride;
                }
            });
        }

        public override string ToString(string format = "o")
        {
            StringBuilder resultString;
            switch (format)
            {
                case "i":
                    resultString = new StringBuilder("#Inputs\n" + inputDepth + "\t" + inputHeight + "\t" + inputWidth + "\n", inputSize * 8);
                    for (int id = 0; id < inputDepth; id++)
                    {
                        for (int ih = 0; ih < inputHeight; ih++)
                        {
                            for (int iw = 0; iw < inputWidth; iw++)
                            {
                                resultString.Append(inputs[id][ih, iw] + "\t");
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
                                resultString.Append(outputs[od][oh, ow] + "\t");
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
                        "PoolingSize:" + poolSize + "x" + poolSize + ", " + "Stride:" + stride);
                    break;
                default:
                    resultString = new StringBuilder("None\n");
                    break;
            }

            return resultString.ToString();
        }

    }

}
