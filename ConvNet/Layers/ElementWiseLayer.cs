using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace ConvNet.Layers
{
    class ElemWiseLayer<ElementWiseType> : Layer
               where ElementWiseType : ElementWises.IElementWise, new()
    {
        private Matrix<double>[] inputs;
        private Matrix<double>[] outputs;
        private int inputHeight, inputWidth, inputDepth;
        protected int outputHeight, outputWidth, outputDepth;
        private int elementSize;
        ElementWiseType elementWiseType = new ElementWiseType();

        public ElemWiseLayer(int inputHeight, int inputWidth, int inputDepth,
            int elementSize = 2, int stride = 2, string layerName = "")
            : base(layerName)
        {
            
            this.inputHeight = inputHeight;
            this.inputWidth = inputWidth;
            this.inputDepth = inputDepth;
            inputSize = inputHeight * inputWidth * inputDepth;

            inputs = new Matrix<double>[this.inputDepth];
            for (int i = 0; i < this.inputDepth; i++)
            {
                inputs[i] = Matrix<double>.Build.Dense(this.inputHeight, this.inputWidth, 0);
            }

            this.elementSize = elementSize;

            // 出力
            outputHeight = this.inputHeight;
            outputWidth = this.inputWidth;
            outputDepth = (this.inputDepth - this.elementSize) / stride + 1;
            outputSize = outputHeight * outputWidth * outputDepth;

            outputs = new Matrix<double>[outputDepth];
            for (int i = 0; i < outputDepth; i++)
            {
                outputs[i] = Matrix<double>.Build.Dense(outputHeight, outputWidth, 0);
            }

            LayerType = "ElementWiseLayer";
            GenericsType = elementWiseType.Type();

            base.stride = stride;
        }

        public override Vector<double> Inputs
        {
            set
            {
                if (inputSize != value.Count) { throw new ArgumentException("Size of inputs is different"); }
                inputs = Utilities.Converters.ToMatrices(value, inputDepth, inputHeight, inputWidth);
            }
        }

        public override Vector<double> Outputs
        {
            get { return Utilities.Converters.ToVector(outputs, outputHeight, outputWidth); }
        }

        public override void ForwardPropagation() { ElementWising(); }

        /// <summary>
        /// BP (ElementWises ver.)
        /// </summary>
        /// <param name="next_delta">
        ///  (Σ δ_[l+1] * w_[l+1])
        /// </param>
        /// <returns> (Σ δ_[l] * w_[l])</returns>
        public override Vector<double> BackPropagation(Vector<double> next_delta)
        {
            Matrix<double>[] curt_delta = Utilities.Converters.ToMatrices(next_delta, outputDepth, outputHeight, outputWidth);

            //  wdelta
            Matrix<double>[] prev_delta = new Matrix<double>[inputDepth];
            for (int i = 0; i < inputDepth; i++) { prev_delta[i] = Matrix<double>.Build.Dense(inputHeight, inputWidth, 0); }

            int id = 0;

            for (int od = 0; od < outputDepth; od++)
            {
                var _df_delta = inputs.Skip(id).Take(elementSize);

                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        // φ'(_output)
                        var _df = elementWiseType.df(_df_delta.Select(_ => _[oh, ow]).ToArray());

                        // δ_[j,h,w] = wdelta_[k,h,w] * φ'(_output)
                        for (int es = 0; es < elementSize; es++)
                        {
                            prev_delta[id + es][oh, ow] += _df[es] * curt_delta[od][oh, ow];
                        }
                    }
                }
                id += stride;
            }

            return Utilities.Converters.ToVector(prev_delta, inputHeight, inputWidth);
        }

        /// <summary>
        /// Element wise
        /// </summary>
        /// <returns></returns>
        private void ElementWising()
        {
            int od = 0;
            for (int id = 0; id <= inputDepth - elementSize; id += stride)
            {
                // outputs[od] 
                var elem_inputs = inputs.Skip(id).Take(elementSize).ToArray();

         
                for (int ih = 0; ih < inputHeight; ih++)
                {
                    for (int iw = 0; iw < inputWidth; iw++)
                    {
                        outputs[od][ih, iw] = elementWiseType.f(elem_inputs, ih, iw);
                    }
                }
                od++;
            }
        }

        public override string ToString(string fmt = "o")
        {
            StringBuilder _res;
            switch (fmt)
            {
                case "i":
                    _res = new StringBuilder("#Inputs\n" + inputDepth + "\t" + inputHeight + "\t" + inputWidth + "\n", inputSize * 8);
                    for (int id = 0; id < inputDepth; id++)
                    {
                        for (int ih = 0; ih < inputHeight; ih++)
                        {
                            for (int iw = 0; iw < inputWidth; iw++)
                            {
                                _res.Append(inputs[id][ih, iw] + "\t");
                            }
                            _res.Append("\n");
                        }
                        _res.Append("\n");
                    }
                    break;
                case "o":
                    _res = new StringBuilder("#Output\n" + outputDepth + "\t" + outputHeight + "\t" + outputWidth + "\n", outputSize * 8);
                    for (int od = 0; od < outputDepth; od++)
                    {
                        for (int oh = 0; oh < outputHeight; oh++)
                        {
                            for (int ow = 0; ow < outputWidth; ow++)
                            {
                                _res.Append(outputs[od][oh, ow] + "\t");
                            }
                            _res.Append("\n");
                        }
                        _res.Append("\n");
                    }
                    break;
                case "l":
                    _res = new StringBuilder(
                        "Inputs:" + inputHeight + "x" + inputWidth + "x" + inputDepth + ", " +
                        "Outputs:" + outputHeight + "x" + outputWidth + "x" + outputDepth + ", " +
                        "ElementWiseSize:" + elementSize + ", " + "Stride:" + stride);
                    break;
                default:
                    _res = new StringBuilder("None\n");
                    break;
            }

            return _res.ToString();
        }

    }

}
