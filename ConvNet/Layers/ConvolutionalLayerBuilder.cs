using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvNet.Layers
{
    class ConvolutionalLayerBuilder<ActivationType> where ActivationType : Activations.IActivation, new()
    {

        private int inputHeight;
        private int inputWidth;
        private int inputDepth;
        private int kernelSize = 3;
        private int outputDepth = 32;
        private int stride = 1;
        private int padding = 0;

        private int[] connectionTable = null;
        private string layerName = "";
        private Vector<double> kernels = null;
        private Vector<double> biases = null;


        public ConvolutionalLayerBuilder()
        {
            this.stride = 1;
            this.padding = 0;
            this.layerName = "";

        }

        public ConvolutionalLayerBuilder(int inputHeight, int inputWidth, int inputDepth, int kernelSize = 3, int outputDepth = 32,
            int stride = 1, int padding = 0, int[] connectionTable = null, string layerName = "",
            Vector<double> kernels = null, Vector<double> biases = null)
        {
            this.inputHeight = inputHeight;
            this.inputWidth = inputWidth;
            this.inputDepth = inputDepth;
            this.kernelSize = kernelSize;
            this.outputDepth = outputDepth;
            this.stride = stride;
            this.padding = padding;
            this.connectionTable = connectionTable;
            this.layerName = layerName;
            this.kernels = kernels;
            this.biases = biases;
        }

        public ConvolutionalLayerBuilder<ActivationType> SetInputHeight(int inputHeight)
        {
            this.inputHeight = inputHeight;
            return this;
        }

        public ConvolutionalLayerBuilder<ActivationType> SetInputWidth(int inputWidth)
        {
            this.inputWidth = inputWidth;
            return this;
        }

        public ConvolutionalLayer<ActivationType> build()
        {
            return new ConvolutionalLayer<ActivationType>(inputHeight, inputWidth, inputDepth, kernelSize, outputDepth,
            stride, padding, connectionTable, layerName, kernels, biases);
        }





    }
}
