using ConvNet.Layers;
using ConvNet.Networks;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.IO;

namespace ConvNet.Tests
{
    class YoloTest
    {
        private Vector<double>[] trainingInputs;
        private Vector<double>[] trainingOutputs;
        private Vector<double>[] testInputs;
        private Vector<double>[] testOutputs;

        private int trainingNumber;
        private int testNumber;

        private int imageRow;
        private int imageCol;

        private Network<LossFunctions.MSE> YoloNet;
        static int call_back_epochs = 0;
        static int call_back_batches = 0;
        static StreamWriter sw = new StreamWriter(Utilities.Tools.GetMNISTPath("MNIST_result.txt"));


        public YoloTest()
        {
            trainingNumber = 60000;
            testNumber = 10000;
            imageRow = 28;
            imageCol = 28;

            // LoadMNISTImageDatas();
            // LoadMNISTLabelDatas();

            YoloNet = new Network<LossFunctions.MSE>(
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 448, inputWidth: 448, inputDepth: 3, kernelSize: 7, outputDepth: 64, stride: 2, padding: 3, layerName: "Conv-1"),
                new PoolingLayer<Poolings.Max>(inputHeight: 224, inputWidth: 224, inputDepth: 64, poolingSize: 2, stride: 2, layerName: "MaxPool-1"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 112, inputWidth: 112, inputDepth: 64, kernelSize: 3, outputDepth: 192, stride: 1, padding: 0, layerName: "Conv-2"),
                new PoolingLayer<Poolings.Max>(inputHeight: 112, inputWidth: 112, inputDepth: 192, poolingSize: 2, stride: 2, layerName: "MaxPool-2"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 56, inputWidth: 56, inputDepth: 192, kernelSize: 1, outputDepth: 128, layerName: "Conv-3"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 56, inputWidth: 56, inputDepth: 128, kernelSize: 3, outputDepth: 256, layerName: "Conv-4"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 56, inputWidth: 56, inputDepth: 256, kernelSize: 1, outputDepth: 256, layerName: "Conv-5"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 56, inputWidth: 56, inputDepth: 256, kernelSize: 1, outputDepth: 512, layerName: "Conv-6"),
                new PoolingLayer<Poolings.Max>(inputHeight: 56, inputWidth: 56, inputDepth: 512, poolingSize: 2, stride: 2, layerName: "MaxPool-3"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 28, inputWidth: 28, inputDepth: 512, kernelSize: 1, outputDepth: 256, layerName: "Conv-7"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 28, inputWidth: 28, inputDepth: 256, kernelSize: 3, outputDepth: 512, layerName: "Conv-8"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 28, inputWidth: 28, inputDepth: 512, kernelSize: 1, outputDepth: 256, layerName: "Conv-9"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 28, inputWidth: 28, inputDepth: 256, kernelSize: 3, outputDepth: 512, layerName: "Conv-10"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 28, inputWidth: 28, inputDepth: 512, kernelSize: 1, outputDepth: 256, layerName: "Conv-11"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 28, inputWidth: 28, inputDepth: 256, kernelSize: 3, outputDepth: 512, layerName: "Conv-12"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 28, inputWidth: 28, inputDepth: 512, kernelSize: 1, outputDepth: 256, layerName: "Conv-13"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 28, inputWidth: 28, inputDepth: 256, kernelSize: 3, outputDepth: 512, layerName: "Conv-14"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 28, inputWidth: 28, inputDepth: 512, kernelSize: 1, outputDepth: 512, layerName: "Conv-15"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 28, inputWidth: 28, inputDepth: 512, kernelSize: 3, outputDepth: 1024, layerName: "Conv-16"),
                new PoolingLayer<Poolings.Max>(inputHeight: 28, inputWidth: 28, inputDepth: 1024, poolingSize: 2, stride: 2, layerName: "MaxPool-4"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 14, inputWidth: 14, inputDepth: 1024, kernelSize: 1, outputDepth: 512, layerName: "Conv-17"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 14, inputWidth: 14, inputDepth: 512, kernelSize: 3, outputDepth: 1024, layerName: "Conv-18"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 14, inputWidth: 14, inputDepth: 1024, kernelSize: 1, outputDepth: 512, layerName: "Conv-19"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 14, inputWidth: 14, inputDepth: 512, kernelSize: 3, outputDepth: 1024, layerName: "Conv-20"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 14, inputWidth: 14, inputDepth: 1024, kernelSize: 3, outputDepth: 1024, layerName: "Conv-21"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 14, inputWidth: 14, inputDepth: 1024, kernelSize: 3, outputDepth: 1024, stride: 2, layerName: "Conv-22"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 7, inputWidth: 7, inputDepth: 1024, kernelSize: 3, outputDepth: 1024, layerName: "Conv-23"),
                new ConvolutionalLayer<Activations.Leaky>(inputHeight: 7, inputWidth: 7, inputDepth: 1024, kernelSize: 3, outputDepth: 1024, layerName: "Conv-24"),
                // DropoutLayer
                new FullyConnectedLayer<Activations.Tanh>(120, 84, layerName: "FC-1"),
                new FullyConnectedLayer<Activations.Sigmoid>(84, 10, layerName: "FC-2")
            );


        }


        public void run()
        {
            Console.WriteLine(YoloNet.NetworkStructure());
            //YoloNet.TrainWithTest(trainingInputs, trainingOutputs, testInputs, testOutputs);
        }


    }
}
