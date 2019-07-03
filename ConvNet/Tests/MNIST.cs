using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ConvNet.Layers;
using ConvNet.Network;
using MathNet.Numerics.LinearAlgebra;

namespace ConvNet.Tests
{
    class MNIST
    {

        private Vector<double>[] trainingInputs;
        private Vector<double>[] trainingOutputs;
        private Vector<double>[] testInputs;
        private Vector<double>[] testOutputs;

        private int trainingNumber;
        private int testNumber;

        private int imageRow;
        private int imageCol;

        private Network<LossFunctions.MSE> LeNet;
        private Network<LossFunctions.MSE> ConvNetJs;
        private Network<LossFunctions.MultiCrossEntropy> MLP;
        static int call_back_epochs = 0;
        static int call_back_batches = 0;
        static StreamWriter sw = new StreamWriter(Utilities.Tools.GetMNISTPath("MNIST_result.txt"));

        public MNIST()
        {
            trainingNumber = 60000;
            testNumber = 10000;
            imageRow = 28;
            imageCol = 28;

            LoadMNISTImageDatas();
            LoadMNISTLabelDatas();

            LeNet = new Network<LossFunctions.MSE>(
                new ConvolutionalLayer<Activations.Tanh>(inputHeight: 28, inputWidth: 28, inputDepth: 1, kernelSize: 5, outputDepth: 6, stride: 1, padding: 2),
                new PoolingLayer<Poolings.Max>(inputHeight: 28, inputWidth: 28, inputDepth: 6, poolingSize: 2, stride: 2),
                new ConvolutionalLayer<Activations.Tanh>(inputHeight: 14, inputWidth: 14, inputDepth: 6, kernelSize: 5, outputDepth: 16, stride: 1, padding: 0, connectionTable: new int[]{
                            1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,
                            1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,
                            1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1,
                            0,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1,
                            0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1,
                            0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1
                    }),
                new PoolingLayer<Poolings.Max>(inputHeight: 10, inputWidth: 10, inputDepth: 16, poolingSize: 2, stride: 2),
                new ConvolutionalLayer<Activations.Tanh>(inputHeight: 5, inputWidth: 5, inputDepth: 16, kernelSize: 5, outputDepth: 120),
                new FullyConnectedLayer<Activations.Tanh>(120, 84),
                new FullyConnectedLayer<Activations.Sigmoid>(84, 10)
            );

            ConvNetJs = new Network<LossFunctions.MSE>(
                new ConvolutionalLayer<Activations.Sigmoid>(inputHeight: 28, inputWidth: 28, inputDepth: 1, kernelSize: 5, outputDepth: 8, stride: 1, padding: 0),
                new PoolingLayer<Poolings.Max>(inputHeight: 24, inputWidth: 24, inputDepth: 8, poolingSize: 2, stride: 2),
                new ConvolutionalLayer<Activations.Sigmoid>(inputHeight: 12, inputWidth: 12, inputDepth: 8, kernelSize: 5, outputDepth: 16, stride: 1, padding: 2),
                new PoolingLayer<Poolings.Max>(inputHeight: 12, inputWidth: 12, inputDepth: 16, poolingSize: 3, stride: 3),
                new FullyConnectedLayer<Activations.Sigmoid>(4 * 4 * 16, 10)
            );

            MLP = new Network<LossFunctions.MultiCrossEntropy>(
                new FullyConnectedLayer<Activations.ReLU>(imageRow * imageCol, 32),
                new SoftmaxLayer(32, 10)
            );
        }

        /// <summary>
        /// The Main Method to start the neural network
        /// </summary>
        /// <param name="neuralNetworkType">
        /// <para>network type</para>
        /// <para>nueralNetworkType = 0 : MLP</para>
        /// <para>nueralNetworkType = 1 : CNN</para>
        /// </param>
        public void Start(int nueralNetworkType = 0)
        {
            switch (nueralNetworkType)
            {
                case 0:
                    Console.WriteLine(MLP.NetworkStructure("d"));
                    MLP.TrainWithTest(trainingInput: trainingInputs,
                        trainingLabel: trainingOutputs,
                        testInput: testInputs,
                        testOutput: testOutputs,
                        batchSize: 1,
                        epoch: 5,
                        esp: 1,
                        learningRatio: 0.01,
                        momentum: 0.0,
                        L2RegularizationFactor: 0.0001,
                        //OnEpoch: new Func<bool>(() => { MLP.WriteWeights("_epoch_"); return true;}), 
                        OnEpoch: onEpoch,
                        OnBatch: onBatch);
                    //for(int i = 0; i < testNumber; i++)
                    //{
                     //   var testOuptut = MLP.Test(testInputs[i], testOutputs[i]);
                     //   Console.WriteLine("Test Input" + testInputs[i] + " Test output " + testOuptut);
                    //}
                    break;
                case 1:
                    Console.WriteLine(LeNet.NetworkStructure("d"));
                    if (!LeNet.NetworkCheck()) { break; }
                    LeNet.TrainWithTest(trainingInput: trainingInputs, 
                        trainingLabel: trainingOutputs, 
                        testInput: testInputs, 
                        testOutput: testOutputs, 
                        batchSize: 1, 
                        epoch: 200, 
                        esp: 1, 
                        learningRatio: 0.01);
                    break;
                case 2:
                    Console.WriteLine(ConvNetJs.NetworkStructure("d"));
                    if (!ConvNetJs.NetworkCheck()) { break; }
                    ConvNetJs.TrainWithTest(trainingInput: trainingInputs, 
                        trainingLabel: trainingOutputs, 
                        testInput: testInputs, 
                        testOutput: testOutputs, 
                        batchSize: 20, 
                        epoch: 100, 
                        esp: 1, 
                        learningRatio: 0.01, 
                        dec_ratio: 0.85, 
                        momentum: 0.5, 
                        L2RegularizationFactor: 0.0001);
                    break;
                default:
                    break;
            }
        }

        private void LoadMNISTImageDatas()
        {
            // train data
            using (FileStream fs = new FileStream(Utilities.Tools.GetDataPath("MNIST\\train-images.idx3-ubyte"), FileMode.Open, FileAccess.Read))
            {
                byte[] buf = new byte[16];
                fs.Read(buf, 0, 16);
                trainingInputs = new Vector<double>[trainingNumber];
                for (int i = 0; i < trainingNumber; i++)
                {
                    double[] val = new double[imageRow * imageCol];
                    for (int j = 0; j < imageRow * imageCol; j++)
                    {
                        val[j] = fs.ReadByte() / 255.0;
                    }
                    trainingInputs[i] = Vector<double>.Build.Dense(val);
                }
            }

            // test data
            using (FileStream fs = new FileStream(Utilities.Tools.GetDataPath("MNIST\\t10k-images.idx3-ubyte"), FileMode.Open, FileAccess.Read))
            {
                byte[] buf = new byte[16];
                fs.Read(buf, 0, 16);
                testInputs = new Vector<double>[testNumber];
                for (int i = 0; i < testNumber; i++)
                {
                    double[] val = new double[imageRow * imageCol];
                    for (int j = 0; j < imageRow * imageCol; j++)
                    {
                        val[j] = fs.ReadByte() / 255.0;
                    }
                    testInputs[i] = Vector<double>.Build.Dense(val);
                }
            }
        }

        private void LoadMNISTLabelDatas()
        {
            // train labels
            using (FileStream fs = new FileStream(Utilities.Tools.GetDataPath("MNIST\\train-labels.idx1-ubyte"), FileMode.Open, FileAccess.Read))
            {
                byte[] buf = new byte[8];
                fs.Read(buf, 0, 8);
                trainingOutputs = new Vector<double>[trainingNumber];
                for (int i = 0; i < trainingNumber; i++)
                {
                    var label = fs.ReadByte();
                    trainingOutputs[i] = Vector<double>.Build.Dense(10, new Func<int, double>(j => { return j == label ? 1 : 0; }));
                }
            }

            // test labels
            using (FileStream fs = new FileStream(Utilities.Tools.GetDataPath("MNIST\\t10k-labels.idx1-ubyte"), FileMode.Open, FileAccess.Read))
            {
                byte[] buf = new byte[8];
                fs.Read(buf, 0, 8);
                testOutputs = new Vector<double>[testNumber];
                for (int i = 0; i < testNumber; i++)
                {
                    var label = fs.ReadByte();
                    testOutputs[i] = Vector<double>.Build.Dense(10, new Func<int, double>(j => { return j == label ? 1 : 0; }));
                }
            }

        }




        /// <param name="input"></param>
        /// <param name="output"></param>
        /// <returns></returns>
        private bool onEpoch()
        {
            Console.WriteLine("Test start");

            int accuracy = 0;
            int[,] recogTable = new int[10, 10];

            for (int i = 0; i < testInputs.Length; i++)
            {
                var actualOutput = MLP.Prediction(testInputs[i]);
                var actualOutputClassIndex = actualOutput.MaximumIndex();    //Actual Output
                var expectedOutputClassIndex = testOutputs[i].MaximumIndex(); //Expected Output
                 
                recogTable[actualOutputClassIndex, expectedOutputClassIndex]++;
                if (actualOutputClassIndex == expectedOutputClassIndex) { accuracy++; }
            }

            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    Console.Write(recogTable[i, j] + "\t");
                }
                Console.WriteLine();
            }

            Console.WriteLine("accuracy : " + accuracy / (double)testInputs.Length * 100.0);

            sw.WriteLine("#accuracy");
            sw.WriteLine(call_back_epochs + "\t" + accuracy / (double)testInputs.Length * 100.0);
            sw.WriteLine("#accuracy table");
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    sw.Write(recogTable[i, j] + "\t");
                }
                sw.WriteLine();
            }

            call_back_epochs++;

            return false;
        }

        private bool onBatch(double err)
        {
            call_back_batches++;
            return true;
        }
    }

}
