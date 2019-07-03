using ConvNet.Layers;
using ConvNet.LossFunctions;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace ConvNet.Networks
{


    public enum learning_rate_policy
    {
        CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM, SGDR
    }


    public class Network<LossFunctionType> : IDisposable where LossFunctionType : ILossFunction, new()
    {
        private List<Layer> layers;
        private LossFunctionType lossFunction = new LossFunctionType();
        private Stopwatch stopWatch = new Stopwatch();

        public LossFunctionType LossFunction { get { return lossFunction; } }
        public string LossFuncType { get; private set; }
        public string TrainType { get; private set; }

        public double eta;
        public double mu;
        public double lambda;





        /// <summary>
        /// <para>Default constructor</para>
        /// </summary>
        public Network()
        {
            layers = new List<Layer>();
            LossFuncType = lossFunction.Type();
        }
        /// <summary>
        /// constructor
        /// </summary>
        /// <param name="layers">layer</param>
        public Network(params Layer[] layers)
        {
            this.layers = new List<Layer>(layers);
            LossFuncType = lossFunction.Type();
        }

        /// <summary>
        /// Add one layer
        /// </summary>
        /// <param name="layer"></param>
       // public void AddLayer(Layer layer) { layers.Add(layer); }

        public Network<LossFunctionType> AddLayer(Layer layer)
        {

            if (layers.Count == 0)
            {
                layer.Inputs = null;
            }
            else
            {
                var last = layers.Last();
                layer.Inputs = last.Outputs;
            }
            layers.Add(layer);
            return this;
        }
        /// <summary>
        /// Add multiple layers
        /// </summary>
        /// <param name="layers"></param>
        public void AddLayers(params Layer[] layers) { this.layers.AddRange(layers); }



        /// <summary>
        /// <para>learing</para>
        /// <para>batchSize >= trainingInput.Length or batchSize <= 0 : full-batch learning</para>
        /// <para>1 < batchSize < train_input.Length : mini-batch learning</para>
        /// <para>batchSize == 1 : online learning</para>
        ///  <para> If testInput and testOutput are given, the error is calculated with test data each time </ para>
        /// </summary>
        /// <param name="train_input">Input learning Data</param>
        /// <param name="train_teach">Teacher Data</param>
        /// <param name="test_input">Test Data Input</param>
        /// <param name="test_output">Test Data Output</param>
        /// <param name="batch_size">Given size of learning sample</param>
        /// <param name="epoch">Number of Iterations</param>
        /// <param name="esp">allowable error</param>
        /// <param name="learning_ratio">Learning factor</param>
        /// <param name="dec_ratio">Decrease rate of learning coefficient</param>
        /// <param name="momentum">Momentum Factor</param>
        /// <param name="L2RegularizationFactor">L2-regularization factor (or weight decay)</param>
        /// <param name="OnEpoch">Callback called in each epoch</param>
        /// <param name="OnBatch">Callback called in each batch</param>
        public void TrainWithTest(
            Vector<double>[] trainingInput,
            Vector<double>[] trainingLabel,
            Vector<double>[] testInput,
            Vector<double>[] testOutput,
            int batchSize = 20,
            int epoch = 10000,
            double esp = 0.01,
            double learningRatio = 0.05,
            double dec_ratio = 1.0,
            double momentum = 0.0,
            double L2RegularizationFactor = 0.0,
            Func<bool> OnEpoch = null,
            Func<double, bool> OnBatch = null)
        {
            // Initialized because the event is null
            if (OnEpoch == null) { OnEpoch = () => false; }
            if (OnBatch == null) { OnBatch = (val) => false; }

            eta = learningRatio * Math.Sqrt(batchSize);
            mu = momentum;
            lambda = L2RegularizationFactor;

            Console.WriteLine("Training start");
            stopWatch.Start();

            // Make Learning Full Batch
            if (batchSize > trainingInput.Length || batchSize <= 0) { batchSize = trainingInput.Length; }

            // disp count
            int _disp_cnt = trainingInput.Length / 50;

            for (int t = 0; t < epoch; t++)
            {
                // Training Data Error
                double trainingError = 0.0;

                //Learning
                if (batchSize == 1)  // Online Learning
                {
                    for (int i = 0; i < trainingInput.Length; i++)
                    {
                        if (i % _disp_cnt == 0) { Console.Write("*"); }
                        trainingError = TrainOnline(trainingInput[i], trainingLabel[i]);
                        OnBatch(trainingError);
                    }
                }
                else if (batchSize == trainingInput.Length) // Full-batch Learning
                {
                    trainingError = TrainBatch(trainingInput, trainingLabel);
                    OnBatch(trainingError / batchSize);
                }
                else // Mini-batch Learning
                {
                    for (int i = 0; i < trainingInput.Length; i += batchSize)
                    {
                        if (i % _disp_cnt == 0) { Console.Write("*"); }
                        int _bs = (i + batchSize) > trainingInput.Length ? (i + batchSize - trainingInput.Length) : batchSize;
                        trainingError = TrainBatch(trainingInput.Skip(i).Take(_bs).ToArray(), trainingLabel.Skip(i).Take(_bs).ToArray());
                        OnBatch(trainingError / batchSize);
                    }
                }
                Console.WriteLine();
                // Experiment with test data
                if (OnEpoch()) { break; }
            }
            Console.WriteLine("Elapsed Time : " + stopWatch.ElapsedMilliseconds / 1000.0 + " seconds");
        }

        //////////////////////////////////////////////////////////////////////////////////////////
        // Training related methods

        /// <summary>
        /// Online learning
        /// </summary>
        /// <param name="t_in">Learning data input</param>
        /// <param name="t_te">Teacher Data</param>
        /// <returns></returns>
        private double TrainOnline(Vector<double> trainingInput, Vector<double> teacherData)
        {
            // Calculate Output of training data
            var y = ForwardPropagation(trainingInput);

            // Output Layer Error
            double error = 0.0;

            // Differentiation of output layer error δ_o
            var dError = y.Clone();
            for (int i = 0; i < dError.Count; i++)
            {
                error += lossFunction.f(y[i], teacherData[i]);
                dError[i] = lossFunction.df(dError[i], teacherData[i]);
            }

            // Check the error and update the wieghts if it is out of tolerance
            for (int i = layers.Count - 1; i >= 0; i--)
            {
                dError = layers[i].BackPropagation(dError.Clone());
                // Update the weights
                layers[i].WeightUpdate(eta, mu, lambda);
            }

            return error;
        }

        /// <summary>
        /// Batch or Mini-batch learning
        /// </summary>
        /// <param name="t_in">Training Data</param>
        /// <param name="t_te">Training Label</param>
        /// <returns></returns>
        private double TrainBatch(Vector<double>[] trainingInput, Vector<double>[] teacherData)
        {
            // Output Layer Error 
            double error = 0.0;

            for (int i = 0; i < trainingInput.Length; i++)
            {
                // Calculate Output of training data
                var y = ForwardPropagation(trainingInput[i]);

                // Differentiation of output layer error δ_o
                var dError = y.Clone();
                for (int j = 0; j < dError.Count; j++)
                {
                    error += lossFunction.f(y[j], teacherData[i][j]);
                    dError[j] = lossFunction.df(dError[j], teacherData[i][j]);
                }
                // Back-Propagate
                for (int j = layers.Count - 1; j >= 0; j--)
                {
                    dError = layers[j].BackPropagation(dError.Clone());
                }
            }
            //Update Weights
            for (int i = 0; i < layers.Count; i++) { layers[i].WeightUpdate(eta, mu, lambda); }

            return error;
        }

        /// <summary>
        /// Feed Forward calculation
        /// </summary>
        /// <param name="t_in">Learning Data Input</param>
        /// <returns>出力</returns>
        private Vector<double> ForwardPropagation(Vector<double> trainingInput)
        {
            layers[0].Inputs = trainingInput;
            for (int i = 0; i < layers.Count; i++)
            {
                layers[i].ForwardPropagation();
                if (i == layers.Count - 1) { break; }
                layers[i + 1].Inputs = layers[i].Outputs;
            }
            return layers.Last().Outputs;
        }

        //
        ///////////////////////////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////////////////////////
        // Test Related Methods

        /// <summary>
        /// <para>Test Provisional</para>
        /// <para>Input data, output, error</para>
        /// </summary>
        /// <param name="t_input">Training Input Data</param>
        /// <param name="t_teach">Teacher/Label Data</param>
        public Vector<double> Test(Vector<double> t_input, Vector<double> t_teach)
        {
            Console.WriteLine("Test start");

            double E = 0.0;
            var y = Prediction(t_input);
            for (int i = 0; i < y.Count; i++)
            {
                E += lossFunction.f(y[i], t_teach[i]);
            }
            Console.Write("Input:\t");
            foreach (var _ in t_input) { Console.Write(_ + " "); }
            Console.WriteLine();
            Console.Write("Output:\t");
            foreach (var _ in y) { Console.Write(_ + " "); }
            Console.WriteLine();
            Console.WriteLine("Error:\t" + E);

            return y.Clone();
        }

        /// <summary>
        /// Test Specific
        /// </summary>
        /// <param name="test_input"></param>
        /// <param name="test_output"></param>
        /// <returns></returns>
        private double Tests(Vector<double>[] test_input, Vector<double>[] test_output)
        {
            Console.WriteLine("Test start");

            int _acc = 0;
            int[,] recog_table = new int[10, 10];

            for (int i = 0; i < test_input.Length; i++)
            {
                var _output = Prediction(test_input[i]);
                var _om = _output.MaximumIndex();
                var _tom = test_output[i].MaximumIndex();

                recog_table[_om, _tom]++;
                if (_om == _tom) { _acc++; }
            }

            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    Console.Write(recog_table[i, j] + "\t");
                }
                Console.WriteLine();
            }

            Console.WriteLine("accuracy : " + _acc / (double)test_input.Length * 100.0);
            return _acc / (double)test_input.Length * 100.0;
        }


        /// <summary>
        /// Feed Forward Computation
        /// </summary>
        /// <param name="t_in"></param>
        /// <returns></returns>
        public Vector<double> Prediction(Vector<double> t_in)
        {
            layers[0].Inputs = t_in;
            for (int i = 0; i < layers.Count; i++)
            {
                layers[i].ForwardPropagation();
                if (i == layers.Count - 1) { break; }
                layers[i + 1].Inputs = layers[i].PredictOutputs;
            }
            return layers.Last().PredictOutputs;
        }

        //
        //////////////////////////////////////////////////////////////////////////////////////////

        /// <summary>
        /// <para>Check for correctness of the Network Structure</para>
        /// <para>Size between input and output</para>
        /// </summary>
        /// <returns></returns>
        public bool NetworkCheck()
        {
            bool flag = true;
            for (int i = 0; i < layers.Count - 1; i++)
            {
                if (!layers[i + 1].CheckSize(layers[i].Outputs.Count))
                {
                    if (layers[i].LayerName == "" || layers[i + 1].LayerName == "")
                    {
                        Console.WriteLine("Network structure is error." + i + " and " + (i + 1) + " layer.");
                    }
                    else
                    {
                        Console.WriteLine("Network structure is error." + layers[i].LayerName + " and " + layers[i + 1].LayerName + " layer.");
                    }
                    flag = false;
                }
            }
            return flag;
        }

        /// <summary>
        /// <para>Network structure output</para>
        /// <para>[Layer No],[Layer Type],[Layer Option]</para>
        /// <para>Layer Option:ActivationType, PoolingType, ElementWizeType etc.</para>
        /// </summary>
        /// <returns></returns>
        public string NetworkStructure(string fmt = "d")
        {
            string structure = "";

            switch (fmt)
            {
                case "d":
                    for (int i = 0; i < layers.Count; i++)
                    {
                        structure += (i + 1) + ", " + layers[i].LayerType + ", " + layers[i].GenericsType + ", " + layers[i].ToString("l") + "\n";
                    }
                    structure += (layers.Count + 1) + ", OutputLayer, " + LossFuncType + "\n";
                    break;
                default:
                    for (int i = 0; i < layers.Count; i++)
                    {
                        structure += (i + 1) + ", " + layers[i].LayerType + "\n";
                    }
                    structure += (layers.Count + 1) + ", OutputLayer" + "\n";
                    break;
            }

            return structure;
        }

        public bool WriteWeights(string prefix = "")
        {
            foreach (var _l in layers)
            {
                using (System.IO.StreamWriter sw = new System.IO.StreamWriter(Utilities.Tools.GetMNISTPath(prefix + "_weight_" + _l.LayerName + ".txt")))
                {
                    sw.WriteLine(_l.ToString("w"));
                }
            }

            return true;
        }

        public bool WriteBiases(string prefix = "")
        {
            foreach (var _l in layers)
            {
                using (System.IO.StreamWriter sw = new System.IO.StreamWriter(prefix + "_biase_" + _l.LayerName + ".txt"))
                {
                    sw.WriteLine(_l.ToString("b"));
                }
            }

            return true;
        }

        public void Dispose() { layers.Clear(); }

    }

}
