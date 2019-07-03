using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ConvNet.Layers;
using MathNet.Numerics.LinearAlgebra;
using System.IO;
using ConvNet.Network;
using ConvNet.Utilities;
using ConvNet.ElementWises;

namespace ConvNet.Tests
{
    class HumanDirection
    {

        private Vector<double>[] trainingInputs;
        private Vector<double>[] trainingOutputs;
        private Vector<double>[] testInputs;
        private Vector<double>[] testOutputs;

        private int trainingNumber;
        private int testNumber;

        private int imageRow;
        private int imageCol;

        
        private Network<LossFunctions.MultiCrossEntropy> NetCNN;
        private Network<LossFunctions.MultiCrossEntropy> NetMLP;

        int[] _cnct_tbl1_rgb = new int[]{
                1,1,1,1, 0,0,0,0, 0,0,0,0, 1,1,1,1, 0,0,0,0, 1,1,1,1, 1,1,1,1,
                0,0,0,0, 1,1,1,1, 0,0,0,0, 1,1,1,1, 1,1,1,1, 0,0,0,0, 1,1,1,1,
                0,0,0,0, 0,0,0,0, 1,1,1,1, 0,0,0,0, 1,1,1,1, 1,1,1,1, 1,1,1,1
            };

        int[] _cnct_tbl2_gray = new int[]{
                1,0,1, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,0,1, 1,0,1, 1,0,0, 1,0,0,
                1,0,1, 0,1,0, 0,1,0, 1,1,0, 1,1,0, 0,1,0, 0,1,0, 1,1,0, 1,1,0, 0,1,0, 0,1,0,
                1,0,1, 0,0,1, 0,0,1, 0,1,1, 0,1,1, 0,0,1, 0,0,1, 0,1,1, 0,1,1, 0,0,1, 0,0,1,
                1,1,0, 0,0,0, 0,0,0, 1,0,1, 1,0,1, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,1,0, 1,1,0,
                1,1,0, 0,0,0, 0,0,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,1, 0,1,1,
                1,1,0, 0,0,0, 0,0,0, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 1,0,1, 1,0,1,
                1,1,1, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 1,1,1, 1,1,1, 1,1,1, 1,1,1,
            };

        public HumanDirection(int imageRow, int imageCol)
        {
            this.imageRow = imageRow;
            this.imageCol = imageCol;

            Console.WriteLine("datas loading start");
            Tools.LoadDataList("HumanDirection\\train_all_in_wh.txt", out trainingInputs, out trainingOutputs, true);
            Tools.LoadDataList("HumanDirection\\test_all_in_wh.txt", out testInputs, out testOutputs, true);

            trainingNumber = trainingInputs.Length;
            testNumber = testInputs.Length;

            var imageDepth = trainingInputs[0].Count / (this.imageRow * this.imageCol);

            NetCNN = new Network<LossFunctions.MultiCrossEntropy>(
                new ConvolutionalLayer<Activations.Sigmoid>(this.imageRow, this.imageCol, imageDepth, 9, 28, 1, 4, _cnct_tbl1_rgb, "C1"),
                new PoolingLayer<Poolings.Max>(96, 48, 28, 2, 2, "P2"),
                new ElemWiseLayer<ElementWises.MaxOut>(48, 24, 28, 4, 4, "E3"),
                new ConvolutionalLayer<Activations.Sigmoid>(48, 24, 7, 5, 33, 1, 2, _cnct_tbl2_gray, "C4"),
                new PoolingLayer<Poolings.Max>(48, 24, 33, 2, 2, "P5"),
                new ElemWiseLayer<ElementWises.MaxOut>(24, 12, 33, 3, 3, "E6"),
                new ConvolutionalLayer<Activations.Sigmoid>(24, 12, 11, 5, 32, 1, 2, null, "C7"),
                new PoolingLayer<Poolings.Max>(24, 12, 32, 2, 2, "P8"),
                new DropOutLayer<Activations.Sigmoid>(12 * 6 * 32, 600, 0.5, "DO9"),
                new FullyConnectedLayer<Activations.Sigmoid>(600, 100, "F10"),
                new SoftmaxLayer(100, 8, "S11")
            );

            NetMLP = new Network<LossFunctions.MultiCrossEntropy>(
                new DropConnectLayer<Activations.Sigmoid>(this.imageRow * this.imageCol * imageDepth, 16 * 8 * 3, 0.5, "L1"),
                new SoftmaxLayer(16 * 8 * 3, 8, "L2")
            );
        }

        Network<LossFunctions.MultiCrossEntropy> _net;

        static int call_back_epochs = 0;
        static int call_back_batches = 0;
        static StreamWriter sw_trerr = new StreamWriter("DIRECTION_train_err.txt");
        static StreamWriter sw_tserr = new StreamWriter("DIRECTION_test_err.txt");
        static StreamWriter sw_netlr = new StreamWriter("DIRECTION_net_lr.txt");
        double pre_acc = 0;
        double tr_err = 0;

        /// <param name="input"></param>
        /// <param name="output"></param>
        /// <returns></returns>
        private bool onEpoch()
        {
            Console.WriteLine(tr_err / trainingNumber);
            tr_err = 0;
            Console.WriteLine("Test start in epoch" + call_back_epochs);
            double _acc = 0;
            int[,] recog_table = new int[8, 8];
            for (int i = 0; i < testInputs.Length; i++)
            {
                var _output = _net.Prediction(testInputs[i]);
                var _om = _output.MaximumIndex();
                var _tom = testOutputs[i].MaximumIndex();

                recog_table[_om, _tom]++;
                if (_om == _tom) { _acc++; }
            }
            
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    Console.Write(recog_table[i, j] + "\t");
                }
                Console.WriteLine();
            }
            
            _acc = _acc / (double)testInputs.Length;
            Console.WriteLine("accuracy : " + _acc);
            sw_tserr.Write(call_back_epochs + "\t" + _acc);
            double each_ts_num = (testInputs.Length / (double)recog_table.GetLength(0));
            for (int i = 0; i < 8; i++)
            {
                sw_tserr.Write("\t" + (recog_table[i, i] / each_ts_num));
            }
            sw_tserr.WriteLine();

            
            Console.WriteLine("lr:" + _net.eta + "\tmoment:" + _net.mu + "\tweight decay" + _net.lambda);
            sw_netlr.WriteLine(_net.eta + "\t" + _net.mu + "\t" + _net.lambda);

            call_back_epochs++;

            
            int[] _idx = Tools.RandomIndex(0, trainingNumber, trainingNumber);
            Vector<double>[] _tmp_train_in = new Vector<double>[trainingNumber];
            Vector<double>[] _tmp_train_out = new Vector<double>[trainingNumber];
            for (int i = 0; i < trainingNumber; i++)
            {
                _tmp_train_in[i] = trainingInputs[_idx[i]].Clone();
                _tmp_train_out[i] = trainingOutputs[_idx[i]].Clone();
            }
            trainingInputs = _tmp_train_in;
            trainingOutputs = _tmp_train_out;

            // パラメータの調整
            if (pre_acc >= _acc)
            {
                _net.eta = Math.Max(1e-5, _net.eta * 0.5);
                _net.mu = Math.Min(1.0 - 1.0e-5, _net.mu * 1.5);
            }
            else { pre_acc = _acc; }

            
            sw_trerr.Flush();
            sw_tserr.Flush();
            sw_netlr.Flush();

            if (_acc > 99.0) { return true; }
            else { return false; }
        }

        private bool onBatch(double err)
        {
            tr_err += err;
            sw_trerr.WriteLine(err);
            call_back_batches++;
            return true;
        }

        public void Start(int net_type = 0)
        {
            switch (net_type)
            {
                case 0:
                    _net = NetMLP;
                    Console.WriteLine(_net.NetworkStructure("d"));
                    if (!_net.NetworkCheck()) { break; }
                    _net.TrainWithTest(trainingInputs, 
                        trainingOutputs, 
                        testInputs, 
                        testOutputs, 
                        1, 
                        epoch: 5000,
                        esp: 1, 
                        learningRatio: 0.05, 
                        momentum: 0.0, 
                        L2RegularizationFactor: 0.001,
                        OnEpoch: onEpoch, 
                        OnBatch: onBatch);
                    break;
                case 1:
                    _net = NetCNN;
                    Console.WriteLine(_net.NetworkStructure("d"));
                    if (!_net.NetworkCheck()) { break; }
                    _net.TrainWithTest(trainingInputs, 
                        trainingOutputs, 
                        testInputs, 
                        testOutputs, 
                        5, 
                        epoch : 20,
                        esp: 1, 
                        learningRatio: 0.05, 
                        momentum: 0.0, 
                        L2RegularizationFactor: 0.0001,
                        OnEpoch: onEpoch, 
                        OnBatch: onBatch);
                    break;
                default:
                    break;
            }
        }
    }

}
