using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ConvNet.Layers;
using MathNet.Numerics.LinearAlgebra;
using ConvNet.Network;
using ConvNet.ElementWises;
using System.IO;
using ConvNet.Utilities;

namespace ConvNet.Tests
{
    class HumanExtraction
    {
        
        private Vector<double>[] trainingInputs;
        private Vector<double>[] trainingOutputs;
        private Vector<double>[] testInputs;
        private Vector<double>[] testOutputs;

        private int trainingNumber;
        private int testNumber;

        private int imageRow;
        private int imageCol;

        private Network<LossFunctions.MSE> Net;
        private Network<LossFunctions.MSE> NetRGB;
        private Network<LossFunctions.MSE> NetGray;

        int[] ConnectionTableRGB1 = new int[]{
                1,1,1, 0,0,0, 0,0,0, 1,1,1, 0,0,0 ,1,1,1, 1,1,1,
                0,0,0, 1,1,1, 0,0,0, 1,1,1, 1,1,1 ,0,0,0, 1,1,1,
                0,0,0, 0,0,0, 1,1,1, 0,0,0, 1,1,1 ,1,1,1, 1,1,1
            };

        int[] ConnectionTableRGB2 = new int[]{
                1,0,1, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,0,1, 1,0,1, 1,0,0, 1,0,0,
                1,0,1, 0,1,0, 0,1,0, 1,1,0, 1,1,0, 0,1,0, 0,1,0, 1,1,0, 1,1,0, 0,1,0, 0,1,0,
                1,0,1, 0,0,1, 0,0,1, 0,1,1, 0,1,1, 0,0,1, 0,0,1, 0,1,1, 0,1,1, 0,0,1, 0,0,1,
                1,1,0, 0,0,0, 0,0,0, 1,0,1, 1,0,1, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,1,0, 1,1,0,
                1,1,0, 0,0,0, 0,0,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,1, 0,1,1,
                1,1,0, 0,0,0, 0,0,0, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 1,0,1, 1,0,1,
                1,1,1, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 1,1,1, 1,1,1, 1,1,1, 1,1,1,
            };

        int[] ConnectionTableGray = new int[]{
                1,1,1,0,0,0,1,1,0,1,
                1,0,0,1,1,0,1,0,1,1,
                0,1,0,1,0,1,1,1,1,1,
                0,0,1,0,1,1,0,1,1,1
            };

        const string OUTPUT_PREFIX = "HumanExtraction\\";

        public HumanExtraction(int imageRow, int imageCol)
        {
            this.imageRow = imageRow;
            this.imageCol = imageCol;

            NetRGB = new Network<LossFunctions.MSE>(
                new ConvolutionalLayer<Activations.Sigmoid>(this.imageRow, this.imageCol, 3, 9, 21, 1, 4, ConnectionTableRGB1, "C1"),
                new PoolingLayer<Poolings.Max>(96, 48, 21, 3, 3, "P2"),
                new ElemWiseLayer<ElementWises.MaxOut>(32, 16, 21, 3, 3, "E3"),
                new ConvolutionalLayer<Activations.Sigmoid>(32, 16, 7, 5, 33, 1, 2, ConnectionTableRGB2, "C4"),
                new PoolingLayer<Poolings.Max>(32, 16, 33, 2, 2, "P5"),
                new ElemWiseLayer<ElementWises.MaxOut>(16, 8, 33, 3, 3, "E6"),
                new ConvolutionalLayer<Activations.Sigmoid>(16, 8, 11, 5, 16, 1, 2, null, "C7"),
                new DropConnectLayer<Activations.Sigmoid>(16 * 8 * 16, 1024, 0.5, "DC8"),
                new FullyConnectedLayer<Activations.Sigmoid>(1024, this.imageRow * this.imageCol, "F9")
            );


            Vector<double> _wei_C1 = Tools.LoadWeightOrBiaseFromFile(OUTPUT_PREFIX + "result\\lr_0.05_weight_C1.txt");
            Vector<double> _wei_C4 = Tools.LoadWeightOrBiaseFromFile(OUTPUT_PREFIX + "result\\lr_0.05_weight_C4.txt");
            Vector<double> _wei_DC6 = Tools.LoadWeightOrBiaseFromFile(OUTPUT_PREFIX + "result\\lr_0.05_weight_DC6.txt");
            Vector<double> _wei_F7 = Tools.LoadWeightOrBiaseFromFile(OUTPUT_PREFIX + "result\\lr_0.05_biase_F7.txt");
            Vector<double> _bis_C1 = Tools.LoadWeightOrBiaseFromFile(OUTPUT_PREFIX + "result\\lr_0.05_biase_C1.txt");
            Vector<double> _bis_C4 = Tools.LoadWeightOrBiaseFromFile(OUTPUT_PREFIX + "result\\lr_0.05_biase_C4.txt");
            Vector<double> _bis_DC6 = Tools.LoadWeightOrBiaseFromFile(OUTPUT_PREFIX + "result\\lr_0.05_biase_DC6.txt");
            Vector<double> _bis_F7 = Tools.LoadWeightOrBiaseFromFile(OUTPUT_PREFIX + "result\\lr_0.05_biase_F7.txt");


            NetGray = new Network<LossFunctions.MSE>(
                new ConvolutionalLayer<Activations.Sigmoid>(this.imageRow, this.imageCol, 1, 9, 8, 1, 4, null, "C1", _wei_C1, _bis_C1),
                new PoolingLayer<Poolings.Max>(96, 48, 8, 3, 3, "P2"),
                new ElemWiseLayer<ElementWises.MaxOut>(32, 16, 8, 2, 2, "E3"),
                new ConvolutionalLayer<Activations.Sigmoid>(32, 16, 4, 5, 10, 1, 2, ConnectionTableGray, "C4", _wei_C4, _bis_C4),
                new PoolingLayer<Poolings.Max>(32, 16, 10, 2, 2, "P5"),
                new DropConnectLayer<Activations.Sigmoid>(16 * 8 * 10, 1156, 0.5, "DC6", _wei_DC6, _bis_DC6),
                new FullyConnectedLayer<Activations.Sigmoid>(1156, this.imageRow * this.imageCol, "F7", _wei_F7, _bis_F7)
            );
        }

        public double _eta;

        int call_back_epochs = 0;
        int call_back_batches = 0;
        StreamWriter sw_epoch;
        StreamWriter sw_batch;

        double pre_error = double.MaxValue;

        public void Start(int net)
        {
            sw_epoch = new StreamWriter(OUTPUT_PREFIX + "result\\lr_" + _eta + "_on_epoch.txt");
            sw_epoch.WriteLine("#epoch\ttrain error\ttrain error per image\ttrain error per pix");
            sw_batch = new StreamWriter(OUTPUT_PREFIX + "result\\lr_" + _eta + "_on_batch.txt");

            Console.WriteLine("datas loading start");
            switch (net)
            {
                case 0:
                    Tools.LoadDataList(OUTPUT_PREFIX + "data\\train_4_in_RGB_wh.txt", out trainingInputs, out trainingOutputs, true);
                    Tools.LoadDataList(OUTPUT_PREFIX + "data\\test_4_in_RGB_wh.txt", out testInputs, out testOutputs, false);
                    Net = NetRGB;
                    break;
                case 1:
                    Tools.LoadDataList(OUTPUT_PREFIX + "data\\train_567_in_L_wh_sc.txt", out trainingInputs, out trainingOutputs, true);
                    Tools.LoadDataList(OUTPUT_PREFIX + "data\\test_567_in_L_wh_sc.txt", out testInputs, out testOutputs, false);
                    Net = NetGray;
                    break;
            }

            trainingNumber = trainingInputs.Length;
            testNumber = testInputs.Length;

            Net.NetworkStructure();
            if (!Net.NetworkCheck()) { return; }

            Net.TrainWithTest(trainingInputs, trainingOutputs, testInputs, testOutputs, 1, 20, 10, _eta, 1.0, 0.0, 0.001, onEpoch, onBatch);

            sw_epoch.Close();
            sw_batch.Close();
        }

        double tr_err = 0.0;

        private bool onEpoch()
        {
            Console.WriteLine("lr:" + Net.eta + "\tmoment:" + Net.mu + "\tweight decay" + Net.lambda);

            Console.WriteLine("Test start");
            Vector<double>[] _ys = new Vector<double>[testInputs.Length];

            
            double ts_err = 0;
            for (int i = 0; i < testInputs.Length; i++)
            {
                _ys[i] = Net.Prediction(testInputs[i]);
                for (int j = 0; j < _ys[i].Count; j++)
                {
                    ts_err += Net.LossFunction.f(_ys[i][j], testOutputs[i][j]);
                }
            }
            ts_err = Math.Sqrt(ts_err);
            var ts_err_per_img = ts_err / testInputs.Length;
            var ts_err_per_pix = ts_err_per_img / (imageRow * imageCol);
            tr_err = Math.Sqrt(tr_err);
            var tr_err_per_img = tr_err / trainingInputs.Length;
            var tr_err_per_pix = tr_err_per_img / (imageRow * imageCol);

            Console.WriteLine(call_back_epochs + "\t" + tr_err + "\t" + tr_err_per_img + "\t" + tr_err_per_pix + "\t" + ts_err + "\t" + ts_err_per_img + "\t" + ts_err_per_pix);
            sw_epoch.WriteLine(call_back_epochs + "\t" + tr_err + "\t" + tr_err_per_img + "\t" + tr_err_per_pix + "\t" + ts_err + "\t" + ts_err_per_img + "\t" + ts_err_per_pix);

            if (ts_err < pre_error)
            {
                
                Net.WriteWeights(OUTPUT_PREFIX + "result\\lr_" + _eta);
                
                Net.WriteBiases(OUTPUT_PREFIX + "result\\lr_" + _eta);

                using (StreamWriter _sw = new StreamWriter(OUTPUT_PREFIX + "result\\lr_" + _eta + "outputs.cnno"))
                {
                    _sw.WriteLine(imageRow + "\t" + imageCol);
                    for (int i = 0; i < testInputs.Length; i++)
                    {
                        foreach (var _v in _ys[i])
                        {
                            _sw.Write((_v * 255) + "\t");
                        }
                        _sw.WriteLine();
                    }
                }
            }

            if (ts_err < pre_error) { pre_error = ts_err; }
            else
            {
                Net.eta = Math.Max(1e-5, Net.eta * 0.5);
                Net.mu = Math.Min(1.0 - 1.0e-5, Net.mu * 1.05);
            }

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

            sw_batch.Flush();
            sw_epoch.Flush();

            tr_err = 0.0;

            return ts_err < 10;
        }

        private bool onBatch(double err)
        {
            tr_err += err;
            sw_batch.WriteLine(err);
            call_back_batches++;
            return true;
        }
    }

}
