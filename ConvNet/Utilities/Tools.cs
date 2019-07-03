using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.IO;
using OpenCV = Emgu.CV;

namespace ConvNet.Utilities
{
    static class Tools
    {
        public static MathNet.Numerics.LinearAlgebra.Vector<double> LoadWeightOrBiaseFromFile(string filename)
        {
            using (StreamReader sr = new StreamReader(filename))
            {
                string header = sr.ReadLine();
                if (header != "#Kernels" && header != "#Weights" && header != "#Biases") { throw new FormatException("Read file format error"); }
                int[] _sizes = sr.ReadLine().Split('\t').Select(_ => int.Parse(_)).ToArray();
                int _size = 0;
                if (_sizes.Length == 1) { _size = _sizes[0]; }
                else if (_sizes.Length == 2) { _size = _sizes[0] * _sizes[1]; }
                else { _size = _sizes[0] * _sizes[1] * _sizes[2]; }
                MathNet.Numerics.LinearAlgebra.Vector<double> _weight = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(_size);
                int idx = 0;
                while (!sr.EndOfStream)
                {
                    double _val;
                    string[] _val_str = sr.ReadLine().Split('\t');
                    foreach (var _str in _val_str)
                    {
                        if (double.TryParse(_str, out _val)) { _weight[idx] = _val; idx++; }
                    }
                }
                return _weight;
            }
        }

        /// <summary>
        /// [min,max)間の重複のないランダムな整数列を得る
        /// </summary>
        /// <param name="min">min</param>
        /// <param name="max">max</param>
        /// <param name="size">整数列の長さ</param>
        public static int[] RandomIndex(int min, int max, int size)
        {
            HashSet<int> _rand_idx = new HashSet<int>();
            Random _rand = new Random();
            if (size <= 0 || size > max - min) { size = max - min; }
            while (_rand_idx.Count < size) { _rand_idx.Add(_rand.Next(min, max)); }
            return _rand_idx.ToArray();
        }

        public static bool LoadDataList(string filePath, out MathNet.Numerics.LinearAlgebra.Vector<double>[] input, out MathNet.Numerics.LinearAlgebra.Vector<double>[] output,
            bool shuffle = true, string w_infname = null, string w_outfname = null)
        {
            input = null;
            output = null;

            string[] data_type;
            int[] data_length;
            string[] in_list;
            string[] out_list;

            using (System.IO.StreamReader streamReader = new StreamReader(filePath))
            {
                data_type = streamReader.ReadLine().Split('\t');
                if (data_type.Length != 2) { return false; }

                try { data_length = streamReader.ReadLine().Split('\t').Select(str => int.Parse(str)).ToArray(); }
                catch (Exception) { return false; }

                if (!(data_type[0] == "data" && data_type[1] == "label" || data_type[0] == "data" && data_type[1] == "data"))
                {
                    return false;
                }

                in_list = new string[data_length[0]];
                out_list = new string[data_length[0]];
                input = new MathNet.Numerics.LinearAlgebra.Vector<double>[data_length[0]];
                output = new MathNet.Numerics.LinearAlgebra.Vector<double>[data_length[0]];

                int[] idx;
                // shuffle する場合はランダムなindex生成
                if (shuffle)
                {
                    idx = RandomIndex(0, data_length[0], data_length[0]);
                }
                else
                {
                    idx = Enumerable.Range(0, data_length[0]).ToArray();
                }

                for (int i = 0; !streamReader.EndOfStream; i++)
                {
                    var lists = streamReader.ReadLine().Split('\t');
                    in_list[idx[i]] = lists[0];
                    out_list[idx[i]] = lists[1];
                }
            }

            // input
            for (int i = 0; i < in_list.Length; i++)
            {
                using (StreamReader sr_data = new StreamReader(in_list[i]))
                {
                    input[i] = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.DenseOfEnumerable(sr_data.ReadLine().Split('\t').Select(str => double.Parse(str)));
                }
            }

            // output
            if (data_type[1] == "data")
            {
                for (int i = 0; i < out_list.Length; i++)
                {
                    using (StreamReader sr_data = new StreamReader(out_list[i]))
                    {
                        output[i] = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.DenseOfEnumerable(sr_data.ReadLine().Split('\t').Select(str => double.Parse(str)));
                    }
                }
            }
            else if (data_type[1] == "label")
            {
                for (int i = 0; i < out_list.Length; i++)
                {
                    output[i] = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(data_length[1], (idx) => { return int.Parse(out_list[i]) == idx ? 1 : 0; });
                }
            }

            // mean and std datas
            if (w_infname != null)
            {
                MathNet.Numerics.LinearAlgebra.Vector<double> m;
                MathNet.Numerics.LinearAlgebra.Vector<double> s;
                using (StreamReader sr_w = new StreamReader(w_infname))
                {
                    m = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.DenseOfEnumerable(sr_w.ReadLine().Split('\t').Select(str => double.Parse(str)));
                    s = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.DenseOfEnumerable(sr_w.ReadLine().Split('\t').Select(str => double.Parse(str)));
                }
                for (int i = 0; i < input.Length; i++)
                {
                    input[i] = (input[i] - m) / s.Map<double>((val) => val == 0 ? 1 : 0);
                }
            }
            if (w_outfname != null && data_type[1] == "data")
            {
                MathNet.Numerics.LinearAlgebra.Vector<double> m;
                MathNet.Numerics.LinearAlgebra.Vector<double> s;
                using (StreamReader sr_w = new StreamReader(w_outfname))
                {
                    m = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.DenseOfEnumerable(sr_w.ReadLine().Split('\t').Select(str => double.Parse(str)));
                    s = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.DenseOfEnumerable(sr_w.ReadLine().Split('\t').Select(str => double.Parse(str)));
                }
                for (int i = 0; i < input.Length; i++)
                {
                    output[i] = (output[i] - m) / s.Map<double>((val) => val == 0 ? 1 : 0);
                }
            }

            return true;
        }
        public static bool SaveImage(String path, MathNet.Numerics.LinearAlgebra.Matrix<double>[] matrices)
        {


            MathNet.Numerics.LinearAlgebra.Matrix<double>[] BRGMatrices = Converters.ToBRGMatrix(matrices);
            double[,,] ThreeDMatrix = new double[BRGMatrices[0].RowCount, BRGMatrices[0].ColumnCount, 3];
            Parallel.For(0, BRGMatrices.Length, index =>
            {
                for (int i = 0; i < BRGMatrices[0].RowCount; i++)
                    for (int j = 0; j < BRGMatrices[0].ColumnCount; j++)
                        ThreeDMatrix[i, j, index] = BRGMatrices[index][i, j];

            });
            OpenCV.Image<OpenCV.Structure.Bgr, double> image = new OpenCV.Image<OpenCV.Structure.Bgr, double>(ThreeDMatrix);
            return OpenCV.CvInvoke.Imwrite(path, image, new KeyValuePair<OpenCV.CvEnum.ImwriteFlags, int>(OpenCV.CvEnum.ImwriteFlags.JpegQuality, 100));



        }

        public static IList<String> GetVideoFrames(String fileName)
        {
            IList<String> imageArray = new List<String>();
            String path = Utilities.Tools.GetDataPath(fileName);
            OpenCV.VideoCapture capture = new OpenCV.VideoCapture(path);

            double totalFrames = capture.GetCaptureProperty(OpenCV.CvEnum.CapProp.FrameCount);
            double fps = capture.GetCaptureProperty(OpenCV.CvEnum.CapProp.Fps); //30 
            double frameNumber = 0.0;
           double TimeDiff = 0.5;

            bool Reading = true;

            while (Reading)
            {
                capture.SetCaptureProperty(OpenCV.CvEnum.CapProp.PosFrames, frameNumber);
                OpenCV.Mat frame = capture.QueryFrame();
                if (frame != null)
                {
                    String imagePath = Utilities.Tools.GetDataPath("image-" + DateTime.Now.ToString("yyyyMMddHHmmssfff") + ".jpg");
                  imageArray.Add(imagePath);
                    frame.Save(imagePath);
                }
                else
                {
                    Reading = false;
                }
                frameNumber += (fps * TimeDiff);
            }
            return imageArray;

        }

        public static bool SaveImageColor(String path, MathNet.Numerics.LinearAlgebra.Matrix<double>[] matrices)
        {


            MathNet.Numerics.LinearAlgebra.Matrix<double>[] BRGMatrices = Converters.ToBRGMatrix(matrices);
            double[,,] ThreeDMatrix = new double[BRGMatrices[0].RowCount, BRGMatrices[0].ColumnCount, 3];
            Parallel.For(0, BRGMatrices.Length, index =>
            {
                for (int i = 0; i < BRGMatrices[0].RowCount; i++)
                    for (int j = 0; j < BRGMatrices[0].ColumnCount; j++)
                        ThreeDMatrix[i, j, index] = BRGMatrices[index][i, j];

            });
            
            OpenCV.Image<OpenCV.Structure.Bgr, double> image = new OpenCV.Image<OpenCV.Structure.Bgr, double>(ThreeDMatrix);
            return OpenCV.CvInvoke.Imwrite(path, image, new KeyValuePair<OpenCV.CvEnum.ImwriteFlags, int>(OpenCV.CvEnum.ImwriteFlags.JpegQuality, 100));


        }


        public static MathNet.Numerics.LinearAlgebra.Matrix<double>[] LoadNativeImageMatrix(string path)
        {

            System.Drawing.Bitmap img = new System.Drawing.Bitmap(path);
            MathNet.Numerics.LinearAlgebra.Matrix<double> redMatrix = MathNet.Numerics.LinearAlgebra.Matrix<double>.Build.Dense(img.Width, img.Height);
            MathNet.Numerics.LinearAlgebra.Matrix<double> blueMatrix = MathNet.Numerics.LinearAlgebra.Matrix<double>.Build.Dense(img.Width, img.Height);
            MathNet.Numerics.LinearAlgebra.Matrix<double> greenMatrix = MathNet.Numerics.LinearAlgebra.Matrix<double>.Build.Dense(img.Width, img.Height);
            int x, y;

            // Loop through the images pixels to reset color.
            for (x = 0; x < img.Width; x++)
            {
                for (y = 0; y < img.Height; y++)
                {
                    System.Drawing.Color pixelColor = img.GetPixel(x, y);
                    redMatrix[x, y] = pixelColor.R;
                    blueMatrix[x, y] = pixelColor.B;
                    greenMatrix[x, y] = pixelColor.G;
                }
            }

            Console.WriteLine(img.PixelFormat.ToString());


            return new MathNet.Numerics.LinearAlgebra.Matrix<double>[3] { redMatrix, blueMatrix, greenMatrix };
        }

      

        public static MathNet.Numerics.LinearAlgebra.Matrix<double>[] LoadImageMatrix(string path)
        {


            OpenCV.Mat mat = OpenCV.CvInvoke.Imread(path, OpenCV.CvEnum.ImreadModes.AnyColor);
            OpenCV.Image<OpenCV.Structure.Bgr, double> image = mat.ToImage<OpenCV.Structure.Bgr, double>();

            OpenCV.Image<OpenCV.Structure.Gray, double> blue = image[0];
            OpenCV.Image<OpenCV.Structure.Gray, double> green = image[1];
            OpenCV.Image<OpenCV.Structure.Gray, double> red = image[2];


            OpenCV.Matrix<double> matrixRed = new OpenCV.Matrix<double>(red.Height, red.Width, red.NumberOfChannels);
            OpenCV.Matrix<double> matrixBlue = new OpenCV.Matrix<double>(blue.Height, blue.Width, blue.NumberOfChannels);
            OpenCV.Matrix<double> matrixGreen = new OpenCV.Matrix<double>(green.Height, green.Width, green.NumberOfChannels);



            red.CopyTo(matrixRed);
            blue.CopyTo(matrixBlue);
            green.CopyTo(matrixGreen);

            return new MathNet.Numerics.LinearAlgebra.Matrix<double>[3] { Converters.ToMathNet(matrixRed), Converters.ToMathNet(matrixGreen), Converters.ToMathNet(matrixBlue) };

        }

        public static OpenCV.Matrix<double>[] ConvertImageToOpenCVMatrix(string path)
        {


            OpenCV.Mat mat = OpenCV.CvInvoke.Imread(path, OpenCV.CvEnum.ImreadModes.AnyColor);
            OpenCV.Image<OpenCV.Structure.Bgr, double> image = mat.ToImage<OpenCV.Structure.Bgr, double>();
            
            OpenCV.Image<OpenCV.Structure.Gray, double> blue = image[0];
            OpenCV.Image<OpenCV.Structure.Gray, double> green = image[1];
            OpenCV.Image<OpenCV.Structure.Gray, double> red = image[2];


            OpenCV.Matrix<double> matrixRed = new OpenCV.Matrix<double>(red.Height, red.Width, red.NumberOfChannels);
            OpenCV.Matrix<double> matrixBlue = new OpenCV.Matrix<double>(blue.Height, blue.Width, blue.NumberOfChannels);
            OpenCV.Matrix<double> matrixGreen = new OpenCV.Matrix<double>(green.Height, green.Width, green.NumberOfChannels);



            red.CopyTo(matrixRed);
            blue.CopyTo(matrixBlue);
            green.CopyTo(matrixGreen);

            return new OpenCV.Matrix<double>[3] { matrixRed, matrixGreen, matrixBlue };

        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;
            return Path.Combine(assemblyFolderPath, relativePath);
        }

        public static string GetDataPath(string relativePath)
        {
            string assetsRelativePath = @"../../Data";
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;
            return Path.Combine(assemblyFolderPath, assetsRelativePath, relativePath);
        }

        public static string GetMNISTPath(string relativePath)
        {
            string assetsRelativePath = @"../../Data/MNIST";
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;
            return Path.Combine(assemblyFolderPath, assetsRelativePath, relativePath);
        }
    }
 
 }
