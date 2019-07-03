using ConvNet.Utilities;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvNet
{
    class Program
    {
        static void Main(string[] args)
        {
            
            Console.WriteLine("Starting");
            // Tests.MNIST mnist = new Tests.MNIST();
            //  mnist.Start();
            //  Tests.YoloTest yolo = new Tests.YoloTest();
            // yolo.run();
            //diff();
            GetVideoFrames();
            Console.WriteLine("Enter a key");
            Console.Read();
        }

        public static void GetVideoFrames()
        {
            IList<String> frames = Utilities.Tools.GetVideoFrames("video.mp4");
            int noOfPictures = frames.Count;
            Console.WriteLine($"Counting the number of frames {noOfPictures}");
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            MathNet.Numerics.Control.NativeProviderPath = @"C:\MKL"; // @"C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2019.4.245\windows\redist";
            MathNet.Numerics.Control.UseNativeMKL();
            for (int i = 0;  i < noOfPictures; i++)
            {
                Console.WriteLine($"{i}");
                if (i+1 < noOfPictures)
                {

                    MathNet.Numerics.LinearAlgebra.Matrix<double>[] matrix1 = Utilities.Tools.LoadImageMatrix( frames[i]);
                    MathNet.Numerics.LinearAlgebra.Matrix<double>[] matrix2 = Utilities.Tools.LoadImageMatrix(frames[i + 1]);
                    
                    Diff(matrix1, matrix2);
                }
            }
            stopwatch.Stop();
            Console.WriteLine($"Time  {stopwatch.ElapsedMilliseconds}");
        }


        public static void Diff(MathNet.Numerics.LinearAlgebra.Matrix <double> [] matrix1, MathNet.Numerics.LinearAlgebra.Matrix<double> [] matrix2)
        {
            
            MathNet.Numerics.LinearAlgebra.Matrix<double>[] diffMatrix = new MathNet.Numerics.LinearAlgebra.Matrix<double>[3];
            for (int id = 0; id < 3; id++)
            {
                diffMatrix[id] = MathNet.Numerics.LinearAlgebra.Matrix<double>.Build.Dense(matrix1[0].RowCount, matrix1[0].ColumnCount, 0);
            }

            Parallel.For(0, matrix1.Length, index =>
            {
                diffMatrix[index] = matrix1[index] - matrix2[index];
            });

            
            Int32 unixTimestamp = (Int32)(DateTime.UtcNow.Subtract(new DateTime(1970, 1, 1))).TotalSeconds;
            Utilities.Tools.SaveImage(Utilities.Tools.GetDataPath($"diff-{unixTimestamp}.jpg"), diffMatrix);
        }
        

              
    }
}
