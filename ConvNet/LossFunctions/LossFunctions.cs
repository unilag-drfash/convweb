using System;

namespace ConvNet.LossFunctions
{
    public interface ILossFunction
    {
        double f(double y, double t);
        double df(double y, double t);
        string Type();
    }


    public class MSE : ILossFunction
    {
        public double f(double y, double t)
        {
            // MSE = 1/N∑(y-t)^2
            // (y-t)^2
            return (y - t) * (y - t) / 2.0;
        }
        public double df(double y, double t)
        {
            return y - t;
        }
        public string Type() { return "MSE"; }
    }


    public class MultiCrossEntropy : ILossFunction
    {
        public double f(double y, double t)
        {
            if (y == t || t == 0) { return 0; }
            return -(t * Math.Log(y));
        }
        public double df(double y, double t)
        {
            return y - t;
        }
        public string Type() { return "MultiClassCrossEntropy"; }
    }

    public class BinaryCrossEntropy : ILossFunction
    {
        public double f(double y, double t)
        {
            if (y == t) { return 0; }
            else
            {
                ;
                return -(t * Math.Log(y) + (1.0 - t) * Math.Log(1.0 - y));
            }
        }
        public double df(double y, double t)
        {
            return (y - t);
        }
        public string Type() { return "BinaryCrossEntropy"; }
    }

}
