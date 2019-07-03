using MathNet.Numerics.LinearAlgebra;

namespace ConvNet.Layers
{
    public enum LAYER_TYPE
    {
        CONVOLUTIONAL,
        DECONVOLUTIONAL,
        CONNECTED,
        MAXPOOL,
        SOFTMAX,
        DETECTION,
        DROPOUT,
        CROP,
        ROUTE,
        COST,
        NORMALIZATION,
        AVGPOOL,
        LOCAL,
        SHORTCUT,
        SCALE_CHANNELS,
        ACTIVE,
        RNN,
        GRU,
        LSTM,
        CONV_LSTM,
        CRNN,
        BATCHNORM,
        NETWORK,
        XNOR,
        REGION,
        YOLO,
        ISEG,
        REORG,
        REORG_OLD,
        UPSAMPLE,
        LOGXENT,
        L2NORM,
        EMPTY,
        BLANK
    }

    public abstract class Layer
    {
        protected int inputSize;
        public abstract Vector<double> Inputs { set; }
        protected int outputSize;
        public abstract Vector<double> Outputs { get; }
        protected int stride;
        public string LayerType { get; protected set; }
        public string LayerName { get; protected set; }
        public string GenericsType { get; protected set; }
        public virtual Vector<double> PredictOutputs { get { return Outputs; } }
        public virtual Vector<double> Weights { get { return null; } protected set { } }
        public virtual Vector<double> Biases { get { return null; } protected set { } }

        public Layer(string layerName)
        {
            LayerName = layerName;
            LayerType = "Layer";
            GenericsType = "None";
            inputSize = -1;
            outputSize = -1;
            stride = -1;
        }
        public virtual void Prediction() { ForwardPropagation(); }

        public abstract void ForwardPropagation();

        public abstract Vector<double> BackPropagation(Vector<double> nextDelta);
        public virtual void WeightUpdate(double eta, double mu, double lambda) { }

        public virtual void GenerateWeights(double lower = -0.1, double upper = 0.1) { }
        public virtual void GenerateWeights(Vector<double> weights) { }

        public virtual string ToString(string fmt) { return "None\n"; }

        public bool CheckSize(int previousLayerOutputSize) { return previousLayerOutputSize == inputSize; }
    }
}