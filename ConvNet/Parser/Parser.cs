using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ConvNet.Layers;

namespace ConvNet.Parser
{
    public class section
    {
        public string type;
        public LinkedList<kvp> options;

        static bool is_network(section s)
        {
            return s.is_network();
        }

        public bool is_network()
        {
            return (this.type.Equals("[net]")
                    || this.type.Equals("[Network]"));
        }
    }

    public class Parser
    {
        LAYER_TYPE string_to_layer_type(string type)
        {
            if (type.Equals("[shortcut]")) return LAYER_TYPE.SHORTCUT;
            if (type.Equals("[scale_channels]")) return LAYER_TYPE.SCALE_CHANNELS;
            if (type.Equals("[crop]")) return LAYER_TYPE.CROP;
            if (type.Equals("[cost]")) return LAYER_TYPE.COST;
            if (type.Equals("[detection]")) return LAYER_TYPE.DETECTION;
            if (type.Equals("[region]")) return LAYER_TYPE.REGION;
            if (type.Equals("[yolo]")) return LAYER_TYPE.YOLO;
            if (type.Equals("[local]")) return LAYER_TYPE.LOCAL;
            if (type.Equals("[conv]")
             || type.Equals("[convolutional]")) return LAYER_TYPE.CONVOLUTIONAL;
            if (type.Equals("[activation]")) return LAYER_TYPE.ACTIVE;
            if (type.Equals("[net]")
             || type.Equals("[Network]")) return LAYER_TYPE.NETWORK;
            if (type.Equals("[crnn]")) return LAYER_TYPE.CRNN;
            if (type.Equals("[gru]")) return LAYER_TYPE.GRU;
            if (type.Equals("[lstm]")) return LAYER_TYPE.LSTM;
            if (type.Equals("[ConvLSTMLayer]")) return LAYER_TYPE.CONV_LSTM;
            if (type.Equals("[rnn]")) return LAYER_TYPE.RNN;
            if (type.Equals("[conn]")
             || type.Equals("[connected]")) return LAYER_TYPE.CONNECTED;
            if (type.Equals("[max]")
             || type.Equals("[maxpool]")) return LAYER_TYPE.MAXPOOL;
            if (type.Equals("[reorg3d]")) return LAYER_TYPE.REORG;
            if (type.Equals("[reorg]")) return LAYER_TYPE.REORG_OLD;
            if (type.Equals("[avg]")
             || type.Equals("[avgpool]")) return LAYER_TYPE.AVGPOOL;
            if (type.Equals("[dropout]")) return LAYER_TYPE.DROPOUT;
            if (type.Equals("[lrn]")
             || type.Equals("[normalization]")) return LAYER_TYPE.NORMALIZATION;
            if (type.Equals("[batchnorm]")) return LAYER_TYPE.BATCHNORM;
            if (type.Equals("[soft]")
             || type.Equals("[softmax]")) return LAYER_TYPE.SOFTMAX;
            if (type.Equals("[route]")) return LAYER_TYPE.ROUTE;
            if (type.Equals("[upsample]")) return LAYER_TYPE.UPSAMPLE;
            if (type.Equals("[empty]")) return LAYER_TYPE.EMPTY;
            return LAYER_TYPE.BLANK;
        }
    }

}