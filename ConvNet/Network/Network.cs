using ConvNet.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvNet.Network
{
    class Network
    {
        private List<Layer> layers;

        public Network()
        {

        }

        public Network(params Layer[] layers)
        {
            this.layers = new List<Layer>(layers);
        }

        public void AddLayer(Layer layer) { layers.Add(layer); }
        public void AddLayers(params Layer[] layers) { this.layers.AddRange(layers); }
    }
}
