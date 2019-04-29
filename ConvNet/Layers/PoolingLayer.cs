﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace ConvNet.Layers
{
    class PoolingLayer<PoolingType> : Layer where PoolingType : Poolings.PoolingType, new()
    {
        public PoolingLayer(string layerName) : base(layerName)
        {

        }

        public override Vector<double> Inputs { set => throw new NotImplementedException(); }

        public override Vector<double> Outputs => throw new NotImplementedException();

        public override Vector<double> BackPropagation(Vector<double> nextDelta)
        {
            throw new NotImplementedException();
        }

        public override void ForwardPropagation()
        {
            throw new NotImplementedException();
        }
    }
}
