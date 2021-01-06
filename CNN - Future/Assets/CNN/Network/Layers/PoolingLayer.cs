using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using UnityEngine;

public class PoolingLayer : Layer
{
    public enum PoolingType
    {
        Max
    }
    protected PoolingType PoolingFunction;

    private int[] PoolingKernelIndex;
    private int[] BackPropPoolingKernelIndex;

    private ComputeBuffer InputBuffer;
    private ComputeBuffer ConvolutionBuffer;

    private Matrix ReconstructedTensor; //Used for back prop

    int PoolingSize;
    int Stride;

    public override void LogParams()
    {
    }

    public PoolingLayer(TensorSize InputTensor, int PoolingSize, int Stride, PoolingType PoolingFunction = PoolingType.Max) 
        : base(Layer.TensorInputFormat.Tensor, 
               InputTensor,
                new TensorSize(
                   Mathf.CeilToInt((InputTensor.Width - (PoolingSize - 1)) / (float)Stride),
                   Mathf.CeilToInt((InputTensor.Height - (PoolingSize - 1)) / (float)Stride),
                   InputTensor.Depth))
    {
        DebugObject.name = "PoolingLayer";

        ReconstructedTensor = new Matrix(InputTensor.Depth, InputTensor.Width * InputTensor.Height);

        this.PoolingFunction = PoolingFunction;
        this.PoolingSize = PoolingSize;
        this.Stride = Stride;

        PoolingKernelIndex = new int[3]
        {
            FilterOperation.FindKernel("PoolingOperation_8"),
            FilterOperation.FindKernel("PoolingOperation_16"),
            FilterOperation.FindKernel("PoolingOperation_32")
        };
        BackPropPoolingKernelIndex = new int[3]
        {
            FilterOperation.FindKernel("BackPropPoolingOperation_8"),
            FilterOperation.FindKernel("BackPropPoolingOperation_16"),
            FilterOperation.FindKernel("BackPropPoolingOperation_32")
        };

        FilterOperation.SetInt("TextureWidth", InputTensor.Width);
        FilterOperation.SetInt("TextureHeight", InputTensor.Height);
        FilterOperation.SetInt("FilterWidth", PoolingSize);
        FilterOperation.SetInt("FilterHeight", PoolingSize);
        FilterOperation.SetInt("Stride", Stride);

        InputBuffer = new ComputeBuffer(InputTensor.Width * InputTensor.Height * InputTensor.Depth, sizeof(float));
        ConvolutionBuffer = new ComputeBuffer(OutputTensor.Width * OutputTensor.Height * OutputTensor.Depth, sizeof(float));

        for (int i = 0; i < PoolingKernelIndex.Length; i++)
        {
            FilterOperation.SetBuffer(PoolingKernelIndex[i], "MatrixInput", InputBuffer);
            FilterOperation.SetBuffer(PoolingKernelIndex[i], "ConvolutionTensor", ConvolutionBuffer);
            FilterOperation.SetBuffer(BackPropPoolingKernelIndex[i], "MatrixInput", InputBuffer);
            FilterOperation.SetBuffer(BackPropPoolingKernelIndex[i], "ConvolutionTensor", ConvolutionBuffer);
        }

        Output = new Matrix(OutputTensor.Depth, OutputTensor.Width * OutputTensor.Height);
    }

    public override void AssignInput(object In)
    {
        Input = In;
    }

    public override void ForwardProp()
    {
        InputBuffer.SetData(((Matrix)Input).GetData());

        //Pick the correct number of thread groups based on size of image
        int TotalConvolutionTensorSize = OutputTensor.Width * OutputTensor.Height;
        int FilterThreadGroupKernel = PoolingKernelIndex[0];
        int ThreadGroups = TotalConvolutionTensorSize / 65000;
        if (ThreadGroups < 1) ThreadGroups = 1;
        if (TotalConvolutionTensorSize > 65000)
        {
            if (ThreadGroups <= 8)
            {
                ThreadGroups = 8;
                FilterThreadGroupKernel = PoolingKernelIndex[0];
            }
            else if (ThreadGroups <= 16)
            {
                ThreadGroups = 16;
                FilterThreadGroupKernel = PoolingKernelIndex[1];
            }
            else if (ThreadGroups <= 32)
            {
                ThreadGroups = 32;
                FilterThreadGroupKernel = PoolingKernelIndex[2];
            }
            else
            {
                UnityEngine.Debug.LogError("Image size is too large.");
                return;
            }
        }
        int NumThreadGroups = TotalConvolutionTensorSize / ThreadGroups;
        if (TotalConvolutionTensorSize % ThreadGroups != 0) NumThreadGroups++; //Add 1 to fit in the remainder
        FilterOperation.Dispatch(FilterThreadGroupKernel, NumThreadGroups, InputTensor.Depth, 1); //3 is because depth of 3 for RGB colour
        ConvolutionBuffer.GetData(Output.Buffer);
        Output.SetData();
    }

    public override void Render(int RenderIndex)
    {
        FilterOperation.SetInt("FilterIndex", RenderIndex);
        FilterOperation.SetInt("NumFeatureMaps", OutputTensor.Depth);
        FilterOperation.SetInt("RenderBufferLength", Output.Buffer.Length);
        RenderBuffer.SetData(Output.GetData());
        FilterOperation.Dispatch(GenerateTextureKernelIndex, OutputTensor.Width / 8 + 1, OutputTensor.Height / 8 + 1, 1); //we add 1 here to also fill in remainder
    }

    protected override void Release()
    {
        InputBuffer.Dispose();
        ConvolutionBuffer.Dispose();
    }

    public override Network.WeightBiasDeltas Backprop(Matrix LayerDeltas)
    {
        //https://datascience.stackexchange.com/questions/78132/back-propagation-through-a-simple-convolutional-neural-network?noredirect=1&lq=1
        //Simply rebuilding input tensor of layer deltas in the right location per pixels where picked
        ConvolutionBuffer.SetData(LayerDeltas.GetData());

        //Pick the correct number of thread groups based on size of image => tbh dont need to recalculate this we hv it from forward prop but laziness
        int TotalConvolutionTensorSize = OutputTensor.Width * OutputTensor.Height;
        int FilterThreadGroupKernel = BackPropPoolingKernelIndex[0];
        int ThreadGroups = TotalConvolutionTensorSize / 65000;
        if (ThreadGroups < 1) ThreadGroups = 1;
        if (TotalConvolutionTensorSize > 65000)
        {
            if (ThreadGroups <= 8)
            {
                ThreadGroups = 8;
                FilterThreadGroupKernel = BackPropPoolingKernelIndex[0];
            }
            else if (ThreadGroups <= 16)
            {
                ThreadGroups = 16;
                FilterThreadGroupKernel = BackPropPoolingKernelIndex[1];
            }
            else if (ThreadGroups <= 32)
            {
                ThreadGroups = 32;
                FilterThreadGroupKernel = BackPropPoolingKernelIndex[2];
            }
            else
            {
                UnityEngine.Debug.LogError("(BackProp)Image size is too large.");
                return new Network.WeightBiasDeltas();
            }
        }
        int NumThreadGroups = TotalConvolutionTensorSize / ThreadGroups;
        if (TotalConvolutionTensorSize % ThreadGroups != 0) NumThreadGroups++; //Add 1 to fit in the remainder
        FilterOperation.Dispatch(FilterThreadGroupKernel, NumThreadGroups, InputTensor.Depth, 1); //3 is because depth of 3 for RGB colour
        this.LayerDeltas = new Matrix(InputTensor.Depth, InputTensor.Width * InputTensor.Height);
        InputBuffer.GetData(this.LayerDeltas.Buffer);
        this.LayerDeltas.SetData();

        return new Network.WeightBiasDeltas(); //Pooling layer has nothing to return
    }
    public override void ApplyWeightBiasDeltas(Network.WeightBiasDeltas WeightBiasDeltas, float LearningRate)
    {
        return; //No application, pooling layer has no weights or biases
    }
}
