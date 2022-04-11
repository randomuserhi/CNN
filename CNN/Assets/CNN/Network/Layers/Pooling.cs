using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using UnityEngine;

public class Pooling : Layer
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

    private int PoolingSize;
    private int Stride;

    public Pooling(TensorSize InputTensor, int PoolingSize, int Stride, PoolingType PoolingFunction = PoolingType.Max)
        : base
        (
            TensorInputFormat.Tensor,
            InputTensor,
            new TensorSize
            (
                 Mathf.CeilToInt((InputTensor.Width - (PoolingSize - 1)) / (float)Stride),
                 Mathf.CeilToInt((InputTensor.Height - (PoolingSize - 1)) / (float)Stride),
                 InputTensor.Depth
            )
        )
    {
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

        FilterOperation.SetInt("FilterWidth", PoolingSize);
        FilterOperation.SetInt("FilterHeight", PoolingSize);
        FilterOperation.SetInt("TextureWidth", InputTensor.Width);
        FilterOperation.SetInt("TextureHeight", InputTensor.Height);
        FilterOperation.SetInt("Stride", Stride);

        InputBuffer = new ComputeBuffer(InputTensor.Width * InputTensor.Height * InputTensor.Depth, sizeof(float));
        ConvolutionBuffer = new ComputeBuffer(OutputTensor.Width * OutputTensor.Height * OutputTensor.Depth, sizeof(float));

        for (int i = 0; i < PoolingKernelIndex.Length; i++)
        {
            FilterOperation.SetBuffer(PoolingKernelIndex[i], "MatrixInput", InputBuffer);
            FilterOperation.SetBuffer(PoolingKernelIndex[i], "ConvolutionTensor", ConvolutionBuffer);

            //For backprop
            FilterOperation.SetBuffer(BackPropPoolingKernelIndex[i], "MatrixInput", InputBuffer);
            FilterOperation.SetBuffer(BackPropPoolingKernelIndex[i], "ConvolutionTensor", ConvolutionBuffer);
        }

        Output = new Matrix(OutputTensor.Depth, OutputTensor.Width * OutputTensor.Height);

        for (int i = 0; i < DebugObject.Length; i++)
            DebugObject[i].name = "Pooling Layer";
    }

    protected override void Release()
    {
        InputBuffer.Dispose();
        ConvolutionBuffer.Dispose();
    }

    public override void AssignInput(object Input)
    {
        this.Input = Input;
    }

    public override void ForwardProp()
    {
        InputBuffer.SetData(((Matrix)Input).Buffer);

        ThreadGroup Threading = CalculateNumThreadGroups(PoolingKernelIndex, OutputTensor.Width * OutputTensor.Height);
        FilterOperation.Dispatch(Threading.FilterThreadGroupKernel, Threading.NumThreadGroups, InputTensor.Depth, 1);
        ConvolutionBuffer.GetData(Output.Buffer);
        Output.SetData();
    }

    public override Network.WeightBiasDeltas Backprop(Matrix LayerDeltas)
    {
        ConvolutionBuffer.SetData(LayerDeltas.GetData());

        ThreadGroup Threading = CalculateNumThreadGroups(BackPropPoolingKernelIndex, OutputTensor.Width * OutputTensor.Height);
        FilterOperation.Dispatch(Threading.FilterThreadGroupKernel, Threading.NumThreadGroups, InputTensor.Depth, 1);
        this.LayerDeltas = new Matrix(InputTensor.Depth, InputTensor.Width * InputTensor.Height);
        InputBuffer.GetData(this.LayerDeltas.Buffer);
        this.LayerDeltas.SetData();

        return new Network.WeightBiasDeltas();
    }

    public override void ApplyWeightBiasDeltas(Network.WeightBiasDeltas Deltas, float LearningRate)
    {
        return;
    }

    public override void Render(int RenderIndex)
    {
        for (int i = 0; i < OutputTensor.Depth; i++)
        {
            FilterOperation.SetTexture(GenerateTextureKernelIndex, "Output", OutputRender[i]);
            FilterOperation.SetInt("FilterIndex", i);
            FilterOperation.SetInt("NumFeatureMaps", OutputTensor.Depth);
            FilterOperation.SetInt("RenderBufferLength", Output.Buffer.Length);
            RenderBuffer.SetData(Output.GetData());
            FilterOperation.Dispatch(GenerateTextureKernelIndex, OutputTensor.Width / 8 + 1, OutputTensor.Height / 8 + 1, 1);
        }
    }
}
