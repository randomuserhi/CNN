using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using UnityEngine;

public class Flattening : Layer
{
    public Flattening(TensorSize InputTensor) : base(Layer.TensorInputFormat.Tensor, InputTensor, new TensorSize(InputTensor.Width * InputTensor.Height * InputTensor.Depth))
    {
        RenderLength = (int)(Mathf.Sqrt(OutputTensor.Width)) + 1;

        Output = new Matrix(1, OutputTensor.Width);

        OutputRender[0].Release();
        OutputRender[0].width = RenderLength;
        OutputRender[0].height = RenderLength;
        OutputRender[0].Create();

        FilterOperation.SetTexture(GenerateTextureKernelIndex, "Output", OutputRender[0]);

        FilterOperation.SetInt("Depth", 1);
        FilterOperation.SetInt("ConvolutionWidth", RenderLength);
        FilterOperation.SetInt("ConvolutionHeight", RenderLength);

        InitDebug();
        for (int i = 0; i < DebugObject.Length; i++)
            DebugObject[i].name = "FlatteningLayer";
    }

    protected override void Release()
    {
    }

    public override void AssignInput(object Input)
    {
        this.Input = Input;
    }

    public override void ForwardProp()
    {
        Output.CopyValues(((Matrix)Input).Buffer);
        PriorActivationOutput = Output;
    }

    public override Network.WeightBiasDeltas Backprop(Matrix LayerDeltas)
    {
        this.LayerDeltas = new Matrix(InputTensor.Depth, InputTensor.Width * InputTensor.Height);
        this.LayerDeltas.CopyValues(LayerDeltas.GetData());
        return new Network.WeightBiasDeltas();
    }

    public override void ApplyWeightBiasDeltas(Network.WeightBiasDeltas Deltas, float LearningRate)
    {
        return;
    }

    private int RenderLength;
    public override void Render(int RenderIndex)
    {
        FilterOperation.SetInt("FilterIndex", RenderIndex);
        FilterOperation.SetInt("NumFeatureMaps", OutputTensor.Depth);
        FilterOperation.SetInt("RenderBufferLength", Output.Buffer.Length);
        RenderBuffer.SetData(Output.GetData());
        FilterOperation.Dispatch(GenerateTextureKernelIndex, RenderLength / 8 + 1, RenderLength / 8 + 1, 1);
    }
}
