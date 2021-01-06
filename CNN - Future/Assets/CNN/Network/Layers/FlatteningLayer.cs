using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using UnityEngine;

public class FlatteningLayer : Layer
{
    private int RenderLength;

    public override void LogParams()
    {
    }

    public FlatteningLayer(TensorSize InputTensor) : base(Layer.TensorInputFormat.Tensor, InputTensor, new TensorSize(InputTensor.Width * InputTensor.Height * InputTensor.Depth), true)
    {
        Output = new Matrix(1, OutputTensor.Width);

        RenderLength = (int)(Mathf.Sqrt(OutputTensor.Width)) + 1;
        OutputRender = new RenderTexture(RenderLength, RenderLength, 8)
        {
            enableRandomWrite = true,
            filterMode = FilterMode.Point,
            anisoLevel = 1
        };
        OutputRender.Create();

        FilterOperation = UnityEngine.Object.Instantiate(Resources.Load<ComputeShader>("FilterOperation"));
        GetGenerateTextureKernel();
        RenderBuffer = new ComputeBuffer(OutputTensor.Width * OutputTensor.Height * OutputTensor.Depth, sizeof(float));
        FilterOperation.SetBuffer(GenerateTextureKernelIndex, "Tensor", RenderBuffer);
        FilterOperation.SetTexture(GenerateTextureKernelIndex, "Output", OutputRender);

        FilterOperation.SetInt("Depth", 1);
        FilterOperation.SetInt("ConvolutionWidth", RenderLength);
        FilterOperation.SetInt("ConvolutionHeight", RenderLength);

        InitDebug();
        DebugObject.name = "FlatteningLayer";
    }

    public override void AssignInput(object In)
    {
        Input = In;
    }

    public override void ForwardProp()
    {
        Output.CopyValues(((Matrix)Input).Buffer);
        PriorActivationOutput = Output;
    }

    public override void Render(int RenderIndex)
    {
        FilterOperation.SetInt("FilterIndex", 0);
        FilterOperation.SetInt("NumFeatureMaps", 1);
        RenderBuffer.SetData(Output.GetData());
        FilterOperation.SetInt("RenderBufferLength", Output.Buffer.Length);
        FilterOperation.Dispatch(GenerateTextureKernelIndex, RenderLength / 8 + 1, RenderLength / 8 + 1, 1); //we add 1 here to also fill in remainder
    }

    protected override void Release()
    {
    }

    public override Network.WeightBiasDeltas Backprop(Matrix LayerDeltas)
    {
        this.LayerDeltas = new Matrix(InputTensor.Depth, InputTensor.Width * InputTensor.Height);
        this.LayerDeltas.CopyValues(LayerDeltas.GetData());
        return new Network.WeightBiasDeltas(); //no relevant backprop for flattening, simply unflattening the layerdeltas
    }
    public override void ApplyWeightBiasDeltas(Network.WeightBiasDeltas WeightBiasDeltas, float LearningRate)
    {
        return; //no relevant backprop for flattening
    }
}
