using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

public class DenseLayer : Layer
{
    protected ActivationType ActivationFunction;

    public Matrix Weights;
    public Matrix Bias;

    private int RenderLength;

    public DenseLayer(TensorSize InputTensor, int NumNeurons, ActivationType ActivationFunction) : base(Layer.TensorInputFormat.Tensor, InputTensor, new TensorSize(NumNeurons))
    {
        this.ActivationFunction = ActivationFunction;

        Output = new Matrix(1, OutputTensor.Width);
        Weights = new Matrix(InputTensor.Width, OutputTensor.Width);
        for (int i = 0; i < Weights.Buffer.Length; i++)
        {
            Weights.Buffer[i] = UnityEngine.Random.Range(-1f, 1f);
        }
        Weights.SetData();
        Bias = new Matrix(1, OutputTensor.Width);
        for (int i = 0; i < Bias.Buffer.Length; i++)
        {
            Bias.Buffer[i] = UnityEngine.Random.Range(-1f, 1f);
        }
        Bias.SetData();

        RenderLength = (int)Mathf.Sqrt(OutputTensor.Width) + 1;

        OutputRender.Release();
        OutputRender.width = RenderLength;
        OutputRender.height = RenderLength;
        OutputRender.Create();

        FilterOperation.SetTexture(GenerateTextureKernelIndex, "Output", OutputRender);

        FilterOperation.SetInt("Depth", 1);
        FilterOperation.SetInt("ConvolutionWidth", RenderLength);
        FilterOperation.SetInt("ConvolutionHeight", RenderLength);

        InitDebug();
        DebugObject.name = "DenseLayer";
    }

    protected override void Release() { }

    public override void ForwardProp()
    {
        PriorActivationOutput = (Matrix)Input * Weights;
        PriorActivationOutput.AddInPlace(Bias);
        Output.CopyValues(PriorActivationOutput.GetData());
        switch (ActivationFunction)
        {
            case ActivationType.Tanh: Output.TanhActivation(); break;
            case ActivationType.ReLU: Output.MaxActivation(); break;
            case ActivationType.Sigmoid: Output.SigmoidActivation(); break;
            default: Output.TanhActivation(); break;
        }
        Output.GetData();
    }

    public override void ApplyWeightBiasDeltas(Network.WeightBiasDeltas Deltas, float LearningRate)
    {
        Weights.SubInPlace(Deltas.WeightDeltas * LearningRate);
        Bias.SubInPlace(Deltas.BiasDeltas * LearningRate);
    }

    public override Network.WeightBiasDeltas Backprop(Matrix LayerDeltas)
    {
        Network.WeightBiasDeltas WeightBiasDeltas = new Network.WeightBiasDeltas()
        {
            WeightDeltas = new Matrix(Weights.Rows, Weights.Cols),
            BiasDeltas = new Matrix(Bias.Rows, Bias.Cols)
        };

        Matrix DerivedActivationOutput = new Matrix(1, OutputTensor.Width);
        DerivedActivationOutput.CopyValues(PriorActivationOutput.Buffer);
        switch (ActivationFunction)
        {
            case ActivationType.Tanh: DerivedActivationOutput.TanhActivation_Derivation(); break;
            case ActivationType.Sigmoid: DerivedActivationOutput.SigmoidActivation_Derivation(); break;
            case ActivationType.ReLU: DerivedActivationOutput.ReLUActivation_Derivation(); break;
            default: DerivedActivationOutput.TanhActivation_Derivation(); break;
        }
        Matrix GammaValues = new Matrix(1, OutputTensor.Width);
        Matrix.CWiseMultiply(LayerDeltas, DerivedActivationOutput, GammaValues);
        GammaValues.GetData();

        for (int j = 0; j < OutputTensor.Width; j++)
        {
            for (int i = 0; i < InputTensor.Width; i++)
            {
                WeightBiasDeltas.WeightDeltas.Buffer[i + j * InputTensor.Width] = GammaValues.Buffer[j] * ((Matrix)Input).Buffer[i];
            }
        }
        WeightBiasDeltas.WeightDeltas.SetData();
        WeightBiasDeltas.BiasDeltas.CopyValues(GammaValues.Buffer);

        this.LayerDeltas = GammaValues * Weights.Transpose();

        return WeightBiasDeltas;
    }

    public override void AssignInput(object Input)
    {
        this.Input = Input;
    }

    public override void Render(int RenderIndex)
    {
        FilterOperation.SetInt("FilterIndex", 0);
        FilterOperation.SetInt("NumFeatureMaps", OutputTensor.Depth);
        FilterOperation.SetInt("RenderBufferLength", Output.Buffer.Length);
        RenderBuffer.SetData(Output.GetData());
        FilterOperation.Dispatch(GenerateTextureKernelIndex, RenderLength / 8 + 1, RenderLength / 8 + 1, 1);
    }
}
