using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using UnityEngine;

public class DenseLayer : Layer
{
    public enum ActivationType
    {
        Tanh,
        Sigmoid,
        ReLU
    }
    protected ActivationType ActivationFunction;

    private int RenderLength;
    public Matrix Weights;
    public Matrix Bias;

    public override void LogParams()
    {
        Weights.GetData();
        Debug.Log("Dense: [W]" + Weights);
        Bias.GetData();
        Debug.Log("Dense: [B]" + Bias);
    }

    public DenseLayer(TensorSize InputTensor, int NumNeurons, ActivationType ActivationFunction) : base(Layer.TensorInputFormat.Tensor, InputTensor, new TensorSize(NumNeurons), true)
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
        DebugObject.name = "DenseLayer";
    }

    public override void AssignInput(object In)
    {
        Input = In;
    }

    public override void ForwardProp()
    {
        PriorActivationOutput = (Matrix)Input * Weights;
        PriorActivationOutput.AddInPlace(Bias);
        Output.CopyValues(PriorActivationOutput.GetData());

        switch (ActivationFunction)
        {
            case ActivationType.Tanh: Output.TanhActivation(); break;
            case ActivationType.Sigmoid: Output.SigmoidActivation(); break;
            default: Output.TanhActivation(); break;
        }
        Output.GetData();
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
        Network.WeightBiasDeltas WeightBiasDeltas = new Network.WeightBiasDeltas()
        {
            WeightDeltas = new Matrix(Weights.Rows, Weights.Cols),
            BiasDeltas = new Matrix(Bias.Rows, Bias.Cols)
        };

        //Copy pre activated values
        Matrix DerivedActivationOutput = new Matrix(1, OutputTensor.Width);
        DerivedActivationOutput.CopyValues(PriorActivationOutput.Buffer);
        switch (ActivationFunction)
        {
            case ActivationType.Tanh: DerivedActivationOutput.TanhActivation_Derivation(); break;
            case ActivationType.Sigmoid: DerivedActivationOutput.SigmoidActivation_Derivation(); break;
            case ActivationType.ReLU: DerivedActivationOutput.ReLUActivation_Derivation(); break;
            default: DerivedActivationOutput.TanhActivation_Derivation(); break;
        }
        DerivedActivationOutput.GetData();

        //Calculate gamma values
        GammaValues = new Matrix(1, OutputTensor.Width);
        Matrix.CWiseMultiply(DerivedActivationOutput, LayerDeltas, GammaValues);
        GammaValues.GetData();

        //Calculate weight and bias deltas => TODO:: optimize this process using GEMM
        for (int j = 0; j < OutputTensor.Width; j++)
        {
            for (int i = 0; i < InputTensor.Width; i++)
            {
                WeightBiasDeltas.WeightDeltas.Buffer[j * InputTensor.Width + i] = GammaValues.Buffer[j] * ((Matrix)Input).Buffer[i];
            }
        }
        WeightBiasDeltas.WeightDeltas.SetData();
        WeightBiasDeltas.BiasDeltas.CopyValues(GammaValues.Buffer);

        //Calculate LayerDeltas 
        this.LayerDeltas = GammaValues * Weights.Transpose();

        return WeightBiasDeltas;
    }

    public override void ApplyWeightBiasDeltas(Network.WeightBiasDeltas WeightBiasDeltas, float LearningRate)
    {
        Debug.Log("DW:" + WeightBiasDeltas.WeightDeltas);
        Debug.Log("DB:" + WeightBiasDeltas.BiasDeltas);
        Weights.SubInPlace(WeightBiasDeltas.WeightDeltas * LearningRate);
        Bias.SubInPlace(WeightBiasDeltas.BiasDeltas * LearningRate);
    }
}