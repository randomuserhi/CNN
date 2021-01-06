using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using UnityEngine;
using UnityEngine.UI;

public struct TensorSize
{
    public int Width;
    public int Height;
    public int Depth;

    public TensorSize(int Width = 1, int Height = 1, int Depth = 1)
    {
        this.Width = Width;
        this.Height = Height;
        this.Depth = Depth;
    }

    public override string ToString()
    {
        return "(" + Width + ", " + Height + ", " + Depth + ")";
    }
}

public abstract class Layer
{
    public enum TensorInputFormat
    {
        Tensor,
        Texture
    }

    public readonly TensorInputFormat InputFormat;
    public readonly TensorSize InputTensor;
    public readonly TensorSize OutputTensor;

    public object Input;
    public Matrix PriorActivationOutput;
    public Matrix Output;

    protected ComputeShader FilterOperation;
    protected int GenerateTextureKernelIndex;
    protected ComputeBuffer RenderBuffer;

    private static int RenderOffset = 0;

    #region Debugging

    protected GameObject DebugObject;
    protected RenderTexture OutputRender;

    protected void InitDebug()
    {
        DebugObject = new GameObject();
        DebugObject.transform.position = new Vector3(RenderOffset, 0, 0);

        DebugObject.AddComponent<Canvas>().renderMode = RenderMode.WorldSpace;
        DebugObject.AddComponent<CanvasScaler>();
        DebugObject.AddComponent<GraphicRaycaster>();
        DebugObject.GetComponent<RectTransform>().sizeDelta = Vector2.zero;
        RawImage Image = DebugObject.AddComponent<RawImage>();
        Image.rectTransform.sizeDelta = new Vector2(1, 1);
        Image.texture = OutputRender;

        RenderOffset += 1;
    }

    public abstract void LogParams();

    #endregion

    public Layer(TensorInputFormat InputFormat, TensorSize InputTensor, TensorSize OutputTensor, bool CustomRender)
    {
        MemoryManager.DisposeList.Add(this);

        this.InputFormat = InputFormat;
        this.InputTensor = InputTensor;
        this.OutputTensor = OutputTensor;
    }

    public Layer(TensorInputFormat InputFormat, TensorSize InputTensor, TensorSize OutputTensor)
    {
        MemoryManager.DisposeList.Add(this);

        this.InputFormat = InputFormat;
        this.InputTensor = InputTensor;
        this.OutputTensor = OutputTensor;

        OutputRender = new RenderTexture(OutputTensor.Width, OutputTensor.Height, 8)
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

        FilterOperation.SetInt("Depth", InputTensor.Depth);
        FilterOperation.SetInt("ConvolutionWidth", OutputTensor.Width);
        FilterOperation.SetInt("ConvolutionHeight", OutputTensor.Height);

        InitDebug();
    }

    public virtual void GetGenerateTextureKernel()
    {
        GenerateTextureKernelIndex = FilterOperation.FindKernel("GenerateTexture");
    }
    public abstract void AssignInput(object In);
    public abstract void ForwardProp();
    public abstract void Render(int RenderIndex); //RenderIndex refers to which featuremap / neuron

    public void Dispose()
    {
        MemoryManager.DisposeList.Remove(this);
        RenderBuffer.Dispose();
        Release();
    }

    protected abstract void Release();

    public Matrix GammaValues;
    public Matrix LayerDeltas;
    public abstract Network.WeightBiasDeltas Backprop(Matrix LayerDeltas);
    public abstract void ApplyWeightBiasDeltas(Network.WeightBiasDeltas BackPropEval, float LearningRate);
}