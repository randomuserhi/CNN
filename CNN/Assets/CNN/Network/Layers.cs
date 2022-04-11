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

public enum ActivationType
{
    Sigmoid,
    Tanh,
    ReLU
}

public abstract class Layer
{
    public enum TensorInputFormat
    {
        Tensor,
        Texture
    }

    public readonly TensorInputFormat InputFormat;

    public object Input;
    public readonly TensorSize InputTensor;
    protected Matrix PriorActivationOutput;
    public Matrix Output;
    public readonly TensorSize OutputTensor;

    protected ComputeShader FilterOperation;
    protected int GenerateTextureKernelIndex;
    protected ComputeBuffer RenderBuffer;

    private static int RenderOffset = 0;

    #region Debugging

    protected GameObject[] DebugObject;
    protected RenderTexture[] OutputRender;

    protected void InitDebug()
    {
        if (DebugObject == null)
        {
            RenderOffset++;
            DebugObject = new GameObject[OutputTensor.Depth];
            for (int i = 0; i < OutputTensor.Depth; i++)
            {
                DebugObject[i] = new GameObject();
                DebugObject[i].transform.position = new Vector3(RenderOffset * 1.2f - 10f/2f, OutputTensor.Depth / 2f - i - 0.5f, 0);
                DebugObject[i].AddComponent<Canvas>().renderMode = RenderMode.WorldSpace;
                DebugObject[i].AddComponent<CanvasScaler>();
                DebugObject[i].AddComponent<GraphicRaycaster>();
                DebugObject[i].GetComponent<RectTransform>().sizeDelta = Vector2.zero;
                DebugObject[i].AddComponent<RawImage>();

                RawImage Image = DebugObject[i].GetComponent<RawImage>();
                Image.rectTransform.sizeDelta = new Vector2(1, 1);
                Image.texture = OutputRender[i];
            }
        }
    }

    #endregion

    public Layer(TensorInputFormat InputFormat, TensorSize InputTensor, TensorSize OutputTensor)
    {
        MemoryManager.DisposeList.Add(this);

        this.InputFormat = InputFormat;
        this.InputTensor = InputTensor;
        this.OutputTensor = OutputTensor;

        OutputRender = new RenderTexture[OutputTensor.Depth];
        for (int i = 0; i < OutputTensor.Depth; i++)
        {
            OutputRender[i] = new RenderTexture(OutputTensor.Width, OutputTensor.Height, 8)
            {
                enableRandomWrite = true,
                filterMode = FilterMode.Point,
                anisoLevel = 1
            };
            OutputRender[i].Create();
        }

        FilterOperation = UnityEngine.Object.Instantiate(Resources.Load<ComputeShader>("FilterOperation"));
        GetGenerateTextureKernel();
        RenderBuffer = new ComputeBuffer(OutputTensor.Width * OutputTensor.Height * OutputTensor.Depth, sizeof(float));
        FilterOperation.SetBuffer(GenerateTextureKernelIndex, "Tensor", RenderBuffer);
        FilterOperation.SetTexture(GenerateTextureKernelIndex, "Output", OutputRender[0]);

        FilterOperation.SetInt("Depth", InputTensor.Depth);
        FilterOperation.SetInt("ConvolutionWidth", OutputTensor.Width);
        FilterOperation.SetInt("ConvolutionHeight", OutputTensor.Height);

        InitDebug();
    }

    protected struct ThreadGroup
    {
        public int NumThreadGroups;
        public int FilterThreadGroupKernel;
    }
    protected ThreadGroup CalculateNumThreadGroups(int[] FilterKernelIndex, int TensorSize)
    {
        ThreadGroup Result = new ThreadGroup();
        /*Result.FilterThreadGroupKernel = FilterKernelIndex[0];
        int ThreadGroups = TensorSize / 65000;
        if (ThreadGroups < 1) ThreadGroups = 1;
        if (TensorSize > 65000)
        {
            if (ThreadGroups <= 8)
            {
                ThreadGroups = 8;
                Result.FilterThreadGroupKernel = FilterKernelIndex[0];
            }
            else if (ThreadGroups <= 16)
            {
                ThreadGroups = 16;
                Result.FilterThreadGroupKernel = FilterKernelIndex[1];
            }
            else if (ThreadGroups <= 32)
            {
                ThreadGroups = 32;
                Result.FilterThreadGroupKernel = FilterKernelIndex[2];
            }
            else
            {
                Debug.LogError("Image size is too large.");
                Result.NumThreadGroups = -1;
                return Result;
            }
        }
        Result.NumThreadGroups = TensorSize / ThreadGroups;
        if (TensorSize % ThreadGroups != 0) Result.NumThreadGroups++; //Add 1 to fit in the remainder*/
        Result.FilterThreadGroupKernel = FilterKernelIndex[2];
        Result.NumThreadGroups = TensorSize / 32 + 1;
        return Result;
    }

    public virtual void GetGenerateTextureKernel()
    {
        GenerateTextureKernelIndex = FilterOperation.FindKernel("GenerateTexture");
    }

    public abstract void AssignInput(object Input);

    public abstract void ForwardProp();

    public abstract void Render(int RenderIndex);

    public abstract void ApplyWeightBiasDeltas(Network.WeightBiasDeltas Deltas, float LearningRate);

    public Matrix LayerDeltas;
    public abstract Network.WeightBiasDeltas Backprop(Matrix LayerDeltas);

    public void Dispose()
    {
        MemoryManager.DisposeList.Remove(this);
        RenderBuffer.Dispose();
        Release();
    }

    protected abstract void Release();
}
