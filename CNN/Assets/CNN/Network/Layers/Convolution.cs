using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using UnityEngine;
using UnityEngine.UI;

public abstract class ConvolutionLayer : Layer
{
    protected int[] FilterKernelIndex;

    private ComputeBuffer ConvolutionBuffer;
    private Matrix ConvolutionTensor;

    public Matrix FilterTensor;
    public Matrix Bias;

    protected ActivationType ActivationFunction = ActivationType.Tanh;

    private int FilterSize;
    private int Stride;
    private int ZeroPadding;

    //for bprop
    protected int[] BackPropFilterKernelIndex;

    protected int DialateKernelIndex;
    protected ComputeShader BackpropWeightOperation;
    private Matrix WeightsConvolutionTensor;
    private ComputeBuffer WeightsConvolutionBuffer;
    private ComputeBuffer WeightsDialatedOutputBuffer;
    private ComputeBuffer WeightsNonDialatedOutputBuffer;

    private int[] FilterMatrixKernel;
    private ComputeShader BackPropLayerDeltasOperation;
    private ComputeBuffer BackPropLayerDeltasConvolutionTensor;
    private Matrix LayerDeltasConvolutionTensor;
    private TensorSize LayerDeltaConvoTensor;
    private ComputeBuffer LayerDeltasConvolutionBuffer;
    private ComputeBuffer LayerDeltasDialatedOutputBuffer;
    private ComputeBuffer LayerDeltasNonDialatedOutputBuffer;

    protected override void Release()
    {
        ConvolutionBuffer.Dispose();
        WeightsDialatedOutputBuffer.Dispose();
        WeightsNonDialatedOutputBuffer.Dispose();
        WeightsConvolutionBuffer.Dispose();
        LayerDeltasDialatedOutputBuffer.Dispose();
        LayerDeltasNonDialatedOutputBuffer.Dispose();
        BackPropLayerDeltasConvolutionTensor.Dispose();
        LayerDeltasConvolutionBuffer.Dispose();
        AdditionalReleases();
    }
    protected virtual void AdditionalReleases()
    {

    }

    public ConvolutionLayer(TensorSize InputTensor, int NumFilters, int ZeroPadding, int FilterSize, int Stride, ActivationType ActivationFunction = ActivationType.Tanh)
        : base
        (
            TensorInputFormat.Texture,
            InputTensor,
            new TensorSize
            (
                 Mathf.CeilToInt((InputTensor.Width + ZeroPadding * 2 - (FilterSize - 1)) / (float)Stride),
                 Mathf.CeilToInt((InputTensor.Height + ZeroPadding * 2 - (FilterSize - 1)) / (float)Stride),
                 NumFilters
            )
        )
    {
        this.ActivationFunction = ActivationFunction;
        if (ActivationType.Sigmoid == ActivationFunction)
            Debug.LogWarning("Sigmoid is problematic for convolution neural networks (its alright for FFNN though), see https://kharshit.github.io/blog/2018/04/20/don't-use-sigmoid-neural-nets");

        this.FilterSize = FilterSize;
        this.Stride = Stride;
        this.ZeroPadding = ZeroPadding;

        InitializeInput();

        Output = new Matrix(OutputTensor.Depth, OutputTensor.Width * OutputTensor.Height);
        Bias = new Matrix(OutputTensor.Depth, OutputTensor.Width * OutputTensor.Height);
        for (int i = 0; i < Bias.Buffer.Length; i++)
        {
            Bias.Buffer[i] = UnityEngine.Random.Range(-1f, 1f);
        }
        Bias.SetData();

        FilterTensor = new Matrix(OutputTensor.Depth, FilterSize * FilterSize * InputTensor.Depth);
        for (int i = 0; i < FilterTensor.Buffer.Length; i++)
        {
            FilterTensor.Buffer[i] = UnityEngine.Random.Range(-1f, 1f);
        }
        FilterTensor.SetData();

        InitializeKernelIndices();

        ConvolutionTensor = new Matrix(FilterSize * FilterSize * InputTensor.Depth, OutputTensor.Width * OutputTensor.Height);
        ConvolutionTensor.SetData();
        ConvolutionBuffer = new ComputeBuffer(ConvolutionTensor.Buffer.Length, sizeof(float));
        ConvolutionBuffer.SetData(ConvolutionTensor.Buffer);

        FilterOperation.SetInt("FilterWidth", FilterSize);
        FilterOperation.SetInt("FilterHeight", FilterSize);
        FilterOperation.SetInt("TextureWidth", InputTensor.Width);
        FilterOperation.SetInt("TextureHeight", InputTensor.Height);
        FilterOperation.SetInt("ZeroPadding", ZeroPadding);
        FilterOperation.SetInt("Stride", Stride);
        for (int i = 0; i < FilterKernelIndex.Length; i++)
        {
            SetFilterInput(i);
            FilterOperation.SetBuffer(FilterKernelIndex[i], "ConvolutionTensor", ConvolutionBuffer);
        }

        //FOR BACKPROP
        BackpropWeightOperation = UnityEngine.Object.Instantiate(Resources.Load<ComputeShader>("FilterOperation"));
        DialateKernelIndex = FilterOperation.FindKernel("DialateMatrix");

        //FOR BACKPROP OF WEIGHT DELTAS
        int DialatedWidth = OutputTensor.Width + (OutputTensor.Width - 1) * (Stride - 1);
        int DialatedHeight = OutputTensor.Height + (OutputTensor.Height - 1) * (Stride - 1);
        WeightsDialatedOutputBuffer = new ComputeBuffer(OutputTensor.Depth * DialatedWidth * DialatedHeight * InputTensor.Depth, sizeof(float));
        WeightsNonDialatedOutputBuffer = new ComputeBuffer(OutputTensor.Depth * OutputTensor.Width * OutputTensor.Height, sizeof(float));

        BackpropWeightOperation.SetBuffer(DialateKernelIndex, "DialatedOutput", WeightsDialatedOutputBuffer);
        BackpropWeightOperation.SetBuffer(DialateKernelIndex, "NonDialatedOutput", WeightsNonDialatedOutputBuffer);

        BackpropWeightOperation.SetInt("Depth", InputTensor.Depth);
        BackpropWeightOperation.SetInt("OutputDepth", OutputTensor.Depth);
        BackpropWeightOperation.SetInt("Stride", 1);
        BackpropWeightOperation.SetInt("DStride", Stride - 1);
        BackpropWeightOperation.SetInt("ConvolutionWidth", FilterSize);
        BackpropWeightOperation.SetInt("ConvolutionHeight", FilterSize);
        BackpropWeightOperation.SetInt("NonDialatedWidth", OutputTensor.Width);
        BackpropWeightOperation.SetInt("NonDialatedHeight", OutputTensor.Height);

        BackpropWeightOperation.SetInt("TextureWidth", InputTensor.Width);
        BackpropWeightOperation.SetInt("TextureHeight", InputTensor.Height);
        BackpropWeightOperation.SetInt("FilterWidth", DialatedWidth);
        BackpropWeightOperation.SetInt("FilterHeight", DialatedHeight);
        BackpropWeightOperation.SetInt("ZeroPadding", ZeroPadding);

        WeightsConvolutionTensor = new Matrix(DialatedWidth * DialatedHeight * InputTensor.Depth, FilterSize * FilterSize * InputTensor.Depth);
        WeightsConvolutionTensor.SetData();
        WeightsConvolutionBuffer = new ComputeBuffer(WeightsConvolutionTensor.Buffer.Length, sizeof(float));
        for (int i = 0; i < BackPropFilterKernelIndex.Length; i++)
        {
            SetBackPropWeightFilterInput(i);
            BackpropWeightOperation.SetBuffer(BackPropFilterKernelIndex[i], "ConvolutionTensor", WeightsConvolutionBuffer);
        }

        //BACKPROP FOR LAYER DELTAS
        BackPropLayerDeltasOperation = UnityEngine.Object.Instantiate(Resources.Load<ComputeShader>("FilterOperation"));
        LayerDeltasNonDialatedOutputBuffer = new ComputeBuffer(OutputTensor.Width * OutputTensor.Height * OutputTensor.Depth, sizeof(float));
        DialatedWidth = OutputTensor.Width + (OutputTensor.Width - 1) * (Stride - 1);
        DialatedHeight = OutputTensor.Height + (OutputTensor.Height - 1) * (Stride - 1);
        LayerDeltasDialatedOutputBuffer = new ComputeBuffer(DialatedWidth * DialatedHeight * OutputTensor.Depth, sizeof(float));

        int BackpropZeroPadding = (FilterSize - 1) - ZeroPadding;
        LayerDeltaConvoTensor = new TensorSize()
        {
            Depth = OutputTensor.Depth,
            Width = Mathf.CeilToInt((OutputTensor.Width + BackpropZeroPadding * 2 - (FilterSize - 1)) / (float)Stride),
            Height = Mathf.CeilToInt((OutputTensor.Height + BackpropZeroPadding * 2 - (FilterSize - 1)) / (float)Stride)
        };

        BackPropLayerDeltasOperation.SetBuffer(DialateKernelIndex, "DialatedOutput", LayerDeltasDialatedOutputBuffer);
        BackPropLayerDeltasOperation.SetBuffer(DialateKernelIndex, "NonDialatedOutput", LayerDeltasNonDialatedOutputBuffer);

        BackPropLayerDeltasOperation.SetInt("Depth", LayerDeltaConvoTensor.Depth);
        BackPropLayerDeltasOperation.SetInt("OutputDepth", InputTensor.Depth);
        BackPropLayerDeltasOperation.SetInt("Stride", 1);
        BackPropLayerDeltasOperation.SetInt("DStride", Stride - 1);
        BackPropLayerDeltasOperation.SetInt("ConvolutionWidth", LayerDeltaConvoTensor.Width);
        BackPropLayerDeltasOperation.SetInt("ConvolutionHeight", LayerDeltaConvoTensor.Height);
        BackPropLayerDeltasOperation.SetInt("NonDialatedWidth", OutputTensor.Width);
        BackPropLayerDeltasOperation.SetInt("NonDialatedHeight", OutputTensor.Height);

        BackPropLayerDeltasOperation.SetInt("TextureWidth", OutputTensor.Width);
        BackPropLayerDeltasOperation.SetInt("TextureHeight", OutputTensor.Height);
        BackPropLayerDeltasOperation.SetInt("FilterWidth", FilterSize);
        BackPropLayerDeltasOperation.SetInt("FilterHeight", FilterSize);
        BackPropLayerDeltasOperation.SetInt("ZeroPadding", BackpropZeroPadding);

        LayerDeltasConvolutionTensor = new Matrix(FilterSize * FilterSize * LayerDeltaConvoTensor.Depth, LayerDeltaConvoTensor.Width * LayerDeltaConvoTensor.Height);
        LayerDeltasConvolutionTensor.SetData();
        LayerDeltasConvolutionBuffer = new ComputeBuffer(LayerDeltasConvolutionTensor.Buffer.Length, sizeof(float));
        BackPropLayerDeltasConvolutionTensor = new ComputeBuffer(DialatedWidth * DialatedHeight * OutputTensor.Depth, sizeof(float));
        FilterMatrixKernel = new int[3]
        {
            FilterOperation.FindKernel("FilterOperationMatrix_8"),
            FilterOperation.FindKernel("FilterOperationMatrix_16"),
            FilterOperation.FindKernel("FilterOperationMatrix_32")
        };
        for (int i = 0; i < FilterMatrixKernel.Length; i++)
        {
            BackPropLayerDeltasOperation.SetBuffer(FilterMatrixKernel[i], "MatrixInput", BackPropLayerDeltasConvolutionTensor);
            BackPropLayerDeltasOperation.SetBuffer(FilterMatrixKernel[i], "ConvolutionTensor", LayerDeltasConvolutionBuffer);
        }
    }

    protected abstract void InitializeInput();
    protected abstract void InitializeKernelIndices();
    protected abstract void SetFilterInput(int Index);
    protected abstract void SetBackPropWeightFilterInput(int Index);
    protected abstract void PrepareInput();

    public override void ForwardProp()
    {
        PrepareInput();

        ThreadGroup Threading = CalculateNumThreadGroups(FilterKernelIndex, OutputTensor.Width * OutputTensor.Height);
        FilterOperation.Dispatch(Threading.FilterThreadGroupKernel, Threading.NumThreadGroups, FilterSize * FilterSize, InputTensor.Depth);
        ConvolutionBuffer.GetData(ConvolutionTensor.Buffer);
        ConvolutionTensor.SetData();
        PriorActivationOutput = FilterTensor * ConvolutionTensor;
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

    public override Network.WeightBiasDeltas Backprop(Matrix LayerDeltas)
    {
        Network.WeightBiasDeltas WeightBiasDeltas = new Network.WeightBiasDeltas();

        Matrix DerivedActivationOutput = new Matrix(PriorActivationOutput.Rows, PriorActivationOutput.Cols);
        DerivedActivationOutput.CopyValues(PriorActivationOutput.Buffer);
        switch (ActivationFunction)
        {
            case ActivationType.Tanh: DerivedActivationOutput.TanhActivation_Derivation(); break;
            case ActivationType.Sigmoid: DerivedActivationOutput.SigmoidActivation_Derivation(); break;
            case ActivationType.ReLU: DerivedActivationOutput.ReLUActivation_Derivation(); break;
            default: DerivedActivationOutput.TanhActivation_Derivation(); break;
        }
        Matrix GammaFilter = new Matrix(OutputTensor.Depth, OutputTensor.Width * OutputTensor.Height);
        Matrix.CWiseMultiply(LayerDeltas, DerivedActivationOutput, GammaFilter);
        GammaFilter.GetData();

        //CALCULATE DELTA WEIGHTS
        int DialatedWidth = OutputTensor.Width + (OutputTensor.Width - 1) * (Stride - 1);
        int DialatedHeight = OutputTensor.Height + (OutputTensor.Height - 1) * (Stride - 1);
        Matrix WeightDialatedGammerFilter = new Matrix(OutputTensor.Depth, DialatedWidth * DialatedHeight * InputTensor.Depth);
        WeightsDialatedOutputBuffer.SetData(WeightDialatedGammerFilter.SetData());
        WeightsNonDialatedOutputBuffer.SetData(GammaFilter.Buffer);
        BackpropWeightOperation.Dispatch(DialateKernelIndex, OutputTensor.Width / 8 + 1, OutputTensor.Height / 8 + 1, OutputTensor.Depth);
        WeightsDialatedOutputBuffer.GetData(WeightDialatedGammerFilter.Buffer);
        WeightDialatedGammerFilter.SetData();

        //PERFORM CONVOLUTION FOR GETTING WEIGHT DELTAS
        ThreadGroup Threading = CalculateNumThreadGroups(BackPropFilterKernelIndex, DialatedWidth * DialatedHeight);
        BackpropWeightOperation.Dispatch(Threading.FilterThreadGroupKernel, FilterSize * FilterSize, Threading.NumThreadGroups, InputTensor.Depth);
        WeightsConvolutionBuffer.GetData(WeightsConvolutionTensor.Buffer);
        WeightsConvolutionTensor.SetData();

        WeightBiasDeltas.WeightDeltas = WeightDialatedGammerFilter * WeightsConvolutionTensor;
        WeightBiasDeltas.BiasDeltas = GammaFilter;

        //CALCULATE LAYER DELTAS
        DialatedWidth = OutputTensor.Width + (OutputTensor.Width - 1) * (Stride - 1);
        DialatedHeight = OutputTensor.Height + (OutputTensor.Height - 1) * (Stride - 1);
        Matrix LayerDeltasDialatedImage = new Matrix(OutputTensor.Depth, DialatedWidth * DialatedHeight);
        LayerDeltasDialatedOutputBuffer.SetData(LayerDeltasDialatedImage.SetData());
        LayerDeltasNonDialatedOutputBuffer.SetData(LayerDeltas.Buffer);
        BackPropLayerDeltasOperation.Dispatch(DialateKernelIndex, OutputTensor.Width / 8 + 1, OutputTensor.Height / 8 + 1, OutputTensor.Depth);
        LayerDeltasDialatedOutputBuffer.GetData(LayerDeltasDialatedImage.Buffer);
        LayerDeltasDialatedImage.SetData();

        //PERFORM CONVOLUTION FOR GETTING LAYER DELTAS
        Threading = CalculateNumThreadGroups(FilterMatrixKernel, DialatedWidth * DialatedHeight);
        BackPropLayerDeltasConvolutionTensor.SetData(LayerDeltasDialatedImage.Buffer);
        BackPropLayerDeltasOperation.Dispatch(Threading.FilterThreadGroupKernel, Threading.NumThreadGroups, FilterSize * FilterSize, OutputTensor.Depth);
        LayerDeltasDialatedOutputBuffer.GetData(LayerDeltasDialatedImage.Buffer);
        LayerDeltasDialatedImage.SetData();

        //GENERATE NEW WEIGHT MATRICES
        Matrix LayerDeltaFilter = new Matrix(InputTensor.Depth, FilterSize * FilterSize * OutputTensor.Depth);
        FilterTensor.GetData();
        int FilterVolume = FilterSize * FilterSize;
        for (int i = 0; i < InputTensor.Depth; i++)
        {
            for (int j = 0; j < OutputTensor.Depth; j++)
            {
                for (int x = 0, y = FilterVolume - 1; x < FilterVolume; x++, y--)
                {
                    LayerDeltaFilter.Buffer[InputTensor.Depth * (j * FilterVolume + x) + i] = FilterTensor.Buffer[OutputTensor.Depth * (i * FilterVolume + y) + j];
                }
            }
        }
        LayerDeltaFilter.SetData();

        this.LayerDeltas = LayerDeltaFilter * LayerDeltasConvolutionTensor;

        return WeightBiasDeltas;
    }

    public override void ApplyWeightBiasDeltas(Network.WeightBiasDeltas WeightBiasDeltas, float LearningRate)
    {
        FilterTensor.SubInPlace(WeightBiasDeltas.WeightDeltas * LearningRate);
        Bias.SubInPlace(WeightBiasDeltas.BiasDeltas * LearningRate);
    }

    public override void Render(int RenderIndex)
    {
        FilterOperation.SetInt("FilterIndex", RenderIndex);
        FilterOperation.SetInt("NumFeatureMaps", OutputTensor.Depth);
        FilterOperation.SetInt("RenderBufferLength", Output.Buffer.Length);
        RenderBuffer.SetData(Output.GetData());
        FilterOperation.Dispatch(GenerateTextureKernelIndex, OutputTensor.Width / 8 + 1, OutputTensor.Height / 8 + 1, 1);
    }
}

public class ConvolutionTextureLayer : ConvolutionLayer
{
    public ConvolutionTextureLayer(TensorSize InputTensor, int NumFilters, int ZeroPadding, int FilterSize, int Stride, ActivationType ActivationFunction = ActivationType.Tanh)
        : base(InputTensor, NumFilters, ZeroPadding, FilterSize, Stride, ActivationFunction)
    {
        DebugObject.name = "ConvolutionTextureLayer"; 
    }

    protected override void PrepareInput()
    {
        
    }


    protected override void InitializeInput()
    {
        Input = new RenderTexture(InputTensor.Width, InputTensor.Height, 8)
        {
            enableRandomWrite = true,
            filterMode = FilterMode.Point,
            anisoLevel = 1
        };
        ((RenderTexture)Input).Create();
    }

    protected override void SetFilterInput(int Index)
    {
        FilterOperation.SetTexture(FilterKernelIndex[Index], "TextureInput", (RenderTexture)Input);
    }

    protected override void SetBackPropWeightFilterInput(int Index)
    {
        BackpropWeightOperation.SetTexture(BackPropFilterKernelIndex[Index], "TextureInput", (RenderTexture)Input);
    }

    protected override void InitializeKernelIndices()
    {
        FilterKernelIndex = new int[3]
        {
            FilterOperation.FindKernel("FilterOperationTexture_8"),
            FilterOperation.FindKernel("FilterOperationTexture_16"),
            FilterOperation.FindKernel("FilterOperationTexture_32")
        };

        BackPropFilterKernelIndex = new int[3]
        {
            FilterOperation.FindKernel("BackPropFilterOperationTexture_8"),
            FilterOperation.FindKernel("BackPropFilterOperationTexture_16"),
            FilterOperation.FindKernel("BackPropFilterOperationTexture_32")
        };
    }

    public override void AssignInput(object InputToAssign)
    {
        Texture2D Texture = null;
        if (InputToAssign.GetType() == typeof(string))
        {
            Texture = new Texture2D(InputTensor.Width, InputTensor.Height);
            Texture.LoadImage(File.ReadAllBytes((string)InputToAssign));
        }
        else
        {
            Texture = (Texture2D)InputToAssign;
        }
        RenderTexture.active = (RenderTexture)Input;
        Graphics.Blit(Texture, (RenderTexture)Input);

        Resources.UnloadUnusedAssets();
    }
}

public class ConvolutionMatrixLayer : ConvolutionLayer
{
    private ComputeBuffer InputBuffer;

    protected override void AdditionalReleases()
    {
        InputBuffer.Dispose();
    }

    public ConvolutionMatrixLayer(TensorSize InputTensor, int NumFilters, int ZeroPadding, int FilterSize, int Stride, ActivationType ActivationFunction = ActivationType.Tanh)
        : base(InputTensor, NumFilters, ZeroPadding, FilterSize, Stride, ActivationFunction)
    {
        DebugObject.name = "ConvolutionMatrixLayer";
    }

    protected override void PrepareInput()
    {
        InputBuffer.SetData(((Matrix)Input).Buffer);
    }


    protected override void InitializeInput()
    {
        InputBuffer = new ComputeBuffer(InputTensor.Width * InputTensor.Height * InputTensor.Depth, sizeof(float));
    }

    protected override void SetFilterInput(int Index)
    {
        FilterOperation.SetBuffer(FilterKernelIndex[Index], "MatrixInput", InputBuffer);
    }
    protected override void SetBackPropWeightFilterInput(int Index)
    {
        BackpropWeightOperation.SetBuffer(BackPropFilterKernelIndex[Index], "MatrixInput", InputBuffer);
    }
    protected override void InitializeKernelIndices()
    {
        FilterKernelIndex = new int[3]
        {
            FilterOperation.FindKernel("FilterOperationMatrix_8"),
            FilterOperation.FindKernel("FilterOperationMatrix_16"),
            FilterOperation.FindKernel("FilterOperationMatrix_32")
        };

        BackPropFilterKernelIndex = new int[3]
        {
            FilterOperation.FindKernel("BackPropFilterOperationMatrix_8"),
            FilterOperation.FindKernel("BackPropFilterOperationMatrix_16"),
            FilterOperation.FindKernel("BackPropFilterOperationMatrix_32")
        };
    }

    public override void AssignInput(object InputToAssign)
    {
        Input = (Matrix)InputToAssign;
    }
}