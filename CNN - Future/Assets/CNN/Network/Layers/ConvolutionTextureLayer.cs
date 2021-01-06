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
    protected int[] BackPropFilterKernelIndex;

    private ComputeBuffer ConvolutionBuffer;
    private Matrix ConvolutionTensor;

    protected int DialateKernelIndex;
    protected ComputeShader BackpropWeightOperation;
    private Matrix WeightsConvolutionTensor;
    private ComputeBuffer WeightsConvolutionBuffer;
    private ComputeBuffer WeightsDialatedOutputBuffer;
    private ComputeBuffer WeightsNonDialatedOutputBuffer;

    private int[] FilterMatrixKernel; //For performing backprop convolutions
    private ComputeShader BackpropLayerDeltasOperation;
    private ComputeBuffer BackpropLayerDeltasConvoInputBuffer;
    private Matrix LayerDeltasConvolutionTensor;
    private TensorSize LayerDeltaConvoTensor;
    private ComputeBuffer LayerDeltasConvolutionBuffer;
    private ComputeBuffer LayerDeltasDialatedOutputBuffer;
    private ComputeBuffer LayerDeltasNonDialatedOutputBuffer;

    public Matrix FilterTensor;
    public Matrix Bias;

    public enum ActivationType
    {
        Tanh,
        Sigmoid,
        ReLU
    }
    protected ActivationType ActivationFunction;

    private int FilterSize;
    private int Stride;
    private int ZeroPadding;

    protected override void Release()
    {
        ConvolutionBuffer.Dispose();
        WeightsDialatedOutputBuffer.Dispose();
        WeightsNonDialatedOutputBuffer.Dispose();
        WeightsConvolutionBuffer.Dispose();
        LayerDeltasDialatedOutputBuffer.Dispose();
        LayerDeltasNonDialatedOutputBuffer.Dispose();
        BackpropLayerDeltasConvoInputBuffer.Dispose();
        LayerDeltasConvolutionBuffer.Dispose();
        AdditionalReleases();
    }
    protected virtual void AdditionalReleases() { }

    public override void LogParams()
    {
        WeightsConvolutionTensor.GetData();
        Debug.Log("Conv: [W]" + WeightsConvolutionTensor);
        Bias.GetData();
        Debug.Log("Conv: [B]" + Bias);
    }

    public ConvolutionLayer(TensorSize InputTensor, int NumFilters, int FilterSize, int Stride, int ZeroPadding, ActivationType ActivationFunction)
        : base(TensorInputFormat.Texture,
               InputTensor,
               new TensorSize(
                   Mathf.CeilToInt((InputTensor.Width + ZeroPadding * 2 - (FilterSize - 1)) / (float)Stride), //TODO:: case when filter is greater than input tensor
                   Mathf.CeilToInt((InputTensor.Height + ZeroPadding * 2 - (FilterSize - 1)) / (float)Stride),
                   NumFilters))
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
        //Creating the Convolution Buffer
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

        //For backprop
        BackpropWeightOperation = UnityEngine.Object.Instantiate(Resources.Load<ComputeShader>("FilterOperation"));
        DialateKernelIndex = FilterOperation.FindKernel("DialateMatrix");

        //Weights
        int DialatedWidth = OutputTensor.Width + (OutputTensor.Width - 1) * (Stride - 1);
        int DialatedHeight = OutputTensor.Height + (OutputTensor.Height - 1) * (Stride - 1);
        WeightsDialatedOutputBuffer = new ComputeBuffer(OutputTensor.Depth * DialatedWidth * DialatedHeight * InputTensor.Depth, sizeof(float));
        WeightsNonDialatedOutputBuffer = new ComputeBuffer(OutputTensor.Width * OutputTensor.Height * OutputTensor.Depth, sizeof(float));

        BackpropWeightOperation.SetBuffer(DialateKernelIndex, "DialatedOutput", WeightsDialatedOutputBuffer);
        BackpropWeightOperation.SetBuffer(DialateKernelIndex, "NonDialatedOutput", WeightsNonDialatedOutputBuffer);

        BackpropWeightOperation.SetInt("Depth", InputTensor.Depth);
        BackpropWeightOperation.SetInt("OutputDepth", OutputTensor.Depth);
        BackpropWeightOperation.SetInt("Stride", 1); //Stride for convolution
        BackpropWeightOperation.SetInt("DStride", Stride - 1); //Dialation Stride
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

        BackpropLayerDeltasOperation = UnityEngine.Object.Instantiate(Resources.Load<ComputeShader>("FilterOperation"));

        //LayerDeltas
        DialatedWidth = OutputTensor.Width + (OutputTensor.Width - 1) * (Stride - 1);
        DialatedHeight = OutputTensor.Height + (OutputTensor.Height - 1) * (Stride - 1);
        LayerDeltasDialatedOutputBuffer = new ComputeBuffer(DialatedWidth * DialatedHeight * OutputTensor.Depth, sizeof(float));
        LayerDeltasNonDialatedOutputBuffer = new ComputeBuffer(OutputTensor.Width * OutputTensor.Height * OutputTensor.Depth, sizeof(float));

        int BackpropZeroPadding = (FilterSize - 1) - ZeroPadding;
        LayerDeltaConvoTensor = new TensorSize()
        {
            Depth = OutputTensor.Depth,
            Width = Mathf.CeilToInt((OutputTensor.Width + BackpropZeroPadding * 2 - (FilterSize - 1)) / (float)Stride),
            Height = Mathf.CeilToInt((OutputTensor.Height + BackpropZeroPadding * 2 - (FilterSize - 1)) / (float)Stride)
        };

        BackpropLayerDeltasOperation.SetBuffer(DialateKernelIndex, "DialatedOutput", LayerDeltasDialatedOutputBuffer);
        BackpropLayerDeltasOperation.SetBuffer(DialateKernelIndex, "NonDialatedOutput", LayerDeltasNonDialatedOutputBuffer);

        BackpropLayerDeltasOperation.SetInt("Depth", LayerDeltaConvoTensor.Depth);
        BackpropLayerDeltasOperation.SetInt("OutputDepth", OutputTensor.Depth);
        BackpropLayerDeltasOperation.SetInt("Stride", 1); //Stride for convolution
        BackpropLayerDeltasOperation.SetInt("DStride", Stride - 1); //Dialation Stride
        BackpropLayerDeltasOperation.SetInt("ConvolutionWidth", LayerDeltaConvoTensor.Width);
        BackpropLayerDeltasOperation.SetInt("ConvolutionHeight", LayerDeltaConvoTensor.Height);
        BackpropLayerDeltasOperation.SetInt("NonDialatedWidth", OutputTensor.Width);
        BackpropLayerDeltasOperation.SetInt("NonDialatedHeight", OutputTensor.Height);

        BackpropLayerDeltasOperation.SetInt("TextureWidth", OutputTensor.Width);
        BackpropLayerDeltasOperation.SetInt("TextureHeight", OutputTensor.Height);
        BackpropLayerDeltasOperation.SetInt("FilterWidth", FilterSize);
        BackpropLayerDeltasOperation.SetInt("FilterHeight", FilterSize);
        BackpropLayerDeltasOperation.SetInt("ZeroPadding", BackpropZeroPadding);

        LayerDeltasConvolutionTensor = new Matrix(FilterSize * FilterSize * LayerDeltaConvoTensor.Depth, LayerDeltaConvoTensor.Width * LayerDeltaConvoTensor.Height);
        LayerDeltasConvolutionTensor.SetData();
        LayerDeltasConvolutionBuffer = new ComputeBuffer(LayerDeltasConvolutionTensor.Buffer.Length, sizeof(float));
        BackpropLayerDeltasConvoInputBuffer = new ComputeBuffer(DialatedWidth * DialatedHeight * OutputTensor.Depth, sizeof(float));
        FilterMatrixKernel = new int[3]
        {
            FilterOperation.FindKernel("FilterOperationMatrix_8"),
            FilterOperation.FindKernel("FilterOperationMatrix_16"),
            FilterOperation.FindKernel("FilterOperationMatrix_32")
        };
        for (int i = 0; i < FilterMatrixKernel.Length; i++)
        {
            BackpropLayerDeltasOperation.SetBuffer(FilterMatrixKernel[i], "MatrixInput", BackpropLayerDeltasConvoInputBuffer);
            BackpropLayerDeltasOperation.SetBuffer(FilterMatrixKernel[i], "ConvolutionTensor", LayerDeltasConvolutionBuffer);
        }
    }

    protected abstract void InitializeInput(); //Initialize input object to be whats needed for GPU processing
    protected abstract void InitializeKernelIndices(); //Initialize list of kernels for GPU processing
    protected abstract void SetFilterInput(int Index); //Set input for GPU Filter process for every kernel at index
    protected abstract void SetBackPropWeightFilterInput(int Index); //Set input for GPU Filter process for every kernel at index
    protected abstract void PrepareInput(); //Prepare the input for forward propagation if needed
    protected abstract void BackPropPrepareInput(); //Prepare the input for backward propagation if needed

    private struct ThreadGroup
    {
        public int NumThreadGroups;
        public int FilterThreadGroupKernel;
    }

    private ThreadGroup CalculatNumThreadGroups(int[] FilterKernelIndex, int TensorSize)
    {
        ThreadGroup Result = new ThreadGroup();
        Result.FilterThreadGroupKernel = FilterKernelIndex[0];
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
        if (TensorSize % ThreadGroups != 0) Result.NumThreadGroups++; //Add 1 to fit in the remainder
        return Result;
    }
    public override void ForwardProp()
    {
        PrepareInput();

        ThreadGroup Threading = CalculatNumThreadGroups(FilterKernelIndex, OutputTensor.Width * OutputTensor.Height);
        FilterOperation.Dispatch(Threading.FilterThreadGroupKernel, Threading.NumThreadGroups, FilterSize * FilterSize, InputTensor.Depth); //3 is because depth of 3 for RGB colour
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

    public override void Render(int RenderIndex)
    {
        FilterOperation.SetInt("FilterIndex", RenderIndex);
        FilterOperation.SetInt("NumFeatureMaps", OutputTensor.Depth);
        FilterOperation.SetInt("RenderBufferLength", Output.Buffer.Length);
        RenderBuffer.SetData(Output.GetData());
        FilterOperation.Dispatch(GenerateTextureKernelIndex, OutputTensor.Width / 8 + 1, OutputTensor.Height / 8 + 1, 1); //we add 1 here to also fill in remainder
    }

    public override Network.WeightBiasDeltas Backprop(Matrix LayerDeltas)
    {
        Network.WeightBiasDeltas WeightBiasDeltas = new Network.WeightBiasDeltas();

        Matrix DerivedActivationOutput = new Matrix(PriorActivationOutput.Rows, PriorActivationOutput.Cols);
        DerivedActivationOutput.CopyValues(PriorActivationOutput.Buffer);
        switch (ActivationFunction) //TODO implement max derivation
        {
            case ActivationType.Tanh: DerivedActivationOutput.TanhActivation_Derivation(); break;
            case ActivationType.Sigmoid: DerivedActivationOutput.SigmoidActivation_Derivation(); break;
            case ActivationType.ReLU: DerivedActivationOutput.ReLUActivation_Derivation(); break;
            default: DerivedActivationOutput.TanhActivation_Derivation(); break;
        }
        Matrix GammaFilter = new Matrix(OutputTensor.Depth, OutputTensor.Width * OutputTensor.Height);
        Matrix.CWiseMultiply(LayerDeltas, DerivedActivationOutput, GammaFilter);
        GammaFilter.GetData();

        //https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-fb2f2efc4faa
        //Calculate delta weights using convolution

        //Dialate GammaFilter
        int DialatedWidth = OutputTensor.Width + (OutputTensor.Width - 1) * (Stride - 1);
        int DialatedHeight = OutputTensor.Height + (OutputTensor.Height - 1) * (Stride - 1);
        Matrix WeightDialatedGammaFilter = new Matrix(OutputTensor.Depth, DialatedWidth * DialatedHeight * InputTensor.Depth);
        WeightsDialatedOutputBuffer.SetData(WeightDialatedGammaFilter.SetData()); //Set stuff with 0s
        WeightsNonDialatedOutputBuffer.SetData(GammaFilter.Buffer); //Set gamma filter
        BackpropWeightOperation.Dispatch(DialateKernelIndex, OutputTensor.Width / 8 + 1, OutputTensor.Height / 8 + 1, OutputTensor.Depth);
        WeightsDialatedOutputBuffer.GetData(WeightDialatedGammaFilter.Buffer);
        WeightDialatedGammaFilter.SetData();

        //Perform convolution
        ThreadGroup Threading = CalculatNumThreadGroups(BackPropFilterKernelIndex, DialatedWidth * DialatedHeight);
        BackpropWeightOperation.Dispatch(Threading.FilterThreadGroupKernel, FilterSize * FilterSize, Threading.NumThreadGroups, InputTensor.Depth); //3 is because depth of 3 for RGB colour
        WeightsConvolutionBuffer.GetData(WeightsConvolutionTensor.Buffer);
        WeightsConvolutionTensor.SetData();

        WeightBiasDeltas.WeightDeltas = WeightDialatedGammaFilter * WeightsConvolutionTensor;
        WeightBiasDeltas.BiasDeltas = GammaFilter;

        //Calculate layer delta using convolution
        //https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710

        //Dialate Layer Deltas
        DialatedWidth = OutputTensor.Width + (OutputTensor.Width - 1) * (Stride - 1);
        DialatedHeight = OutputTensor.Height + (OutputTensor.Height - 1) * (Stride - 1);
        Matrix LayerDeltasDialatedImage = new Matrix(OutputTensor.Depth, DialatedWidth * DialatedHeight);
        LayerDeltasDialatedOutputBuffer.SetData(LayerDeltasDialatedImage.SetData());
        LayerDeltasNonDialatedOutputBuffer.SetData(LayerDeltas.GetData());
        BackpropLayerDeltasOperation.Dispatch(DialateKernelIndex, OutputTensor.Width / 8 + 1, OutputTensor.Height / 8 + 1, OutputTensor.Depth);
        LayerDeltasDialatedOutputBuffer.GetData(LayerDeltasDialatedImage.Buffer);
        LayerDeltasDialatedImage.SetData();

        //Perform convolution
        Threading = CalculatNumThreadGroups(FilterMatrixKernel, DialatedHeight * DialatedHeight);
        BackpropLayerDeltasConvoInputBuffer.SetData(LayerDeltasDialatedImage.Buffer);
        BackpropLayerDeltasOperation.Dispatch(Threading.FilterThreadGroupKernel, Threading.NumThreadGroups, FilterSize * FilterSize, OutputTensor.Depth);
        LayerDeltasConvolutionBuffer.GetData(LayerDeltasConvolutionTensor.Buffer);
        LayerDeltasConvolutionTensor.SetData();

        //Generate new weight matrices
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

        //Convolute for layer deltas
        this.LayerDeltas = LayerDeltaFilter * LayerDeltasConvolutionTensor;

        return WeightBiasDeltas;
    }

    public override void ApplyWeightBiasDeltas(Network.WeightBiasDeltas WeightBiasDeltas, float LearningRate)
    {
        FilterTensor.SubInPlace(WeightBiasDeltas.WeightDeltas * LearningRate);
        Bias.SubInPlace(WeightBiasDeltas.BiasDeltas * LearningRate);
    }
}

public class ConvolutionTextureLayer : ConvolutionLayer
{
    public ConvolutionTextureLayer(TensorSize InputTensor, int NumFilters, int FilterSize, int Stride, int ZeroPadding, ActivationType ActivationFunction)
        : base(InputTensor, NumFilters, FilterSize, Stride, ZeroPadding, ActivationFunction)
    {
        DebugObject.name = "ConvolutonTextureLayer";
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

    protected override void SetFilterInput(int Index)
    {
        FilterOperation.SetTexture(FilterKernelIndex[Index], "TextureInput", (RenderTexture)Input);
    }

    protected override void SetBackPropWeightFilterInput(int Index)
    {
        BackpropWeightOperation.SetTexture(BackPropFilterKernelIndex[Index], "TextureInput", (RenderTexture)Input);
    }

    protected override void PrepareInput() { }
    protected override void BackPropPrepareInput() { }

    public override void AssignInput(object In)
    {
        Texture2D Texture = null;
        if (In.GetType() == typeof(string))
        {
            Texture = new Texture2D(InputTensor.Width, InputTensor.Height);
            Texture.LoadImage(File.ReadAllBytes((string)In));
        }
        else
        {
            Texture = (Texture2D)In;
        }
        RenderTexture.active = (RenderTexture)Input;
        Graphics.Blit(Texture, (RenderTexture)Input);

        Resources.UnloadUnusedAssets(); //Release previous texture from memory and stop unity caching it, there are 19000 images after all
    }
}
