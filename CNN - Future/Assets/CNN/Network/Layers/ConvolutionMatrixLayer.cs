using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using UnityEngine;
using UnityEngine.UI;

public class ConvolutionMatrixLayer : ConvolutionLayer
{
    private ComputeBuffer InputBuffer;

    protected override void AdditionalReleases()
    {
        InputBuffer.Dispose();
    }

    public ConvolutionMatrixLayer(TensorSize InputTensor, int NumFilters, int FilterSize, int Stride, int ZeroPadding, ActivationType ActivationFunction)
        : base(InputTensor, NumFilters, FilterSize, Stride, ZeroPadding, ActivationFunction)
    {
        DebugObject.name = "ConvolutonMatrixLayer";
    }

    protected override void InitializeInput()
    {
        InputBuffer = new ComputeBuffer(InputTensor.Width * InputTensor.Height * InputTensor.Depth, sizeof(float));
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

    protected override void SetFilterInput(int Index)
    {
        FilterOperation.SetBuffer(FilterKernelIndex[Index], "MatrixInput", InputBuffer);
    }

    protected override void SetBackPropWeightFilterInput(int Index)
    {
        BackpropWeightOperation.SetBuffer(BackPropFilterKernelIndex[Index], "MatrixInput", InputBuffer);
    }

    protected override void PrepareInput()
    {
        InputBuffer.SetData(((Matrix)Input).Buffer);
    }

    protected override void BackPropPrepareInput()
    {
        InputBuffer.SetData(((Matrix)Input).Buffer);
    }

    public override void AssignInput(object In)
    {
        Input = (Matrix)In;
    }
}
