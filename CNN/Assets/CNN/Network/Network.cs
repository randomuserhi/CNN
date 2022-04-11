using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

using UnityEngine;

public static class CostFunctions
{
    public static Matrix MeanSquaredError(Matrix Output, Matrix Expected)
    {
        Matrix Cost = new Matrix(Output.Rows, Output.Cols);
        for (int i = 0; i < Output.Buffer.Length; i++)
        {
            float Val = Output.Buffer[i] - Expected.Buffer[i];
            Cost.Buffer[i] = Val * Val * 0.5f;
        }
        Cost.SetData();
        return Cost;
    }

    public static Matrix SquaredError(Matrix Output, Matrix Expected)
    {
        Matrix Cost = new Matrix(Output.Rows, Output.Cols);
        for (int i = 0; i < Output.Buffer.Length; i++)
        {
            float Val = Output.Buffer[i] - Expected.Buffer[i];
            Cost.Buffer[i] = Val * Val;
        }
        Cost.SetData();
        return Cost;
    }

    public static Matrix SoftMax(Matrix Output)
    {
        Matrix SoftMax = new Matrix(Output.Rows, Output.Cols);
        float Sum = Output.Buffer.Sum(V => Mathf.Exp(V));
        for (int i = 0; i < Output.Buffer.Length; i++)
        {
            SoftMax.Buffer[i] = Mathf.Exp(Output.Buffer[i]) / Sum;
        }
        SoftMax.SetData();
        return SoftMax;
    }
    public static Matrix SoftMaxCrossEntropyLoss(Matrix Output, Matrix Expected)
    {
        Matrix Cost = new Matrix(Output.Rows, Output.Cols);
        float Sum = Output.Buffer.Sum(V => Mathf.Exp(V));
        for (int i = 0; i < Output.Buffer.Length; i++)
        {
            float SoftMax = Mathf.Exp(Output.Buffer[i]) / Sum;
            Cost.Buffer[i] = -(Expected.Buffer[i] * Mathf.Log(SoftMax));
        }
        Cost.SetData();
        return Cost;
    }
}

public static class CostFunctionDerivations
{
    public static Matrix MeanSquaredError(Matrix Output, Matrix Expected)
    {
        Matrix Derivations = new Matrix(Output.Rows, Output.Cols);
        for (int i = 0; i < Output.Buffer.Length; i++)
        {
            Derivations.Buffer[i] = Output.Buffer[i] - Expected.Buffer[i];
        }
        Derivations.SetData();
        return Derivations;
    }

    public static Matrix SquaredError(Matrix Output, Matrix Expected)
    {
        Matrix Derivations = new Matrix(Output.Rows, Output.Cols);
        for (int i = 0; i < Output.Buffer.Length; i++)
        {
            Derivations.Buffer[i] = 2 * (Output.Buffer[i] - Expected.Buffer[i]);
        }
        Derivations.SetData();
        return Derivations;
    }

    public static Matrix SoftMaxCrossEntropyLoss(Matrix Output, Matrix Expected)
    {
        Matrix Derivations = new Matrix(Output.Rows, Output.Cols);
        float Sum = Output.Buffer.Sum(V => Mathf.Exp(V));
        float ExpectedSum = Expected.Buffer.Sum();
        for (int i = 0; i < Output.Buffer.Length; i++)
        {
            float SoftMax = Mathf.Exp(Output.Buffer[i]) / Sum;
            Derivations.Buffer[i] = SoftMax * ExpectedSum - Expected.Buffer[i];
        }
        Derivations.SetData();
        return Derivations;
    }
}

public class Network
{
    public enum CostFunctionType
    {
        SquaredError,
        MeanSquaredError,
        SoftMaxCrossEntropy
    }
    public CostFunctionType CostFunc = CostFunctionType.MeanSquaredError;

    public List<Layer> Model = new List<Layer>();

    public void Dispose()
    {
        for (int i = 0; i < Model.Count; i++)
        {
            Model[i].Dispose();
        }
    }

    public Network()
    {
        
    }
    public static int GetIndex(float[] Buffer)
    {
        var (Number, Index) = Buffer.Select((n, i) => (n, i)).Max();
        return Index + 1;
    }

    public string Export(string FilePath, string FileName)
    {
        byte[] GetBytes(Matrix Weights, Matrix Bias)
        {
            byte[] WeightBiasBytes = new byte[Weights.Buffer.Length * 4 + Bias.Buffer.Length * 4];

            Buffer.BlockCopy(Weights.GetData(), 0, WeightBiasBytes, 0, Weights.Buffer.Length * 4);
            Buffer.BlockCopy(Bias.GetData(), 0, WeightBiasBytes, Weights.Buffer.Length * 4, Bias.Buffer.Length * 4);

            return WeightBiasBytes;
        }

        List<byte> Data = new List<byte>();
        for (int i = 0; i < Model.Count; i++)
        {
            ConvolutionLayer Convolution = Model[i] as ConvolutionLayer;
            if (Convolution != null)
            {
                Data.AddRange(GetBytes(Convolution.FilterTensor, Convolution.Bias));
                continue;
            }
            DenseLayer Dense = Model[i] as DenseLayer;
            if (Dense != null)
            {
                Data.AddRange(GetBytes(Dense.Weights, Dense.Bias));
                continue;
            }
        }
        string SavePath = FilePath + Path.DirectorySeparatorChar + FileName + ".arby";
        File.WriteAllBytes(SavePath, Data.ToArray());
        return SavePath;
    }

    public void Import(string FilePath)
    {
        void ParseBytes(byte[] Bytes, ref int ReadIndex, Matrix Weights, Matrix Bias)
        {
            Buffer.BlockCopy(Bytes, ReadIndex, Weights.Buffer, 0, Weights.Buffer.Length * 4);
            ReadIndex += Weights.Buffer.Length * 4;
            Weights.SetData();
            Buffer.BlockCopy(Bytes, ReadIndex, Bias.Buffer, 0, Bias.Buffer.Length * 4);
            ReadIndex += Bias.Buffer.Length * 4;
            Bias.SetData();
        }

        byte[] Data = File.ReadAllBytes(FilePath);
        int ReadDataIndex = 0;
        for (int i = 0; i < Model.Count; i++)
        {
            ConvolutionLayer Convolution = Model[i] as ConvolutionLayer;
            if (Convolution != null)
            {
                ParseBytes(Data, ref ReadDataIndex, Convolution.FilterTensor, Convolution.Bias);
                continue;
            }
            DenseLayer Dense = Model[i] as DenseLayer;
            if (Dense != null)
            {
                ParseBytes(Data, ref ReadDataIndex, Dense.Weights, Dense.Bias);
                continue;
            }
        }
    }

    public void AssignInput(object Input)
    {
        Model[0].AssignInput(Input);
    }

    public Matrix ForwardPropagate(object Input, bool AssignInput = true)
    {
        if (Model.Count == 0)
        {
            Debug.LogError("Your a fucking idiot!");
            return null;
        }

        if (AssignInput)
        {
            Texture2D Texture = new Texture2D(2, 2);
            Texture.LoadImage(File.ReadAllBytes((string)Input));
            Model[0].AssignInput(Texture);
            CNN.Image.texture = Texture;
            Resources.UnloadUnusedAssets();
        }
        for (int i = 1; i < Model.Count; i++)
        {
            Model[i - 1].ForwardProp();
            Model[i].Input = Model[i - 1].Output;
        }
        Model[Model.Count - 1].ForwardProp();
        return Model[Model.Count - 1].Output;
    }

    public struct WeightBiasDeltas
    {
        public Matrix WeightDeltas;
        public Matrix BiasDeltas;
    }

    public struct BackPropagationEvaluation
    {
        public float ErrorCost;
        public WeightBiasDeltas[] Deltas;
        public Matrix Output;
        public Matrix Expected;

        public BackPropagationEvaluation(int ModelSize)
        {
            ErrorCost = 0;
            Output = null;
            Expected = null;
            Deltas = new WeightBiasDeltas[ModelSize];
        }

        public static BackPropagationEvaluation Sum(BackPropagationEvaluation[] Evaluations, int ModelSize)
        {
            BackPropagationEvaluation Sum = new BackPropagationEvaluation(ModelSize);
            Sum.ErrorCost = Evaluations[0].ErrorCost;
            for (int i = 0; i < Evaluations[0].Deltas.Length; i++)
            {
                if (Evaluations[0].Deltas[i].WeightDeltas == null) continue;
                Sum.Deltas[i].WeightDeltas = new Matrix(Evaluations[0].Deltas[i].WeightDeltas);
                Sum.Deltas[i].BiasDeltas = new Matrix(Evaluations[0].Deltas[i].BiasDeltas);
            }
            for (int j = 1; j < Evaluations.Length; j++)
            {
                Sum.ErrorCost += Evaluations[j].ErrorCost;
                for (int i = 0; i < Evaluations[j].Deltas.Length; i++)
                {
                    if (Evaluations[j].Deltas[i].WeightDeltas == null) continue;
                    Sum.Deltas[i].WeightDeltas.AddInPlace(Evaluations[j].Deltas[i].WeightDeltas);
                    Sum.Deltas[i].BiasDeltas.AddInPlace(Evaluations[j].Deltas[i].BiasDeltas);
                }
            }
            return Sum;
        }
    }

    public void ApplyWeightBiasDeltas(BackPropagationEvaluation BackpropEvaluation, float LearningRate)
    {
        for (int i = 0; i < Model.Count; i++)
        {
            Model[i].ApplyWeightBiasDeltas(BackpropEvaluation.Deltas[i], LearningRate);
        }
    }

    public BackPropagationEvaluation Backpropagation(object Input, Matrix Expected, bool AssignInput = true)
    {
        BackPropagationEvaluation BackPropagationEvaluation = new BackPropagationEvaluation(Model.Count);

        BackPropagationEvaluation.Output = ForwardPropagate(Input, AssignInput);
        BackPropagationEvaluation.Expected = Expected;
        switch (CostFunc)
        {
            case CostFunctionType.MeanSquaredError: BackPropagationEvaluation.ErrorCost = CostFunctions.MeanSquaredError(Model[Model.Count - 1].Output, Expected).Buffer.Sum(); break;
            case CostFunctionType.SquaredError: BackPropagationEvaluation.ErrorCost = CostFunctions.SquaredError(Model[Model.Count - 1].Output, Expected).Buffer.Sum(); break;
            case CostFunctionType.SoftMaxCrossEntropy: BackPropagationEvaluation.ErrorCost = CostFunctions.SoftMaxCrossEntropyLoss(Model[Model.Count - 1].Output, Expected).Buffer.Sum(); break;
            default: BackPropagationEvaluation.ErrorCost = CostFunctions.MeanSquaredError(Model[Model.Count - 1].Output, Expected).Buffer.Sum(); break;
        }

        Matrix CostDerivations;
        switch (CostFunc)
        {
            case CostFunctionType.MeanSquaredError: CostDerivations = CostFunctionDerivations.MeanSquaredError(Model[Model.Count - 1].Output, Expected); break;
            case CostFunctionType.SquaredError: CostDerivations = CostFunctionDerivations.SquaredError(Model[Model.Count - 1].Output, Expected); break;
            case CostFunctionType.SoftMaxCrossEntropy: CostDerivations = CostFunctionDerivations.SoftMaxCrossEntropyLoss(Model[Model.Count - 1].Output, Expected); break;
            default: CostDerivations = CostFunctionDerivations.MeanSquaredError(Model[Model.Count - 1].Output, Expected); break;
        }

        BackPropagationEvaluation.Deltas[Model.Count - 1] = Model[Model.Count - 1].Backprop(CostDerivations);
        for (int i = Model.Count - 2; i >= 0; i--)
        {
            BackPropagationEvaluation.Deltas[i] = Model[i].Backprop(Model[i + 1].LayerDeltas);
        }

        return BackPropagationEvaluation;
    }

    public void Render(int Index)
    {
        for (int i = 0; i < Model.Count; i++)
        {
            Model[i].Render(Index);
        }

    }
}
