using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using UnityEngine;

public abstract class DataSet
{
    public Matrix[] ExpectedClassificationValues;
    public List<Data> TrainingSet = new List<Data>();
    public List<Data> TestSet = new List<Data>();

    public struct Data
    {
        public object Input;
        public int ExpectedOutputIndex;
    }

    public readonly string DataSetFilePath;
    public readonly string DataSetCfgFileLocation;
    public readonly string ClassNamesFileLocation;

    public DataSet(string DataSetFilePath, string ClassNamesFileLocation, string DataSetCfgFileLocation)
    {
        this.DataSetFilePath = DataSetFilePath;
        this.ClassNamesFileLocation = ClassNamesFileLocation;
        this.DataSetCfgFileLocation = DataSetCfgFileLocation;
    }

    public void ShuffleData()
    {
        TrainingSet = TrainingSet.OrderBy(a => UnityEngine.Random.value).ToList();
        TestSet = TestSet.OrderBy(a => UnityEngine.Random.value).ToList();
    }
}

public class ImageDataSet : DataSet
{
    public readonly string[] ClassNames;
    public TensorSize InputDataSize;
    public TensorSize OutputDataSize;

    //NOTE:: ASSUMES CLASS INDICES ARE 1 BASED NOT 0 BASED
    public ImageDataSet(TensorSize InputDataSize, string DataSetFilePath, string ClassNamesFileLocation, string DataSetCfgFileLocation) : base(DataSetFilePath, ClassNamesFileLocation, DataSetCfgFileLocation)
    {
        this.InputDataSize = InputDataSize;
        ClassNames = File.ReadLines(ClassNamesFileLocation).ToString().Split(',');
        string[] DataSet = File.ReadAllLines(DataSetCfgFileLocation);

        int HighestIndex = 0;
        for (int i = 0; i < DataSet.Length; i++)
        {
            string[] Parse = DataSet[i].Split(',');
            int index = int.Parse(Parse[1]);
            if (index > HighestIndex) HighestIndex = index;
            Data D = new Data()
            {
                Input = DataSetFilePath + @"\" + Parse[0],
                ExpectedOutputIndex = (index - 1)
            };
            if (Parse[2] == "FALSE")
                TrainingSet.Add(D);
            else
                TestSet.Add(D);
        }

        ExpectedClassificationValues = new Matrix[HighestIndex];
        for (int i = 0; i < ExpectedClassificationValues.Length; i++)
        {
            ExpectedClassificationValues[i] = new Matrix(1, HighestIndex);
            ExpectedClassificationValues[i].Buffer[i] = 1;
            ExpectedClassificationValues[i].SetData();
        }
        OutputDataSize = new TensorSize(HighestIndex);
    }
}

public abstract class NetworkLearning
{
    public float LearningRate;
    public DataSet DataSet;
    public int BatchSize;
    public Network.CostFunctionType CostFunction = Network.CostFunctionType.SoftMaxCrossEntropy;

    private Network[] Networks;

    public bool DisplayEachBatch = true;

    public NetworkLearning(Network.CostFunctionType CostFunction, Network[] Networks, DataSet DataSet, int BatchSize, float LearningRate)
    {
        this.LearningRate = LearningRate;
        this.DataSet = DataSet;
        this.Networks = Networks;
        this.BatchSize = BatchSize;
        this.CostFunction = CostFunction;

        BatchEvaluation = new Network.BackPropagationEvaluation[Networks.Length][];
        for (int i = 0; i < Networks.Length; i++)
        {
            BatchEvaluation[i] = new Network.BackPropagationEvaluation[BatchSize];
        }
        TotalCorrect = new int[Networks.Length];
        TotalErrorCost = new float[Networks.Length];
        TotalBatchErrorCost = new float[Networks.Length];

        this.DataSet.ShuffleData();
    }

    public int LayerRenderIndex = 0;

    private int Epoch = 0;
    private int Batch = 0;
    private int TrainingIndex = 0;
    private int BatchIndex = 0;
    private float[] TotalErrorCost;
    private float[] TotalBatchErrorCost;
    private Network.BackPropagationEvaluation[][] BatchEvaluation;
    public void PerformSingleTeachOperation(bool Render)
    {
        for (int i = 0; i < Networks.Length; i++)
        {
            DataSet.Data D = DataSet.TrainingSet[TrainingIndex];
            BatchEvaluation[i][BatchIndex] = Networks[i].Backpropagation(D.Input, DataSet.ExpectedClassificationValues[D.ExpectedOutputIndex]);
            if (DisplayEachBatch)
                PrintSingleBatchEvaluation(BatchEvaluation[i][BatchIndex], i, Batch, Epoch, BatchIndex);
            TotalErrorCost[i] += BatchEvaluation[i][BatchIndex].ErrorCost;
            TotalBatchErrorCost[i] += BatchEvaluation[i][BatchIndex].ErrorCost;
            if (Render)
            {
                Networks[i].RenderModel(LayerRenderIndex);
            }
        }
        BatchIndex++;
        if (BatchIndex >= BatchSize)
        {
            BatchIndex = 0;
            PrintEndBatchEvaluation();
            EvaluateNetworks();
            for (int i = 0; i < Networks.Length; i++)
            {
                TotalBatchErrorCost[i] = 0;
            }
            Batch++;
        }
        TrainingIndex++;
        if (TrainingIndex >= DataSet.TrainingSet.Count)
        {
            TrainingIndex = 0;
            BatchIndex = 0;
            PrintEndEpochEvaluation();
            EvaluateNetworks();
            for (int i = 0; i < Networks.Length; i++)
            {
                TotalErrorCost[i] = 0;
            }
            Batch = 0;
            DataSet.ShuffleData();
        }
    }

    private int TestIndex;
    private int[] TotalCorrect;
    public void PerformSingleTestOperation(bool Render)
    {
        for (int i = 0; i < Networks.Length; i++)
        {
            DataSet.Data D = DataSet.TestSet[TestIndex];
            Network.BackPropagationEvaluation Eval = Networks[i].Backpropagation(D.Input, DataSet.ExpectedClassificationValues[D.ExpectedOutputIndex]);
            Matrix Output = null;
            switch (CostFunction)
            {
                case Network.CostFunctionType.SoftMaxCrossEntropy: Output = CostFunctions.SoftMax(Eval.Output); break;
                default: Output = Eval.Output; break;
            }
            Output.GetData();
            int NetworkAnswer = 0;
            float HighestNetNode = Output.Buffer[0];
            for (int j = 1; j < Output.Buffer.Length; j++)
            {
                if (Output.Buffer[j] > HighestNetNode)
                    NetworkAnswer = j;
            }
            int ExpectedAnswer = 0;
            Eval.Expected.GetData();
            HighestNetNode = Eval.Expected.Buffer[0];
            for (int j = 1; j < Output.Buffer.Length; j++)
            {
                if (Output.Buffer[j] > HighestNetNode)
                    ExpectedAnswer = j;
            }
            if (NetworkAnswer == ExpectedAnswer)
            {
                TotalCorrect[i]++;
            }
            PrintFinalTestEvaluation(Eval, i, TotalCorrect[i]);
            if (Render)
            {
                Networks[i].RenderModel(LayerRenderIndex);
            }
        }
        TestIndex++;
        if (TestIndex > DataSet.TestSet.Count)
        {
            ResetTest();
        }
    }

    public void ResetTest()
    {
        DataSet.ShuffleData();
        for (int i = 0; i < Networks.Length; i++)
        {
            Debug.Log("Network " + i + " > " + ((float)TotalCorrect[i] / TestIndex) + " > [" + TotalCorrect[i] + "/" + TestIndex + "]");
            TotalCorrect[i] = 0;
        }
        TestIndex = 0;
    }

    private void PrintFinalTestEvaluation(Network.BackPropagationEvaluation Eval, int NetworkIndex, int TotalCorrect)
    {
        if (Eval.Output == null) return;
        Eval.Output.GetData();
        string OutputString;
        switch (CostFunction)
        {
            case Network.CostFunctionType.SoftMaxCrossEntropy: OutputString = CostFunctions.SoftMax(Eval.Output).ToString(); break;
            default: OutputString = Eval.Output.ToString(); break;
        }
        Debug.Log("Network " + NetworkIndex + " > [" + TotalCorrect + "/" + TestIndex + "] > " + OutputString + " > E:" + Eval.Expected);
    }

    private void PrintEndBatchEvaluation()
    {
        Debug.Log("--- BATCH SUMMARY ---");
        for (int i = 0; i < Networks.Length; i++)
        {
            Debug.Log("Network " + i + " > " + Epoch + " > [" + TotalBatchErrorCost[i] + "]");
        }
        Debug.Log("--- ------------- ---");
    }

    private void PrintEndEpochEvaluation()
    {
        Debug.Log("--- EPOCH SUMMARY ---");
        for (int i = 0; i < Networks.Length; i++)
        {
            Debug.Log("Network " + i + " > " + Epoch + " > [" + TotalErrorCost[i] + "]");
        }
        Debug.Log("--- ------------- ---");
    }


    private void EvaluateNetworks()
    {
        for (int i = 0; i < Networks.Length; i++)
        {
            Network.BackPropagationEvaluation SummedDelta = new Network.BackPropagationEvaluation();
            SummedDelta.Deltas = new Network.WeightBiasDeltas[Networks[i].Model.Count];
            for (int l = 0; l < SummedDelta.Deltas.Length; l++)
            {
                SummedDelta.Deltas[l] = new Network.WeightBiasDeltas();
                if (BatchEvaluation[i][0].Deltas[l].WeightDeltas != null)
                {
                    SummedDelta.Deltas[l].WeightDeltas = new Matrix(BatchEvaluation[i][0].Deltas[l].WeightDeltas.Rows, BatchEvaluation[i][0].Deltas[l].WeightDeltas.Cols);
                    SummedDelta.Deltas[l].WeightDeltas.CopyValues(BatchEvaluation[i][0].Deltas[l].WeightDeltas.GetData());
                    SummedDelta.Deltas[l].BiasDeltas = new Matrix(BatchEvaluation[i][0].Deltas[l].BiasDeltas.Rows, BatchEvaluation[i][0].Deltas[l].BiasDeltas.Cols);
                    SummedDelta.Deltas[l].BiasDeltas.CopyValues(BatchEvaluation[i][0].Deltas[l].BiasDeltas.GetData());
                }
            }
            for (int j = 1; j < BatchEvaluation[i].Length; j++)
            {
                for (int k = 0; k < BatchEvaluation[i][j].Deltas.Length; k++)
                {
                    if (BatchEvaluation[i][j].Deltas[k].WeightDeltas != null)
                    {
                        SummedDelta.Deltas[k].WeightDeltas.AddInPlace(BatchEvaluation[i][j].Deltas[k].WeightDeltas * (1f / BatchSize));
                        SummedDelta.Deltas[k].BiasDeltas.AddInPlace(BatchEvaluation[i][j].Deltas[k].BiasDeltas * (1f / BatchSize));
                    }
                }
            }
            Networks[i].ApplyWeightBiasDeltas(SummedDelta, LearningRate);
        }
    }

    public abstract void PrintSingleBatchEvaluation(Network.BackPropagationEvaluation Eval, int NetworkIndex, int Batch, int Epoch, int BatchIndex);
}

public class CNNNetworkLearning : NetworkLearning
{
    public CNNNetworkLearning(Network.CostFunctionType CostFunction, Network[] Networks, DataSet DataSet, int BatchSize, float LearningRate) : base(CostFunction, Networks, DataSet, BatchSize, LearningRate)
    {
       
    }

    public override void PrintSingleBatchEvaluation(Network.BackPropagationEvaluation Eval, int NetworkIndex, int Batch, int Epoch, int BatchIndex)
    {
        Eval.Output.GetData();
        string OutputString;
        switch (CostFunction)
        {
            case Network.CostFunctionType.SoftMaxCrossEntropy: OutputString = CostFunctions.SoftMax(Eval.Output).ToString(); break;
            default: OutputString = Eval.Output.ToString(); break;
        }
        Debug.Log("Network " + NetworkIndex + " > [" + Epoch + "," + Batch + "," + BatchIndex + "] > " + Eval.ErrorCost + " > " + OutputString + " > E:" + Eval.Expected);
    }
}

public class CostFunctions
{
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
    public static Matrix SquaredError_Derivation(Matrix Output, Matrix Expected)
    {
        Matrix Derivations = new Matrix(Output.Rows, Output.Cols);
        for (int i = 0; i < Output.Buffer.Length; i++)
        {
            Derivations.Buffer[i] = 2 * (Output.Buffer[i] - Expected.Buffer[i]);
        }
        Derivations.SetData();
        return Derivations;
    }

    public static Matrix MeanSquaredError(Matrix Output, Matrix Expected)
    {
        Matrix Cost = new Matrix(Output.Rows, Output.Cols);
        for (int i = 0; i < Output.Buffer.Length; i++)
        {
            float Val = Output.Buffer[i] - Expected.Buffer[i];
            Cost.Buffer[i] = (Val * Val) / 2;
        }
        Cost.SetData();
        return Cost;
    }
    public static Matrix MeanSquaredError_Derivation(Matrix Output, Matrix Expected)
    {
        Matrix Derivations = new Matrix(Output.Rows, Output.Cols);
        for (int i = 0; i < Output.Buffer.Length; i++)
        {
            Derivations.Buffer[i] = Output.Buffer[i] - Expected.Buffer[i];
        }
        Derivations.SetData();
        return Derivations;
    }

    public static Matrix SoftMax(Matrix Output)
    {
        Matrix SoftMax = new Matrix(Output.Rows, Output.Cols);
        float Sum = Output.Buffer.Sum(v => Mathf.Exp(v));
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
        float Sum = Output.Buffer.Sum(v => Mathf.Exp(v));
        for (int i = 0; i < Output.Buffer.Length; i++)
        {
            float Val = (Mathf.Exp(Output.Buffer[i]) / Sum);
            Cost.Buffer[i] = -(Expected.Buffer[i] * Mathf.Log(Val));
        }
        Cost.SetData();
        return Cost;
    }
    public static Matrix SoftMaxCrossEntropyLoss_Derivation(Matrix Output, Matrix Expected)
    {
        Matrix Derivations = new Matrix(Output.Rows, Output.Cols);
        float Sum = Output.Buffer.Sum(v => Mathf.Exp(v));
        float ExpectedSum = Expected.Buffer.Sum();
        for (int i = 0; i < Output.Buffer.Length; i++)
        {
            float Val = (Mathf.Exp(Output.Buffer[i]) / Sum);
            Derivations.Buffer[i] = Val * ExpectedSum - Expected.Buffer[i];
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

    public CostFunctionType CostFunc = CostFunctionType.SoftMaxCrossEntropy;
    public List<Layer> Model = new List<Layer>();

    public void Dispose()
    {
        for (int i = 0; i < Model.Count; i++)
        {
            Model[i].Dispose();
        }
    }

    public void RenderModel(int RenderIndex)
    {
        for (int i = 0; i < Model.Count; i++)
            Model[i].Render(RenderIndex);
    }

    public Matrix ForwardPropagate(object Input)
    {
        if (Model.Count == 0)
        {
            Debug.LogError("Model contains no layers!");
            return null;
        }

        Model[0].AssignInput(Input);
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
            Deltas = new WeightBiasDeltas[ModelSize];
            Expected = null;
        }
    }

    public void ApplyWeightBiasDeltas(BackPropagationEvaluation BackpropEvaluation, float LearningRate)
    {
        for (int i = 0; i < Model.Count; i++)
        {
            Model[i].ApplyWeightBiasDeltas(BackpropEvaluation.Deltas[i], LearningRate);
        }
    }

    //Performs backpropagation on a singular input to return an array of weight and bias changes 
    public BackPropagationEvaluation Backpropagation(object Input, Matrix Expected)
    {
        //Initialize Matrix of deltas
        BackPropagationEvaluation BackPropEvaluation = new BackPropagationEvaluation(Model.Count);

        //Firstly, propogate the input through the network
        BackPropEvaluation.Output = ForwardPropagate(Input);
        BackPropEvaluation.Expected = Expected;
        switch (CostFunc)
        {
            case CostFunctionType.SquaredError: BackPropEvaluation.ErrorCost = CostFunctions.SquaredError(Model[Model.Count - 1].Output, Expected).Buffer.Sum(); break;
            case CostFunctionType.MeanSquaredError: BackPropEvaluation.ErrorCost = CostFunctions.MeanSquaredError(Model[Model.Count - 1].Output, Expected).Buffer.Sum(); break;
            case CostFunctionType.SoftMaxCrossEntropy: BackPropEvaluation.ErrorCost = CostFunctions.SoftMaxCrossEntropyLoss(Model[Model.Count - 1].Output, Expected).Buffer.Sum(); break;
            default: BackPropEvaluation.ErrorCost = CostFunctions.SquaredError(Model[Model.Count - 1].Output, Expected).Buffer.Sum(); break;
        }

        //Calculate Delta of output layer
        Matrix CostDerivatives;
        switch (CostFunc)
        {
            case CostFunctionType.SquaredError: CostDerivatives = CostFunctions.SquaredError_Derivation(Model[Model.Count - 1].Output, Expected); break;
            case CostFunctionType.MeanSquaredError: CostDerivatives = CostFunctions.MeanSquaredError_Derivation(Model[Model.Count - 1].Output, Expected); break;
            case CostFunctionType.SoftMaxCrossEntropy: CostDerivatives = CostFunctions.SoftMaxCrossEntropyLoss_Derivation(Model[Model.Count - 1].Output, Expected); break;
            default: CostDerivatives = CostFunctions.SquaredError_Derivation(Model[Model.Count - 1].Output, Expected); break;
        }
        CostDerivatives.GetData();

        BackPropEvaluation.Deltas[Model.Count - 1] = Model[Model.Count - 1].Backprop(CostDerivatives);
        //Loop backwards through the model and perform backpropagation
        for (int i = Model.Count - 2; i >= 0; i--)
        {
            BackPropEvaluation.Deltas[i] = Model[i].Backprop(Model[i + 1].LayerDeltas);
        }

        return BackPropEvaluation;
    }
}
