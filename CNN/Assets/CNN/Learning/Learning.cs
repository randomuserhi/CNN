using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

public struct Test
{
    public List<Question> Questions;
}

public class Learning
{
    private Network N;
    public float LearningRate = 0.01f;
    public string ImageFilePath;

    public DataSet DataSet;
    private Dictionary<int, List<Question>> SortedQuestions = new Dictionary<int, List<Question>>();
    public List<Test> Tests = new List<Test>();
    public const int NumTests = 10;

    public Learning(DataSet DataSet, Network N, string ImageFilePath)
    {
        this.ImageFilePath = ImageFilePath;
        this.N = N;
        this.DataSet = DataSet;
        for (int i = 0; i < DataSet.Test.Count; i++)
        {
            if (!SortedQuestions.ContainsKey(DataSet.Test[i].ExpectedGroup))
            {
                SortedQuestions.Add(DataSet.Test[i].ExpectedGroup, new List<Question>());
            }
            SortedQuestions[DataSet.Test[i].ExpectedGroup].Add(DataSet.Test[i]);
        }

        BatchEvals = new Network.BackPropagationEvaluation[NumTests];
        GenerateTests();
    }

    private int BatchNum = 0;
    private int IterNum = 0;
    private int EpochNum = 0;
    private Network.BackPropagationEvaluation[] BatchEvals;
    public void Teach()
    {
        if (IterNum >= Tests[BatchNum].Questions.Count) Batch();
        if (BatchNum >= Tests.Count) Epoch();
        BatchEvals[IterNum] = N.Backpropagation(ImageFilePath + System.IO.Path.DirectorySeparatorChar + Tests[BatchNum].Questions[IterNum].Input, Tests[BatchNum].Questions[IterNum].ExpectedOutput);
        Debug.Log("Iteration: " + BatchEvals[BatchNum].ErrorCost);
        IterNum++;
    }

    private void Batch()
    {
        IterNum = 0;
        BatchNum++;
        Network.BackPropagationEvaluation Sum = Network.BackPropagationEvaluation.Sum(BatchEvals, N.Model.Count);
        Debug.Log("EndBatch: " + Sum.ErrorCost);
        N.ApplyWeightBiasDeltas(Sum, LearningRate);
    }

    private void Epoch()
    {
        BatchNum = 0;
        EpochNum++;
        GenerateTests();
        Debug.Log("EndEpoch: " + EpochNum);
    }

    public void GenerateTests()
    {
        Tests.Clear();
        List<int> Topics = SortedQuestions.Keys.ToList();
        for (int i = 0; i < NumTests; i++)
        {
            const int NumQuestionsPerTopic = 5;
            Test Test = new Test()
            {
                Questions = new List<Question>()
            };
            for (int j = 0; j < Topics.Count; j++)
            {
                for (int k = 0; k < NumQuestionsPerTopic; k++)
                {
                    List<Question> Qs = SortedQuestions[Topics[j]];
                    Test.Questions.Add(Qs[UnityEngine.Random.Range(0, Qs.Count)]);
                }
            }
            Tests.Add(Test);
        }
    }
}
