using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

public struct Question
{
    public object Input;
    public int ExpectedGroup;
    public Matrix ExpectedOutput;
}

public class DataSet
{
    public string ClassNameFilePath;
    public string QuestionAnswerFilePath;

    public Dictionary<int, string> IDToClass = new Dictionary<int, string>();
    public Dictionary<int, int> IDToGroup = new Dictionary<int, int>();
    public Dictionary<int, List<string>> IDToClassName = new Dictionary<int, List<string>>();
    public List<Question> Test = new List<Question>();

    public DataSet(string ClassNameFilePath, string QuestionAnswerFilePath)
    {
        this.ClassNameFilePath = ClassNameFilePath;
        this.QuestionAnswerFilePath = QuestionAnswerFilePath;

        string[] ClassNames = File.ReadAllLines(ClassNameFilePath);
        string[] Classes = ClassNames[0].Split(',');
        string[] Grouping = ClassNames[1].Split(',');

        for (int i = 1; i < Classes.Length + 1; i++)
        {
            IDToClass.Add(int.Parse(Grouping[i - 1]), Classes[i - 1]);
            IDToGroup.Add(i, int.Parse(Grouping[i - 1]));
        }

        string[] Questions = File.ReadAllLines(QuestionAnswerFilePath);
        int NumAnswers = 0;
        for (int i = 0; i < Questions.Length; i++)
        {
            string[] Question = Questions[i].Split(',');
            int ID = IDToGroup[int.Parse(Question[0])];
            if (!IDToClassName.ContainsKey(ID))
            {
                NumAnswers++;
                IDToClassName.Add(ID, new List<string>());
            }
        }

        List<int> PotentialAnswers = IDToClassName.Keys.ToList();
        for (int i = 0; i < Questions.Length; i++)
        {
            string[] Question = Questions[i].Split(',');
            Matrix Answer = new Matrix(1, NumAnswers);
            int QuestionAnswer = IDToGroup[int.Parse(Question[0])];
            for (int j = 0; j < PotentialAnswers.Count; j++)
            {
                if (PotentialAnswers[j] == QuestionAnswer)
                {
                    Answer.Buffer[j] = 1;
                    break;
                }
            }
            Answer.SetData();
            Test.Add(new Question()
            {
                Input = Question[1],
                ExpectedGroup = QuestionAnswer,
                ExpectedOutput = Answer
            });
        }
    }
}


