using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CNN
{
    public static DataSet TrainingSet = new DataSet(@"D:\Compsci\ImageDataSets\ClassNames.csv", @"D:\Compsci\ImageDataSets\ImageNames.csv");
    public static Learning TeacherBot;
    public static Network N = new Network();

    public static void Start()
    {
        Matrix.LoadDLL();

        N.CostFunc = Network.CostFunctionType.MeanSquaredError;
        N.Model.Add(new ConvolutionTextureLayer(new TensorSize(50, 50, 3), 5, 0, 3, 1));
        N.Model.Add(new Pooling(N.Model[0].OutputTensor, 2, 2));
        N.Model.Add(new ConvolutionMatrixLayer(N.Model[1].OutputTensor, 5, 0, 3, 1));
        N.Model.Add(new Pooling(N.Model[2].OutputTensor, 2, 2));
        N.Model.Add(new ConvolutionMatrixLayer(N.Model[3].OutputTensor, 5, 0, 3, 1));
        N.Model.Add(new Pooling(N.Model[4].OutputTensor, 2, 2));
        N.Model.Add(new Flattening(N.Model[5].OutputTensor));
        N.Model.Add(new DenseLayer(N.Model[6].OutputTensor, 20, ActivationType.Tanh));
        N.Model.Add(new DenseLayer(N.Model[7].OutputTensor, 10, ActivationType.Tanh));
        N.Model.Add(new DenseLayer(N.Model[8].OutputTensor, 2, ActivationType.Tanh));

        TeacherBot = new Learning(new DataSet(@"D:\Compsci\ImageDataSets\ClassNames.csv", @"D:\Compsci\ImageDataSets\ImageNames.csv"), N, @"D:\Compsci\ImageDataSets");
    }

    public static bool Learning = true;
    public static int RenderLayer = 0;
    public static int PrevRenderLayer = -1;
    public static void Update()
    {
        if (Learning == true) TeacherBot.Teach();
        if (RenderLayer != PrevRenderLayer || Learning)
        {
            PrevRenderLayer = RenderLayer;
            N.Render(RenderLayer);
        }
    }
}
