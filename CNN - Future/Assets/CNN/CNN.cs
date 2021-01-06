﻿using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class CNN : MonoBehaviour
{
    [RuntimeInitializeOnLoadMethod]
    private static void OnLoad()
    {
        //Load EigenDLL
        Matrix.LoadDLL();

        /*Matrix M1 = new Matrix(2, 2);
        for (int i = 0; i < M1.Buffer.Length; i++)
        {
            M1.Buffer[i] = 3;
        }
        M1.SetData();
        Matrix M2 = new Matrix(2, 2);
        for (int i = 0; i < M2.Buffer.Length; i++)
        {
            M2.Buffer[i] = 1;
        }
        M2.SetData();
        M1.SubInPlace(M2);
        M1.GetData();
        UnityEngine.Debug.Log(M1);*/

        //TODO:: implement batch size learning and averaging of weight matrices for each batch for multiple inputs
        //TODO:: implement backprop for other layers

        /*Matrix Input = new Matrix(1, 2);
        Input.Buffer[0] = 1;
        Input.Buffer[1] = 0;
        Input.SetData();

        Matrix Expected = new Matrix(1, 3);
        Expected.Buffer[0] = 0.27f;
        Expected.Buffer[1] = 0.69f;
        Expected.Buffer[2] = -0.72f;
        Expected.SetData();

        Network N = new Network();
        N.Model.Add(new DenseLayer(new TensorSize(2), 2, DenseLayer.ActivationType.Tanh));
        N.Model.Add(new DenseLayer(N.Model[0].OutputTensor, 3, DenseLayer.ActivationType.Tanh));
        N.Model.Add(new DenseLayer(N.Model[1].OutputTensor, 3, DenseLayer.ActivationType.Tanh));
        N.Model.Add(new DenseLayer(N.Model[2].OutputTensor, 3, DenseLayer.ActivationType.Tanh));

        for (int i = 0; i < 1000; i++)
        {
            Network.BackPropagationEvaluation Deltas = N.Backpropagation(Input, Expected);
            Deltas.Output.GetData();
            Debug.Log(Deltas.Output);
            N.ApplyWeightBiasDeltas(Deltas, 0.1f);
        }

        for (int i = 0; i < N.Model.Count; i++)
        {
            N.Model[i].Render(0);
        }
        //N.Dispose();*/
    }

    CNNNetworkLearning Learner;

    public bool Training;

    public void Start()
    {
        /*N = new Network();
        N.Model.Add(new ConvolutionTextureLayer(new TensorSize(Inputs[0].width, Inputs[0].height, 3), 3, 3, 1, 0, ConvolutionLayer.ActivationType.Tanh));
        N.Model.Add(new PoolingLayer(N.Model[0].OutputTensor, 2, 2));
        N.Model.Add(new ConvolutionMatrixLayer(N.Model[1].OutputTensor, 5, 3, 1, 0, ConvolutionLayer.ActivationType.Tanh));
        N.Model.Add(new PoolingLayer(N.Model[2].OutputTensor, 2, 2));
        N.Model.Add(new ConvolutionMatrixLayer(N.Model[3].OutputTensor, 5, 3, 1, 0, ConvolutionLayer.ActivationType.Tanh));
        N.Model.Add(new PoolingLayer(N.Model[4].OutputTensor, 2, 2));
        N.Model.Add(new FlatteningLayer(N.Model[5].OutputTensor));
        N.Model.Add(new DenseLayer(N.Model[6].OutputTensor, 20, DenseLayer.ActivationType.Tanh));
        N.Model.Add(new DenseLayer(N.Model[7].OutputTensor, 20, DenseLayer.ActivationType.Tanh));
        N.Model.Add(new DenseLayer(N.Model[8].OutputTensor, 10, DenseLayer.ActivationType.Tanh));
        N.Model.Add(new DenseLayer(N.Model[9].OutputTensor, 2, DenseLayer.ActivationType.Tanh));*/

        ImageDataSet D = new ImageDataSet(
            new TensorSize(100, 60),
            @"C:\Users\LenovoY720\Documents\ImageDataSets",
            @"C:\Users\LenovoY720\Documents\ImageDataSets\ClassNames.csv",
            @"C:\Users\LenovoY720\Documents\ImageDataSets\ImageLabels.csv"
        );

        Network[] NetworkList = new Network[1];
        for (int i = 0; i < NetworkList.Length; i++)
        {
            Network N = new Network();
            N.Model.Add(new ConvolutionTextureLayer(new TensorSize(100, 60, 3), 10, 3, 1, 0, ConvolutionLayer.ActivationType.Tanh));
            N.Model.Add(new PoolingLayer(N.Model[0].OutputTensor, 2, 2));
            N.Model.Add(new ConvolutionMatrixLayer(N.Model[1].OutputTensor, 20, 3, 1, 0, ConvolutionLayer.ActivationType.Tanh));
            N.Model.Add(new PoolingLayer(N.Model[2].OutputTensor, 2, 2));
            N.Model.Add(new ConvolutionMatrixLayer(N.Model[3].OutputTensor, 20, 3, 1, 0, ConvolutionLayer.ActivationType.Tanh));
            N.Model.Add(new PoolingLayer(N.Model[4].OutputTensor, 2, 2));
            N.Model.Add(new FlatteningLayer(N.Model[5].OutputTensor));
            N.Model.Add(new DenseLayer(N.Model[6].OutputTensor, 200, DenseLayer.ActivationType.Tanh));
            N.Model.Add(new DenseLayer(N.Model[7].OutputTensor, 200, DenseLayer.ActivationType.Tanh));
            N.Model.Add(new DenseLayer(N.Model[8].OutputTensor, 200, DenseLayer.ActivationType.Tanh));
            N.Model.Add(new DenseLayer(N.Model[9].OutputTensor, D.OutputDataSize.Width, DenseLayer.ActivationType.Tanh));
            NetworkList[i] = N;
        }
        Learner = new CNNNetworkLearning(Network.CostFunctionType.SoftMaxCrossEntropy, NetworkList, D, 100, 0.1f);
    }


    private bool ResetTag = false;
    public bool DisplayEachBatch = true;
    public float LearningRate = 1f;

    public void Update()
    {
        Learner.LearningRate = LearningRate;
        Learner.DisplayEachBatch = DisplayEachBatch;
        if (Training)
        {
            Learner.PerformSingleTeachOperation(true);
            ResetTag = false;
        }
        else
        {
            if (!ResetTag)
            {
                Learner.ResetTest();
                ResetTag = true;
            }
            Learner.PerformSingleTestOperation(true);
        }
    }
}