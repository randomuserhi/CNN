using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.IO;
using UnityEngine;

public class Matrix
{
    private IntPtr Data;
    private IntPtr EigenMatrix;
    public float[] Buffer;
    public readonly int Rows;
    public readonly int Cols;

    public Matrix(int Rows, int Cols)
    {
        if (Rows < 1 || Cols < 1)
            Debug.LogWarning("Invalid Rows or Cols values: " + Rows + ", " + Cols);

        this.Rows = Rows;
        this.Cols = Cols;
        Buffer = new float[Rows * Cols];

        //Allocate memory
        Data = Marshal.AllocHGlobal(Buffer.Length * sizeof(float));
        EigenMatrix = CreateEigenMatrix(Data, Rows, Cols);
    }

    public float[] SetData()
    {
        Marshal.Copy(Buffer, 0, Data, Buffer.Length);
        return Buffer;
    }

    public float[] GetData()
    {
        Marshal.Copy(Data, Buffer, 0, Buffer.Length);
        return Buffer;
    }

    public void CopyValues(float[] Values)
    {
        Array.Copy(Values, 0, Buffer, 0, Buffer.Length);
        SetData();
    }

    public static Matrix operator *(Matrix A, Matrix B)
    {
        Matrix Result = new Matrix(A.Rows, B.Cols);
        MultiplyEigenMatrix(A.EigenMatrix, B.EigenMatrix, Result.EigenMatrix);
        return Result;
    }

    public static Matrix operator *(Matrix A, float Const)
    {
        Matrix Result = new Matrix(A.Rows, A.Cols);
        MultiplyConstantEigenMatrix(A.EigenMatrix, Const, Result.EigenMatrix);
        return Result;
    }

    public static Matrix operator +(Matrix A, Matrix B)
    {
        Matrix Result = new Matrix(A.Rows, B.Cols);
        AddEigenMatrix(A.EigenMatrix, B.EigenMatrix, Result.EigenMatrix);
        return Result;
    }

    public static Matrix CWiseMultiply(Matrix A, Matrix B, Matrix Result)
    {
        CWiseMultiplyEigenMatrix(A.EigenMatrix, B.EigenMatrix, Result.EigenMatrix);
        return Result;
    }

    public Matrix Transpose()
    {
        Matrix TransposeResult = new Matrix(Cols, Rows);
        TransposeEigenMatrix(EigenMatrix, TransposeResult.EigenMatrix);
        return TransposeResult;
    }

    public Matrix ColWiseFlip()
    {
        Matrix Result = new Matrix(Cols, Rows);
        ColWiseFlipEigenMatrix(EigenMatrix, Result.EigenMatrix);
        return Result;
    }

    public Matrix RowWiseFlip()
    {
        Matrix Result = new Matrix(Cols, Rows);
        RowWiseFlipEigenMatrix(EigenMatrix, Result.EigenMatrix);
        return Result;
    }

    public static void Transpose(Matrix Mat, Matrix TransposeResult)
    {
        TransposeEigenMatrix(Mat.EigenMatrix, TransposeResult.EigenMatrix);
    }

    public void AddInPlace(Matrix Other)
    {
        AddInPlaceEigenMatrix(EigenMatrix, Other.EigenMatrix);
    }

    public void SubInPlace(Matrix Other)
    {
        SubInPlaceEigenMatrix(EigenMatrix, Other.EigenMatrix);
    }

    public void TanhActivation()
    {
        TanhActivationEigenMatrix(EigenMatrix);
    }

    public void MaxActivation()
    {
        MaxActivationEigenMatrix(EigenMatrix);
    }

    public void SigmoidActivation()
    {
        SigmoidActivationEigenMatrix(EigenMatrix);
    }

    public void TanhActivation_Derivation()
    {
        TanhActivation_DerivationEigenMatrix(EigenMatrix);
    }

    public void ReLUActivation_Derivation()
    {
        MaxActivation_DerivationEigenMatrix(EigenMatrix);
    }

    public void SigmoidActivation_Derivation()
    {
        SigmoidActivation_DerivationEigenMatrix(EigenMatrix);
    }

    ~Matrix()
    {
        Marshal.FreeHGlobal(Data);
        DeleteEigenMatrix(EigenMatrix);
    }

    public override string ToString()
    {
        return "[ " + String.Join(", ", Buffer) + " ]";
    }

    [DllImport("EigenInterface.dll")]
    private static extern IntPtr CreateEigenMatrix(IntPtr Data, int Rows, int Cols);

    [DllImport("EigenInterface.dll")]
    private static extern void DeleteEigenMatrix(IntPtr Matrix);

    [DllImport("EigenInterface.dll")]
    private static extern int TransposeEigenMatrix(IntPtr Matrix, IntPtr TransposeResult);

    [DllImport("EigenInterface.dll")]
    private static extern int ColWiseFlipEigenMatrix(IntPtr Matrix, IntPtr FlippedResult);

    [DllImport("EigenInterface.dll")]
    private static extern int RowWiseFlipEigenMatrix(IntPtr Matrix, IntPtr FlippedResult);

    [DllImport("EigenInterface.dll")]
    private static extern int MultiplyEigenMatrix(IntPtr MatrixA, IntPtr MatrixB, IntPtr MatrixC);

    [DllImport("EigenInterface.dll")]
    private static extern int MultiplyConstantEigenMatrix(IntPtr MatrixA, float Const, IntPtr MatrixC);

    [DllImport("EigenInterface.dll")]
    private static extern int CWiseMultiplyEigenMatrix(IntPtr MatrixA, IntPtr MatrixB, IntPtr MatrixC);

    [DllImport("EigenInterface.dll")]
    private static extern int AddEigenMatrix(IntPtr MatrixA, IntPtr MatrixB, IntPtr MatrixC);

    [DllImport("EigenInterface.dll")]
    private static extern int AddInPlaceEigenMatrix(IntPtr MatrixA, IntPtr MatrixB);

    [DllImport("EigenInterface.dll")]
    private static extern int SubInPlaceEigenMatrix(IntPtr MatrixA, IntPtr MatrixB);

    [DllImport("EigenInterface.dll")]
    private static extern int TanhActivationEigenMatrix(IntPtr Matrix);

    [DllImport("EigenInterface.dll")]
    private static extern int SigmoidActivationEigenMatrix(IntPtr Matrix);

    [DllImport("EigenInterface.dll")]
    private static extern int MaxActivationEigenMatrix(IntPtr Matrix);

    [DllImport("EigenInterface.dll")]
    private static extern int TanhActivation_DerivationEigenMatrix(IntPtr Matrix);

    [DllImport("EigenInterface.dll")]
    private static extern int SigmoidActivation_DerivationEigenMatrix(IntPtr Matrix);

    [DllImport("EigenInterface.dll")]
    private static extern int MaxActivation_DerivationEigenMatrix(IntPtr Matrix);

    public static void LoadDLL()
    {
        var currentPath = Environment.GetEnvironmentVariable("PATH",
        EnvironmentVariableTarget.Process);
#if UNITY_EDITOR_32
    var dllPath = Application.dataPath
        + Path.DirectorySeparatorChar + "Plugins"
        + Path.DirectorySeparatorChar + "x86";
#elif UNITY_EDITOR_64
        var dllPath = Application.dataPath
            + Path.DirectorySeparatorChar + "Plugins"
            + Path.DirectorySeparatorChar + "x86_64";
#else // Player
    var dllPath = Application.dataPath
        + Path.DirectorySeparatorChar + "Plugins";
#endif
        Debug.Log("Loaded DLLs from: " + dllPath);
        if (currentPath != null && currentPath.Contains(dllPath) == false)
            Environment.SetEnvironmentVariable("PATH", currentPath + Path.PathSeparator
                + dllPath, EnvironmentVariableTarget.Process);
    }
}