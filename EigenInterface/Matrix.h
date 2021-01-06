#pragma once
#include <Eigen/Dense>

extern "C"
{
	__declspec(dllexport) Eigen::Map<Eigen::MatrixXf>* __stdcall CreateEigenMatrix(float* Data, int Rows, int Cols)
	{
		return new Eigen::Map<Eigen::MatrixXf>(Data, Rows, Cols);
	}

	__declspec(dllexport) void __stdcall DeleteEigenMatrix(Eigen::Map<Eigen::MatrixXf>* Matrix)
	{
		delete Matrix;
	}

	__declspec(dllexport) void __stdcall MultiplyEigenMatrix(Eigen::Map<Eigen::MatrixXf>* MatrixA, Eigen::Map<Eigen::MatrixXf>* MatrixB, Eigen::Map<Eigen::MatrixXf>* MatrixC)
	{
		MatrixC->noalias() = *MatrixA * *MatrixB;
	}

	__declspec(dllexport) void __stdcall MultiplyConstantEigenMatrix(Eigen::Map<Eigen::MatrixXf>* MatrixA, float Const, Eigen::Map<Eigen::MatrixXf>* MatrixC)
	{
		MatrixC->noalias() = *MatrixA * Const;
	}

	__declspec(dllexport) void __stdcall CWiseMultiplyEigenMatrix(Eigen::Map<Eigen::MatrixXf>* MatrixA, Eigen::Map<Eigen::MatrixXf>* MatrixB, Eigen::Map<Eigen::MatrixXf>* MatrixC)
	{
		MatrixC->noalias() = MatrixA->cwiseProduct(*MatrixB);
	}

	__declspec(dllexport) void __stdcall AddEigenMatrix(Eigen::Map<Eigen::MatrixXf>* MatrixA, Eigen::Map<Eigen::MatrixXf>* MatrixB, Eigen::Map<Eigen::MatrixXf>* MatrixC)
	{
		MatrixC->noalias() = *MatrixA + *MatrixB;
	}

	__declspec(dllexport) void __stdcall AddInPlaceEigenMatrix(Eigen::Map<Eigen::MatrixXf>* MatrixA, Eigen::Map<Eigen::MatrixXf>* MatrixB)
	{
		*MatrixA = *MatrixA + *MatrixB;
	}

	__declspec(dllexport) void __stdcall SubInPlaceEigenMatrix(Eigen::Map<Eigen::MatrixXf>* MatrixA, Eigen::Map<Eigen::MatrixXf>* MatrixB)
	{
		*MatrixA = *MatrixA - *MatrixB;
	}

	__declspec(dllexport) void __stdcall TransposeEigenMatrix(Eigen::Map<Eigen::MatrixXf>* Matrix, Eigen::Map<Eigen::MatrixXf>* Result)
	{
		*Result = Matrix->transpose();
	}

	__declspec(dllexport) void __stdcall ColWiseFlipEigenMatrix(Eigen::Map<Eigen::MatrixXf>* Matrix, Eigen::Map<Eigen::MatrixXf>* Result)
	{
		Result->noalias() = Matrix->colwise().reverse();
	}

	__declspec(dllexport) void __stdcall RowWiseFlipEigenMatrix(Eigen::Map<Eigen::MatrixXf>* Matrix, Eigen::Map<Eigen::MatrixXf>* Result)
	{
		Result->noalias() = Matrix->rowwise().reverse();
	}

	__declspec(dllexport) void __stdcall TanhActivationEigenMatrix(Eigen::Map<Eigen::MatrixXf>* Matrix)
	{
		Matrix->noalias() = Matrix->unaryExpr([](float val)
			{
				return tanh(val);
			});
	}

	__declspec(dllexport) void __stdcall MaxActivationEigenMatrix(Eigen::Map<Eigen::MatrixXf>* Matrix)
	{
		Matrix->noalias() = Matrix->unaryExpr([](float val)
			{
				return max(0, val);
			});
	}

	__declspec(dllexport) void __stdcall SigmoidActivationEigenMatrix(Eigen::Map<Eigen::MatrixXf>* Matrix)
	{
		Matrix->noalias() = Matrix->unaryExpr([](float val)
			{
				float Exp = exp(val);
				return Exp / (Exp + 1);
			});
	}

	__declspec(dllexport) void __stdcall TanhActivation_DerivationEigenMatrix(Eigen::Map<Eigen::MatrixXf>* Matrix)
	{
		Matrix->noalias() = Matrix->unaryExpr([](float val)
			{
				float TanhVal = tanh(val);
				return 1 - TanhVal * TanhVal;
			});
	}

	//derivation of y=x since thats what it is, but the gradient below 0 is 0 (technically not strictly defined but lets say tis 0)
	__declspec(dllexport) void __stdcall MaxActivation_DerivationEigenMatrix(Eigen::Map<Eigen::MatrixXf>* Matrix)
	{
		Matrix->noalias() = Matrix->unaryExpr([](float val)
			{
				return val < 0 ? 0 : 1;
			});
	}

	__declspec(dllexport) void __stdcall SigmoidActivation_DerivationEigenMatrix(Eigen::Map<Eigen::MatrixXf>* Matrix)
	{
		Matrix->noalias() = Matrix->unaryExpr([](float val)
			{
				float Exp = exp(val);
				float Recip = (Exp + 1);
				return Exp / (Recip * Recip);
			});
	}
}