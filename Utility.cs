using System;
using ActivationTemplate;



public class HyperbolicTangent : Activation
{
	public override float ActiveFunction(float input){		
		float positive = (float)Math.Pow(Math.E , input);
		float negative = (float)Math.Pow(Math.E , -input);
		if(Single.IsNaN((positive-negative)/(positive+negative))){
			return 0;
		}
		return (positive-negative)/(positive+negative);
	}
	
	public override float ActiveFunctionDeriv(float input){		
		return (1-(float)Math.Pow(input, 2));
	}
	
}

public class Sigmoid : Activation
{
	public override float ActiveFunction(float input){
		return (float)1/(float)(1+Math.Pow(Math.E, -input));
	}
	
	public override float ActiveFunctionDeriv(float input){
		return (float)(input)*(1-input);
	}
}

public class Default : Activation
{
	public override float ActiveFunction(float input){
		return input;
	}
	
	public override float ActiveFunctionDeriv(float input){
		return 1;
	}
	
}


namespace Utility{
	public class InOutPair{
		public float[] Inputs;
		public float[] Goals;
		
		public InOutPair(float[] I, float[] G){
			Inputs = new float[I.Length];
			Goals = new float[G.Length];
			
			for(int i = 0; i < Inputs.Length; i++)
				Inputs[i] = I[i];
			for(int i = 0; i < Goals.Length; i++)
				Goals[i] = G[i];
			
		}
		
		
		
	}

	public class Epoch{
		public InOutPair[] Set;
		public bool Normalize;
		public int Runs;
		
		public Epoch(InOutPair[] S, int R = 10, bool Norm = false){
			Normalize = Norm;
			Runs = R;
			
			Set = S;
		}
	}

	public class Matrix{
		private float[][] M;
		
		public int length;
		public int height;
		
		public float[] this[int i]
		{
			get
			{
				return M[i];
			}
			set
			{
				M[i] = value;
			}
		}
		
		public float this[int i, int j]
		{
			get
			{
				return M[i][j];
			}
			set
			{
				M[i][j] = value;
			}
		}
		
		
		
		public Matrix(float[][] Input){
			length = Input.Length;
			height = Input[0].Length;
			M = new float[length][];
			for(int i = 0; i < length; i++){
				M[i] = new float[height];
				for(int j = 0; j < height; j++){
					M[i][j] = Input[i][j];
				}
			}
		}
		
		
		
		public Matrix(int A, int B, float Init = 0){
			length = A;
			height = B;
			M = new float[length][];
			for(int i = 0; i < length; i ++){
				M[i] = new float[height];
				for(int j = 0; j < height; j++)
					M[i][j] = Init;
			}
		}
		
		
		
		public static Matrix operator/ (Matrix A, float B){
			float[][] newMatrix = new float[A.length][];
			
			for(int i = 0; i < A.length; i++){
				newMatrix[i] = new float[A.height];
				for(int j = 0; j < A.height; j++){
					newMatrix[i][j] = A[i,j]/B;	
				}
			}
			return new Matrix(newMatrix);
		}
		
		public static Matrix operator* (Matrix A, Matrix B){
			
			float[][] newMatrix = new float[B.length][];
			
			for(int i = 0; i < B.length; i++){
				newMatrix[i] = new float[A.height];
				for(int j = 0; j < A.height; j++){
					for(int k = 0; k < A.length; k++){
						newMatrix[i][j] += A[k, j] * B[i, k];
					}
				}
			}
			return new Matrix(newMatrix);
		}
		
		public static Matrix operator* (Matrix A, float B){		
			float[][] newMatrix = new float[A.length][];
			
			for(int i = 0; i < A.length; i++){
				newMatrix[i] = new float[A.height];
				for(int j = 0; j < A.height; j++){
					newMatrix[i][j] = A[i,j]*B;	
				}
			}
			return new Matrix(newMatrix);
		}
		
		
		public static Matrix operator+ (Matrix A, Matrix B){
			
			float[][] newMatrix = new float[A.length][];
			
			for(int i = 0; i < A.length; i++){
				newMatrix[i] = new float[A.height];
				for(int j = 0; j < A.height; j++){
					newMatrix[i][j] = A[i,j] + B[i,j];	
				}
			}
			return new Matrix(newMatrix);
		}
		
		public static Matrix operator- (Matrix A, Matrix B){
			
			float[][] newMatrix = new float[A.length][];
			
			for(int i = 0; i < A.length; i++){
				newMatrix[i] = new float[A.height];
				for(int j = 0; j < A.height; j++){
					newMatrix[i][j] = A[i,j] - B[i,j];	
				}
			}
			return new Matrix(newMatrix);
		}
		
	}

}