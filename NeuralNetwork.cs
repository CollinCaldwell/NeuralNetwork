using System;
using System.Collections.Generic;

using Utility;
using ActivationTemplate;


namespace NeuralNetworkFile{
	public class NeuralNetwork{
		
		public List<Activation> Act = new List<Activation>();
	
		public List<Matrix> Weights = new List<Matrix>();
		
		private List<Matrix> WeightChanges = new List<Matrix>();
		
		public List<Matrix> Nodes = new List<Matrix>();
		
		private List<Matrix> NodeChanges = new List<Matrix>();
		
		private List<Matrix> Bias = new List<Matrix>();
		
		private List<Matrix> BiasChange = new List<Matrix>();
		
		private float LowerBounds;
		private float UpperBounds;
		
		private float learnRate;
		private float fixedRate;
		private int dropout;
		
		private Random rnd;
		
		public NeuralNetwork(){
			
		}
		
		public NeuralNetwork(int L, Random rd, float lR = .1f, float LowBound = 0, float UpBound = 1, int dropOut = 1){
			rnd = rd;
			fixedRate = lR;
			learnRate = lR;
			dropout = dropOut;
			LowerBounds = LowBound;
			UpperBounds = UpBound;
			
			Nodes.Add(new Matrix(1, L));
			
			Bias.Add(new Matrix(1, L, .2f));
			
			Weights.Add(new Matrix(1,1));
			
			Act.Add(new Default());
			
			ResetConstructives();
		}
		

		private Activation CheckType(string type){
			Activation tempAct;
			
			switch (type){
				case "Sigmoid":
					tempAct = new Sigmoid();
					break;
				case "HypTan":
					tempAct = new HyperbolicTangent();
					break;
				default:
					tempAct = new Default();
				break;
				
			}
			
			return tempAct;
		}
		
		private void NewWeights(int index){
			index--;
			
			for(int i = 0; i < Weights[index].length; i++)
			{
				for(int j = 0; j < Weights[index].height; j++)
					Weights[index][i,j] = (float)(Math.Sqrt(-2 * Math.Log(rnd.Next(1, 99) / 100f)) * Math.Cos(2 * 3.14159f * rnd.Next(1, 99) / 100f));
			}
		}
		
		
		
		
		public void AddLayer(int L, string type = "Default"){
			Weights.Add(new Matrix(Nodes[Nodes.Count-1].height, L, 1));
			
			NewWeights(Weights.Count-1);
			
			Nodes.Add(new Matrix(1, L));
			
			Bias.Add(new Matrix(1, L, .2f));
			
			Act.Add(CheckType(type));
			
			ResetConstructives();
		}
		
		
		public float Train(Epoch Ep){
			float totalError = 0;
			int counter = 0;
			for(int i = 0; i < Ep.Runs; i++){
				learnRate = (float)(Math.Sin(fixedRate*Ep.Runs)/.5f+2f);
				
				for(int j = 0; j < Ep.Set.Length; j++){
					float[] error = runNetwork(Ep.Set[j].Inputs, Ep.Normalize);
					for(int k = 0; k < error.Length; k++)
						error[k] -= Ep.Set[j].Goals[k];
					BackPropigate(error);
				}				
				ApplyBackProp(Ep.Set.Length);
			}
			for(int j = 0; j < Ep.Set.Length; j++){
				float[] Error = runNetwork(Ep.Set[j].Inputs, Ep.Normalize);
				float tempNum = 0;
				for(int i = 0; i < Error.Length; i++)
					tempNum += ((float)Math.Pow(Error[i]-Ep.Set[j].Goals[i], 2));
				totalError += tempNum / Error.Length;
				
				counter ++;
			}
			return totalError /= counter;
		}
		
		
		
		
		private void ApplyBackProp(int Input){
			
			
			for(int i = 1; i < Weights.Count; i++){
				if(rnd.Next(1, dropout) != 1 && dropout != 1)
				{
					Weights[i] = Weights[i] - (WeightChanges[i]/Input);
				}else{
					if(dropout != 1){
						Weights[i] = Weights[i] - (WeightChanges[i]/Input);
					}
				}
			}
			for(int i = 0; i < Bias.Count; i++)
				for(int j = 0; j < Bias[i].height; j++)
					Bias[i][0,j] -= BiasChange[i][0,j]/Input;
			ResetConstructives();
		}
		
		
		private void ResetConstructives(){
			BiasChange = new List<Matrix>();
			for(int i = 0; i < Bias.Count; i++){
				BiasChange.Add(new Matrix(1, Bias[i].height, 0));
			}
			
			WeightChanges = new List<Matrix>();
			WeightChanges.Add(new Matrix(1,1));
			for(int i = 1; i < Nodes.Count; i++){
				WeightChanges.Add(new Matrix(Nodes[i-1].height, Nodes[i].height, 0));	
			}
			
			ResetNodeChanges();
		}
		
		private void ResetNodeChanges(){
			NodeChanges = new List<Matrix>();
			for(int i = 0; i < Nodes.Count; i++)
				NodeChanges.Add(new Matrix(1, Nodes[i].height, 0));
		}
		
		
		
		public float[] BackPropigate(float[] Error){
			for(int i = 0; i < NodeChanges[NodeChanges.Count-1].height; i++){
				NodeChanges[NodeChanges.Count-1][0, i] = Error[i] * Act[Act.Count-1].ActiveFunctionDeriv(Nodes[NodeChanges.Count-1][0, i]);
			}
			
			for(int i = NodeChanges.Count - 2; i >= 0; i--)
				for(int j = 0; j < NodeChanges[i].length; j++)
					for(int k = 0; k < Weights[i+1].height; k++)
						NodeChanges[i][0, j] += NodeChanges[i+1][0, k] * Weights[i+1][j, k] * Act[i].ActiveFunctionDeriv(Nodes[i][0, j]);
					
			for(int i = 0; i < BiasChange.Count; i++)
				for(int j = 0; j < BiasChange[i].height; j++){
					BiasChange[i][0, j] = Bias[i][0, j] * NodeChanges[i][0, j] * learnRate;
				}
			
			for(int i = 1; i < WeightChanges.Count; i++)
				for(int j = 0; j < WeightChanges[i].height; j++)
					for(int k = 0; k < WeightChanges[i].length; k++)
						WeightChanges[i][k, j] += NodeChanges[i][0, j] * Nodes[i-1][0, k] * learnRate;
			
			
		
			float[] toReturn = new float[NodeChanges[0].height];
			for(int i = 0; i < toReturn.Length; i++){
				toReturn[i] = NodeChanges[0][0,i];
			}
			
			ResetNodeChanges();
		
			return toReturn;
		}
		
		
		
		
		public float[] OutputNetwork(InOutPair IOP, bool Normalize = false){
			float[] returnArray = runNetwork(IOP.Inputs, Normalize);
			Console.WriteLine("");
			Console.WriteLine("The output of the network is :");
			for(int i = 0; i < returnArray.Length; i++){
				Console.WriteLine("Node " + (i+1) + ": " + returnArray[i]);
			}
			Console.WriteLine("");
			return returnArray;
		}
		
		
		
		
		public float[] runNetwork(float[] Inputs, bool Normalize = false){

			if(Normalize){
				float[] toNormalize = new float[Inputs.Length];
				for(int i = 0; i < toNormalize.Length; i++)
					toNormalize[i] = (Inputs[i] - LowerBounds) / (UpperBounds - LowerBounds);
				for(int i = 0; i < Nodes[0].height; i++)
					Nodes[0][0, i] = toNormalize[i];
			}else
				for(int i = 0; i < Nodes[0].height; i++)
					Nodes[0][0, i] = Inputs[i];
			
			
			for(int i = 1; i < Nodes.Count; i++){
				Nodes[i] = ActivationMatrix((Weights[i] * Nodes[i-1]) + Bias[i], i);
			
			}
			float[] toReturn = new float[Nodes[Nodes.Count-1].height];
			
			for(int i = 0; i <  Nodes[Nodes.Count-1].height; i++)
				toReturn[i] = Nodes[Nodes.Count-1][0, i];
			
			return toReturn;
		}
		
		
		private Matrix ActivationMatrix(Matrix input, int index){
			Matrix temp = new Matrix(input.length, input.height, 0);
			
			for(int i = 0; i < input.length; i++){
				for(int j = 0; j < input.height; j++){
					temp[i,j] = Act[index].ActiveFunction(input[i,j]);
				}
			}
			
			return temp;
		}
	}
}







