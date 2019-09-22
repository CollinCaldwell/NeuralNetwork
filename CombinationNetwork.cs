using System;
using System.Collections.Generic;
using Utility;
using ConvolutionFile;
using NeuralNetworkFile;
using ActivationTemplate;
	
	
namespace CombinationFile{
	public class Combination{
		
		public Convolution Conv;
		public NeuralNetwork Net;
		
		
		public Combination(Convolution C, NeuralNetwork N){
			Conv = C;
			Net = N;
			
			
		}
		
		public float[] Run(float[] Input){
			return Net.runNetwork(Conv.RunNetwork(Input));
		}
		
		
		public void Train(Epoch E){
			for(int i = 0; i < E.Runs; i++){
				for(int j = 0; j < E.Set.Length; j++){
					float[] convError;
					float[] error = Net.runNetwork(Conv.RunNetwork(E.Set[j].Inputs));
					for(int k = 0; k < error.Length; k++)
						error[k] -= E.Set[j].Goals[k];
					convError = Net.BackPropigate(error);
					Conv.Propigate(convError);
					Net.Train(new Epoch(new InOutPair[] {E.Set[0]}, 1));
				}
			}
			
			
		}
		
	}
}