using System;
using System.Collections.Generic;
using Utility;
using ActivationTemplate;
	
namespace ConvolutionFile{
	public class Convolution{
		
		public List<float[]> Network = new List<float[]>();
		public List<float[]> Filters = new List<float[]>();
		public List<float> FilterHeights = new List<float>();
		public List<float> Heights = new List<float>();
		public List<int> Jump = new List<int>();
		public List<int> Method = new List<int>();
		
		
		public List<float[]> NodeErrors = new List<float[]>();
		public List<float[]> FilterErrors = new List<float[]>();
		
		public List<Activation> Act = new List<Activation>();
		
		public float[] Input;
		
		
		
		public Convolution(int h, int w){
			
			Network.Add(new float[h*w]);
			NodeErrors.Add(new float[h*w]);
			Filters.Add(new float[0]);
			FilterErrors.Add(new float[0]);
			FilterHeights.Add(1);
			Heights.Add(h);
			Jump.Add(1);
			Method.Add(0);
			Act.Add(new Default());
			
		}
		
		
		public float Train(Epoch E){
			for(int i = 0; i < E.Runs; i++){
				for(int j = 0; j < E.Set.Length; j++){
					RunNetwork(E.Set[j].Inputs);
					Propigate(E.Set[j].Goals);
				}
			}
			
			float error = 0;
			for(int j = 0; j < E.Set.Length; j++){
				RunNetwork(E.Set[j].Inputs);
				error += Propigate(E.Set[j].Goals);
			}
			return error/E.Set.Length;
		}
		
		
		
		public void AddLayer(int fH, int fW, int j, Activation A){
			AddEntireLayer(fH, fW, j, false);
			Act.Add(A);
		}
		
		
		
		public void AddLayer(int fH, int fW, int j = 1, bool pool = false){
			AddEntireLayer(fH, fW, j, pool);
			Act.Add(new Default());
		}
		
		private void AddEntireLayer(int fH, int fW, int j = 1, bool pool = false){
			
			int index = Filters.Count;
			
			
			if(pool)
				Method.Add(1);
			else
				Method.Add(0);
			
			Filters.Add(new float[fH * fW]);
			FilterErrors.Add(new float[fH*fW]);
			for(int i = 0; i < Filters[Filters.Count-1].Length; i++){
				Filters[index][i] = .1f;
				FilterErrors[index][i] = 0;
			}
			
			
			FilterHeights.Add(fH);
			Jump.Add(j);
			
			float tempNum = Heights[index-1] - (FilterHeights[index] - 1);
			if(j != 1)
				tempNum = Heights[index-1] / j;
			Heights.Add(tempNum);
			
			NodeErrors.Add(new float[0]);
			Network.Add(new float[0]);
			if(j == 1){
				Network[Network.Count-1] = new float[(int)(tempNum * ((Network[index-1].Length/Heights[index-1]) - ((Filters[index].Length / FilterHeights[index]) - 1)))];
				NodeErrors[NodeErrors.Count-1] = new float[Network[Network.Count-1].Length];
			}else{
				Network[Network.Count-1] = new float[(int)tempNum * ((int)(Network[index-1].Length/Heights[index-1])/j)];
				NodeErrors[NodeErrors.Count-1] = new float[Network[Network.Count-1].Length];
			}
		}
		
		
		
		public float[] RunNetwork(float[] Input){
			Network[0] = Input;
			for(int i = 0; i < Network.Count; i++){
				calculateLayer(i);
			}
			return Network[Network.Count-1];
		}
		
		
		
		
		public float Propigate(float[] Desired, int runs = 1){
			float[] error = new float[Network[Network.Count-1].Length];
			
			for(int i = 0; i < error.Length; i++){
				error[i] = Network[Network.Count-1][i] - Desired[i];
				NodeErrors[NodeErrors.Count-1][i] = error[i] * Act[Act.Count-1].ActiveFunctionDeriv(Network[Network.Count-1][i]);
			}
			
			for(int i = Network.Count-1; i > 0; i--){
				propigateLayer(i);
			}
			
			ApplyError(runs);
			
			float totalError = 0;
			for(int i = 0; i < error.Length; i++){
				totalError += (float)Math.Pow(error[i], 2)/2;
			}
			return totalError / error.Length;
		}
		
		private void calculateLayer(int layer){
			int newWidth = (int)(Network[layer].Length / Heights[layer]);
			
			if(Jump[layer] != 1){
				newWidth = (int)(Network[layer-1].Length/Heights[layer-1])/(int)Jump[layer];
			}
			
			int hCounter = 0;
			int wCounter = 0;
			
			if(layer != 0){
				for(int i = 0; i < Network[layer].Length; i++){
					Network[layer][i] = 0;	
				}
			}
			
			for(int i = 0; i < Network[layer].Length; i++){
				
				
				wCounter += Jump[layer];
				if(wCounter >= newWidth){
					wCounter = 0;
					hCounter += Jump[layer];
				}
				
				float[] temp = new float[Filters[layer].Length];
				
				for(int j = 0; j < FilterHeights[layer]; j++){
					for(int k = 0; k < Filters[layer].Length/FilterHeights[layer]; k++){
						if(Method[layer] == 0){
							Network[layer][i] += Network[layer-1][(int)((FilterHeights[layer] * (j+hCounter)) + (k+wCounter))] * Filters[layer][(int)(Filters[layer].Length/FilterHeights[layer]*j) + k];
							
						}else{
							if(Method[layer] == 1){
								temp[(int)(FilterHeights[layer]*j) + k] = Network[layer-1][(int)((FilterHeights[layer] * (j+hCounter)) + (k+wCounter))];
							}
						}
					}
					
				}
				
				
				if(Method[layer] == 1){
					float max = temp[0];
					for(int j = 0; j < temp.Length; j++){
						if(temp[j] >= max)
							max = temp[j];
					}
					Network[layer][i] = max;	
				}else{
					Network[layer][i] = Act[layer].ActiveFunction(Network[layer][i]);
				}
				
			}
			
		}
		
		
		private void propigateLayer(int layer){
			
			int newWidth = (int)(Network[layer].Length / Heights[layer]);
			
			if(Jump[layer] != 1){
				newWidth = (int)(Network[layer-1].Length/Heights[layer-1])/(int)Jump[layer];
			}
			
			int hCounter = 0;
			int wCounter = 0;
			
			if(layer != 1){
				for(int i = 0; i < Network[layer].Length; i++){
					NodeErrors[layer-1][i] = 0;	
				}
			}
			
			
			
			float[] tempFilter = new float[Filters[layer].Length];
			
			for(int i = 0; i < tempFilter.Length; i++){
				tempFilter[i] = 0;
			}
			
			
			for(int i = 0; i < Network[layer].Length; i++){
				wCounter += Jump[layer];
				if(wCounter >= newWidth){
					wCounter = 0;
					hCounter += Jump[layer];
				}
				
				float[] temp = new float[Filters[layer].Length];
				int[] indices = new int[Filters[layer].Length];
				
				for(int j = 0; j < FilterHeights[layer]; j++){
					for(int k = 0; k < Filters[layer].Length/FilterHeights[layer]; k++){
						if(Method[layer] == 0){
							tempFilter[(int)(Filters[layer].Length/FilterHeights[layer]*j) + k] += Network[layer-1][(int)((FilterHeights[layer] * (j+hCounter)) + (k+wCounter))] * NodeErrors[layer][i];
							NodeErrors[layer-1][(int)((FilterHeights[layer] * (j+hCounter)) + (k+wCounter))] += NodeErrors[layer][i] * Filters[layer][(int)(Filters[layer].Length/FilterHeights[layer]*j) + k] * Act[layer-1].ActiveFunctionDeriv(Network[layer-1][(int)((FilterHeights[layer] * (j+hCounter)) + (k+wCounter))]);
						}else{
							if(Method[layer] == 1){
								indices[(int)(FilterHeights[layer]*j) + k] = (int)((FilterHeights[layer] * (j+hCounter)) + (k+wCounter));
								temp[(int)(FilterHeights[layer]*j) + k] = Network[layer-1][(int)((FilterHeights[layer] * (j+hCounter)) + (k+wCounter))];
							}
						}
					}
					
				}
				
				if(Method[layer] == 1){
					int index = 0;
					float max = temp[0];
					for(int j = 0; j < temp.Length; j++){
						if(temp[j] >= max){
							index = indices[j];
							max = temp[j];
						}
					}
					NodeErrors[layer-1][index] += max;	
				}
					
			}
			
			for(int i = 0; i < Filters[layer].Length; i++){
				FilterErrors[layer][i] += tempFilter[i]/Network[layer].Length;
			}
				
			
			
		}
		
		public void ApplyError(int total){
			for(int i = 0; i < FilterErrors.Count; i++){
				for(int j = 0; j < FilterErrors[i].Length; j++){
					Filters[i][j] -= FilterErrors[i][j] / total * .01f;
					FilterErrors[i][j] = 0;
				}
			}
		}
	}


}
