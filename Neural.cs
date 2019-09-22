using System;

using CombinationFile;
using ConvolutionFile;
using NeuralNetworkFile;
using Utility;

public class Program
{
	public static void Main()
	{	
	
		int height = 100;
		int width = 100;
		
		
		float[] input = new float[height*width];
		
		for(int i = 0; i < input.Length; i++){
			input[i] = .5f;
		}
		
		
		Random rnd = new Random();
		
		InOutPair[] IO = new InOutPair[1];
		
		float[] toAdd = new float[height*width];
		
		for(int i = 0; i < height*width; i++){
			toAdd[i] = 1f/i;
		}
		IO[0] = new InOutPair(toAdd, new float[] {.1f, .5f, -.1f});
		
		
		Epoch Ep = new Epoch(IO, 1000, true);
		
		NeuralNetwork NN = new NeuralNetwork(9, rnd, .01f, 1, 10, 200);
		NN.AddLayer(5, "HypTan");
		NN.AddLayer(5, "HypTan");
		NN.AddLayer(5, "HypTan");
		NN.AddLayer(3, "HypTan");
		
		
		Convolution CC = new Convolution(height, width);
		CC.AddLayer(5, 5, 5);
		CC.AddLayer(2, 2, 1, new HyperbolicTangent());		
		CC.AddLayer(6, 6, 1);
		CC.AddLayer(2, 2, 2, true);
		CC.AddLayer(2, 2, 1, new HyperbolicTangent());	
		CC.AddLayer(2, 2, 2, true);
		
		
		Combination C = new Combination(CC, NN);
		float[] layerOut = C.Run(input);
		for(int i = 0; i < layerOut.Length; i++)
			Console.WriteLine(layerOut[i]);
		C.Train(Ep);
		Console.WriteLine();
		layerOut = C.Run(input);
		for(int i = 0; i < layerOut.Length; i++)
			Console.WriteLine(layerOut[i]);
	}
}

