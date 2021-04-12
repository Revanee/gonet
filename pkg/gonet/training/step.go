package training

func stepNeuron(weights []float64, bias float64, weightsGradient []float64, biasGradient float64) (newWeights []float64, newBias float64) {
	stepSize := .03
	newWeights = make([]float64, len(weights))
	for w := range newWeights {
		newWeights[w] = weights[w] - weightsGradient[w]*stepSize
	}
	newBias = bias - biasGradient*stepSize
	return
}

func stepLayer(weights [][]float64, biases []float64, weightsGradient [][]float64, biasesGradient []float64) (newWeights [][]float64, newBias []float64) {
	newWeights = make([][]float64, len(weights))
	newBias = make([]float64, len(biases))
	for n := range weights {
		newWeights[n], newBias[n] = stepNeuron(weights[n], biases[n], weightsGradient[n], biasesGradient[n])
	}
	return
}

func stepNetwork(weights [][][]float64, biases [][]float64, weightsGradient [][][]float64, biasesGradient [][]float64) (newWeights [][][]float64, newBias [][]float64) {
	newWeights = make([][][]float64, len(weights))
	newBias = make([][]float64, len(biases))
	for l := range weights {
		newWeights[l], newBias[l] = stepLayer(weights[l], biases[l], weightsGradient[l], biasesGradient[l])
	}
	return
}
