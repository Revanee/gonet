package backprop

func BiasRatio(z, activationToCostRatio float64) float64 {
	zToActivationRatio := ZToActivationRatio(z)
	biasToZRatio := BiasToZRatio()
	biasToActivationCostRatio := BiasToActivationCostRatio(biasToZRatio, zToActivationRatio, activationToCostRatio)
	return biasToActivationCostRatio
}

func WeightsRatio(z, activationToCostRatio float64, weigths, inputs []float64) []float64 {
	zToActivationRatio := ZToActivationRatio(z)
	weightsToActivationCostRatios := WeightsToActivationCostRatios(activationToCostRatio, zToActivationRatio, weigths, inputs)
	gradients := make([]float64, len(weigths))
	for i := range gradients {
		gradients[i] = weightsToActivationCostRatios[i]
	}
	return gradients
}

func InputsRatio(z, activationToCostRatio float64, inputs, weights []float64) []float64 {
	zToActivationRatio := ZToActivationRatio(z)
	inputsToCostRatio := make([]float64, len(inputs))
	for i := range inputsToCostRatio {
		inputToZRatio := PreviousActivationToZRatio(weights[i])
		inputsToCostRatio[i] = PreviousActivationToCostRatio(inputToZRatio, zToActivationRatio, activationToCostRatio)
	}
	inputsGradient := make([]float64, len(inputs))
	for i := range inputsGradient {
		inputsGradient[i] = inputsToCostRatio[i]
	}
	return inputsGradient
}

func LayerRatio(inputs, activations, activationToCostRatios []float64, weigths [][]float64, biases []float64) (weigthsRatio, inputRatio [][]float64, biasRatio []float64) {
	biasRatio = make([]float64, len(biases))
	weigthsRatio = make([][]float64, len(weigths))
	inputsRatio := make([][]float64, len(weigths))
	for n := range activations {
		weigthsRatio[n], inputsRatio[n], biasRatio[n] = NeuronRatio(activations[n], activationToCostRatios[n], biases[n], inputs, weigths[n])
	}
	return weigthsRatio, inputsRatio, biasRatio
}

func NeuronRatio(activation, activationToCostRatio, bias float64, inputs, weights []float64) (weigthsRatio, inputsRatio []float64, biasRatio float64) {
	z := LogisticCurveDerivative(activation)
	weigthsRatio = WeightsRatio(z, activationToCostRatio, weights, inputs)
	inputsRatio = InputsRatio(z, activationToCostRatio, inputs, weights)
	biasRatio = BiasRatio(z, activationToCostRatio)
	return
}

func NetworkRatio(activations, activationsToCostsRatios []float64, weigths [][][]float64, biases, inputs [][]float64) (weigthsRatio, inputRatio [][][]float64, biasRatio [][]float64) {
	weigthsRatio = make([][][]float64, len(weigths))
	inputRatio = make([][][]float64, len(biases))
	biasRatio = make([][]float64, len(biases))
	lastLayerIndex := len(biases) - 1

	weigthsRatio[lastLayerIndex], inputRatio[lastLayerIndex], biasRatio[lastLayerIndex] = LayerRatio(inputs[lastLayerIndex], activations, activationsToCostsRatios, weigths[lastLayerIndex], biases[lastLayerIndex])
	for l := lastLayerIndex - 1; l >= 0; l-- {
		layerActivationToCostRatio := inputRatio[l+1]
		averageLayerActivationToCostRatio := make([]float64, len(layerActivationToCostRatio[0]))
		for n := range averageLayerActivationToCostRatio {
			for i := range layerActivationToCostRatio {
				averageLayerActivationToCostRatio[n] += layerActivationToCostRatio[i][n]
			}
			averageLayerActivationToCostRatio[n] = averageLayerActivationToCostRatio[n] / float64(len(layerActivationToCostRatio[0]))
		}
		weigthsRatio[l], inputRatio[l], biasRatio[l] = LayerRatio(inputs[l], inputs[l+1], averageLayerActivationToCostRatio, weigths[l], biases[l])
	}
	return
}

func NetworkGradient(activations, expected []float64, weigths [][][]float64, biases, inputs [][]float64) (weigthsGradient [][][]float64, biasGradient [][]float64) {
	weigthsGradient = make([][][]float64, len(weigths))
	biasGradient = make([][]float64, len(weigths))
	activationsToCostsRatios := make([]float64, len(activations))
	for i := range activationsToCostsRatios {
		activationsToCostsRatios[i] = ActivationToCostRatio(activations[i], expected[i])
	}
	cost := .0
	for i := range activations {
		cost += ActivationCost(activations[i], expected[i])
	}

	weigthsRatio, _, biasRatio := NetworkRatio(activations, activationsToCostsRatios, weigths, biases, inputs)

	lastLayerIndex := len(weigths) - 1
	weigthsGradient[lastLayerIndex] = make([][]float64, len(weigthsRatio[lastLayerIndex]))
	biasGradient[lastLayerIndex] = make([]float64, len(weigthsRatio[lastLayerIndex]))
	for n := range weigthsRatio[lastLayerIndex] {
		weigthsGradient[lastLayerIndex][n] = make([]float64, len(weigthsRatio[lastLayerIndex][n]))
		for w := range weigthsRatio[lastLayerIndex][n] {
			weigthsGradient[lastLayerIndex][n][w] += weigthsRatio[lastLayerIndex][n][w] * cost
		}
		biasGradient[lastLayerIndex][n] = biasRatio[lastLayerIndex][n] * cost
	}

	for l := len(weigths) - 2; l >= 0; l-- {
		weigthsGradient[l] = make([][]float64, len(weigthsRatio[l]))
		biasGradient[l] = make([]float64, len(weigthsRatio[l]))
		for n := range weigthsRatio[l] {
			weigthsGradient[l][n] = make([]float64, len(weigthsRatio[l][n]))
			for w := range weigthsRatio[l][n] {
				weigthsGradient[l][n][w] += weigthsRatio[l][n][w] * cost
			}
			biasGradient[l][n] += biasRatio[l][n] * cost
		}
	}

	return
}
