package training

import "github.com/Revanee/gonet/pkg/gonet/functions"

func BiasRatio(z, activationToCostRatio float64) float64 {
	zToActivationRatio := functions.ZToActivationRatio(z)
	biasToZRatio := functions.BiasToZRatio()
	biasToActivationCostRatio := functions.BiasToActivationCostRatio(biasToZRatio, zToActivationRatio, activationToCostRatio)
	return biasToActivationCostRatio
}

func WeightsRatio(z, activationToCostRatio float64, weigths, inputs []float64) []float64 {
	zToActivationRatio := functions.ZToActivationRatio(z)
	weightsToActivationCostRatios := functions.WeightsToActivationCostRatios(activationToCostRatio, zToActivationRatio, weigths, inputs)
	ratios := make([]float64, len(weigths))
	for i := range ratios {
		ratios[i] = weightsToActivationCostRatios[i]
	}
	return ratios
}

func InputsRatio(z, activationToCostRatio float64, inputs, weights []float64) []float64 {
	zToActivationRatio := functions.ZToActivationRatio(z)
	inputsToCostRatio := make([]float64, len(inputs))
	for i := range inputsToCostRatio {
		inputToZRatio := functions.InputToZRatio(weights[i])
		inputsToCostRatio[i] = functions.InputToCostRatio(inputToZRatio, zToActivationRatio, activationToCostRatio)
	}
	inputsGradient := make([]float64, len(inputs))
	for i := range inputsGradient {
		inputsGradient[i] = inputsToCostRatio[i]
	}
	return inputsGradient
}

func LayerRatio(inputs, activations, activationToCostRatios []float64, weigths [][]float64, biases []float64) (weigthsRatio, inputsRatio [][]float64, biasRatio []float64) {
	biasRatio = make([]float64, len(biases))
	weigthsRatio = make([][]float64, len(weigths))
	inputsRatio = make([][]float64, len(weigths))
	for n := range activations {
		weigthsRatio[n], inputsRatio[n], biasRatio[n] = NeuronRatio(activations[n], activationToCostRatios[n], biases[n], inputs, weigths[n])
	}
	return weigthsRatio, inputsRatio, biasRatio
}

func NeuronRatio(activation, activationToCostRatio, bias float64, inputs, weights []float64) (weigthsRatio, inputsRatio []float64, biasRatio float64) {
	z := functions.WeightedSumAndBias(inputs, weights, bias)
	weigthsRatio = WeightsRatio(z, activationToCostRatio, weights, inputs)
	inputsRatio = InputsRatio(z, activationToCostRatio, inputs, weights)
	biasRatio = BiasRatio(z, activationToCostRatio)
	return
}

func NetworkRatio(activations, activationsToCostsRatios []float64, weigths [][][]float64, biases, inputs [][]float64) (weigthsRatio [][][]float64, biasRatio [][]float64) {
	weigthsRatio = make([][][]float64, len(weigths))
	inputRatio := make([][][]float64, len(biases))
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
