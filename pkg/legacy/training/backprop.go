package training

import gonet "github.com/Revanee/gonet/pkg/legacy"

func backpropLayer(inputs, outputs, expectedOutputs []float64, weigths [][]float64, biases []float64) (weigthsGradient [][]float64, biasGradient, inputGradient []float64) {
	activationToCostRatios := activationsToCostsRatios(outputs, expectedOutputs)
	costs := activationsCosts(outputs, expectedOutputs)
	zs := make([]float64, len(biases))
	for i := range zs {
		zs[i] = gonet.WeightedSumAndBias(weigths[i], biases[i], inputs)
	}
	zsToActivationsRatios := zsToActivationsRatios(zs)
	biasesToZsRatios := biasesToZsRatios(biases)
	biasGradient = make([]float64, len(biases))
	for i := range biasGradient {
		biasRatio := biasToActivationCostRatio(biasesToZsRatios[i], zsToActivationsRatios[i], activationToCostRatios[i])
		biasGradient[i] = biasRatio * costs[i]
	}
	weigthsGradient = make([][]float64, len(weigths))
	weightsToZsRatios := weightsToZsRatios(inputs, weigths)
	for n := range weigthsGradient {
		weigthsGradient[n] = make([]float64, len(weigths[n]))
		for w := range weigths[n] {
			weightToActivationCostRatio := weightToActivationCostRatio(weightsToZsRatios[n][w], zsToActivationsRatios[n], activationToCostRatios[n])
			weigthsGradient[n][w] = weightToActivationCostRatio * costs[n]
		}
	}
	inputGradientSum := make([]float64, len(inputs))
	for w := range inputs {
		for n := range outputs {
			previousActivationToZRatio := previousActivationToZRatio(weigths[n][w])
			previousActivationToCostRatio := previousActivationToCostRatio(previousActivationToZRatio, zsToActivationsRatios[n], activationToCostRatios[n])
			inputGradientSum[w] += previousActivationToCostRatio * costs[n]
		}
	}
	return weigthsGradient, biasGradient, inputGradientSum
}
