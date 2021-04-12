package training

import (
	"github.com/Revanee/gonet/pkg/gonet/functions"
	"github.com/Revanee/gonet/pkg/gonet/network"
)

func BackPropNetworkBatch(n network.Network, inputs, expected [][]float64) network.Network {
	weightsGradientSum := make([][][]float64, len(n.GetLayers()))
	biasesGradientSum := make([][]float64, len(n.GetLayers()))
	for i := range inputs {
		weightsGradient, biasesGradient := networkGradient(n, inputs[i], expected[i])
		for l := range weightsGradient {
			weightsGradientSum[l] = make([][]float64, len(weightsGradient[l]))
			biasesGradientSum[l] = make([]float64, len(weightsGradient[l]))
			for n := range weightsGradient[l] {
				weightsGradientSum[l][n] = make([]float64, len(weightsGradient[l][n]))
				for w := range weightsGradient[l][n] {
					weightsGradientSum[l][n][w] += weightsGradient[l][n][w]
				}
				biasesGradientSum[l][n] += biasesGradient[l][n]
			}
		}
	}
	weights, biases := n.GetTensor()
	newWeights, newBiases := stepNetwork(weights, biases, weightsGradientSum, biasesGradientSum)
	return network.NewNetworkFromTensor(newWeights, newBiases)
}

func networkGradient(n network.Network, inputs, expected []float64) (weightsGradient [][][]float64, biasesGradient [][]float64) {
	activations, _ := n.Activate(inputs...)
	cost := functions.ActivationsCostsSum(activations, expected)

	activationsToCostsRatios := make([]float64, len(activations))
	for a := range activationsToCostsRatios {
		activationsToCostsRatios[a] = functions.ActivationToCostRatio(activations[a], expected[a])
	}
	weights, biases := n.GetTensor()
	internalActivations, _ := n.ActivateTransparent(inputs...)
	allInputs := append([][]float64{inputs}, internalActivations...)
	weightsRatio, biasesRatio := NetworkRatio(activations, activationsToCostsRatios, weights, biases, allInputs)
	weightsGradient, biasesGradient = NetworkGradient(weightsRatio, biasesRatio, cost)
	return
}
