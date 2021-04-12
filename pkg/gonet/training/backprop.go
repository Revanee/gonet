package training

import (
	"github.com/Revanee/gonet/pkg/gonet/functions"
	"github.com/Revanee/gonet/pkg/gonet/network"
)

func BackPropNetworkBatch(n network.Network, inputs, expected [][]float64) network.Network {
	newNetwork := n
	for i := range inputs {
		activations, _ := n.Activate(inputs[i]...)
		cost := functions.ActivationsCostsAverage(activations, expected[i])
		weightsRatio, biasesRatio := networkRatio(n, inputs[i], expected[i])
		weightsGradient, biasesGradient := NetworkGradient(weightsRatio, biasesRatio, cost)
		weights, biases := newNetwork.GetTensor()
		newWeights, newBiases := stepNetwork(weights, biases, weightsGradient, biasesGradient)
		newNetwork = network.NewNetworkFromTensor(newWeights, newBiases)
	}
	return newNetwork
}

func networkRatio(n network.Network, inputs, expected []float64) (weightsRatio [][][]float64, biasesRatio [][]float64) {
	activations, _ := n.Activate(inputs...)
	internalActivations, _ := n.ActivateTransparent(inputs...)

	activationsToCostsRatios := make([]float64, len(activations))
	for a := range activationsToCostsRatios {
		activationsToCostsRatios[a] = functions.ActivationToCostRatio(activations[a], expected[a])
	}
	weights, biases := n.GetTensor()
	allInputs := append([][]float64{inputs}, internalActivations...)
	weightsRatio, biasesRatio = NetworkRatio(activations, activationsToCostsRatios, weights, biases, allInputs)
	return weightsRatio, biasesRatio
}
