package training

import gonet "github.com/Revanee/gonet/pkg/legacy"

func TrainNetworkSingle(network gonet.Network, input, expectedOutput []float64) gonet.Network {
	networkGradient := getNetworkGradient(network, input, expectedOutput)
	return stepNetwork(network, networkGradient)
}

func TrainNetworkBatch(network gonet.Network, inputs, expectedOutputs [][]float64) gonet.Network {
	networkGradients := make([]networkGradient, len(inputs))
	for i := range networkGradients {
		networkGradients[i] = getNetworkGradient(network, inputs[i], expectedOutputs[i])
	}
	networkGradientSum := addNetworkGradients(networkGradients...)
	return stepNetwork(network, networkGradientSum)
}
