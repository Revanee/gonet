package training

import gonet "github.com/Revanee/gonet/pkg"

func stepNeuron(bias, output, expectedOutput float64, weights, inputs []float64) (newWeights []float64, newBias float64) {
	z := gonet.WeightedSumAndBias(weights, bias, inputs)

	activationToCostRatio := activationToCostRatio(output, expectedOutput)
	activationCost := activationCost(output, expectedOutput)

	biasToZRatio := biasToZRatio()
	zToActivationRatio := zToActivationRatio(z)
	biasToActivationCostRatio := biasToActivationCostRatio(biasToZRatio, zToActivationRatio, activationToCostRatio)

	newBias = bias - activationCost*biasToActivationCostRatio

	weightsToActivationCostRatios := weightsToActivationCostRatios(activationToCostRatio, zToActivationRatio, weights, inputs)

	newWeights = make([]float64, len(weights))
	for i := range newWeights {
		newWeights[i] = weights[i] - activationCost*weightsToActivationCostRatios[i]
	}

	return
}

func stepNeurons(weights [][]float64, biases, activations, previousActivations, desireds []float64) (newWeights [][]float64, newBiases []float64) {
	newWeights = make([][]float64, len(weights))
	newBiases = make([]float64, len(biases))

	for i := range weights {
		newWeights[i], newBiases[i] = stepNeuron(biases[i], activations[i], desireds[i], weights[i], previousActivations)
	}

	return
}

func StepNeuron(neuron gonet.Neuron, output, expectedOutput float64, inputs []float64) gonet.Neuron {
	newWeights, newBias := stepNeuron(neuron.GetBias(), output, expectedOutput, neuron.GetWeigths(), inputs)
	return gonet.NewNeuron(newBias, newWeights...)
}

func StepNeurons(neurons []gonet.Neuron, outputs, expectedOutputs, inputs []float64) []gonet.Neuron {
	newNeurons := make([]gonet.Neuron, len(neurons))
	for i := range neurons {
		newNeurons[i] = StepNeuron(neurons[i], outputs[i], expectedOutputs[i], inputs)
	}
	return newNeurons
}

func StepLayer(layer gonet.Layer, outputs, expectedOutputs, inputs []float64) gonet.Layer {
	neurons := StepNeurons(layer.GetNeurons(), outputs, expectedOutputs, inputs)
	return gonet.NewLayer(neurons...)
}
