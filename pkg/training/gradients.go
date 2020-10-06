package training

import (
	gonet "github.com/Revanee/gonet/pkg"
	"github.com/pkg/errors"
)

type neuronGradient struct {
	biasGradient     float64
	weightsGradients []float64
	inputsGradients  []float64
}

type layerGradient struct {
	neuronGradients []neuronGradient
}

type networkGradient struct {
	layerGradients []layerGradient
}

func getInputGradient(inputToZGradient, zToActivationGradient, activationToCostGradient float64) float64 {
	return inputToZGradient * zToActivationGradient * activationToCostGradient
}

func getBiasGradient(biasToZGradient, zToActivationGradient, activationToCostGradient float64) float64 {
	return biasToZGradient * zToActivationGradient * activationToCostGradient
}

func getWeightGradient(weightToZGradient, zToActivationGradient, activationToCostGradient float64) float64 {
	return weightToActivationCostRatio(weightToZGradient, zToActivationGradient, activationToCostGradient)
}

func getNeuronGradient(weights []float64, bias, activationToCostGradients float64, inputs []float64) neuronGradient {
	biasToZGradient := biasToZRatio()
	zToActivationGradient := zToActivationRatio(gonet.WeightedSumAndBias(weights, bias, inputs))

	inputsGradients := make([]float64, len(inputs))
	for i := range inputsGradients {
		inputToZGradient := previousActivationToZRatio(weights[i])
		inputsGradients[i] = getInputGradient(inputToZGradient, zToActivationGradient, activationToCostGradients)
	}

	weigthsGradients := make([]float64, len(weights))
	weightToZGradients := weightsToZRatios(inputs)
	for i := range weigthsGradients {
		weigthsGradients[i] = getWeightGradient(weightToZGradients[i], zToActivationGradient, activationToCostGradients)
	}
	biasGradient := getBiasGradient(biasToZGradient, zToActivationGradient, activationToCostGradients)
	return neuronGradient{
		biasGradient:     biasGradient,
		inputsGradients:  inputsGradients,
		weightsGradients: weigthsGradients,
	}
}

func getLayerFullGradient(layer gonet.Layer, activationToCostGradients, inputs []float64) layerGradient {
	neurons := layer.GetNeurons()
	neuronGradients := make([]neuronGradient, len(neurons))
	for i, neuron := range neurons {
		neuronGradients[i] = getNeuronGradient(neuron.GetWeigths(), neuron.GetBias(), activationToCostGradients[i], inputs)
	}
	return layerGradient{
		neuronGradients: neuronGradients,
	}
}

func getNetworkGradient(network gonet.Network, input, expectedOutput []float64) networkGradient {
	layers := network.GetLayers()

	layerOutputs, _ := getLayerOutputs(network, input)
	layerInputs := append([][]float64{input}, layerOutputs...)[:len(layerOutputs)]

	layerOutputGradients := make([][]float64, len(layers))
	layerFullGradients := make([]layerGradient, len(layers))

	for l := len(layers) - 1; l >= 0; l-- {
		if l == len(layers)-1 {
			layerOutputGradients[l] = make([]float64, len(layerOutputs[l]))
			for o := range layerOutputs[l] {
				outputGradient := activationToCostRatio(layerOutputs[l][o], expectedOutput[o])
				layerOutputGradients[l][o] = outputGradient
			}
		}

		if l != 0 {
			layerOutputGradients[l-1] = make([]float64, len(layerInputs[l]))
		}
		for i := range layerInputs[l] {
			if l == 0 {
				continue
			}
			neurons := layers[l].GetNeurons()

			inputGradientSum := .0
			for n := range neurons {
				z := gonet.WeightedSumAndBias(neurons[n].GetWeigths(), neurons[n].GetBias(), layerInputs[l])
				zToActivationRatio := zToActivationRatio(z)
				previousActivationToZRatio := previousActivationToZRatio(neurons[n].GetWeigths()[i])
				inputGradientSum += previousActivationToCostRatio(previousActivationToZRatio, zToActivationRatio, layerOutputGradients[l][i])
			}

			inputGradientAvg := inputGradientSum / float64(len(neurons))

			inputGradient := activationToCostRatio(layerOutputs[l-1][i], layerOutputs[l-1][i]-inputGradientAvg)

			layerOutputGradients[l-1][i] = inputGradient
			continue
		}

		layerFullGradients[l] = getLayerFullGradient(layers[l], layerOutputGradients[l], layerInputs[l])
	}

	return networkGradient{
		layerGradients: layerFullGradients,
	}
}

func getLayerOutputs(network gonet.Network, inputs []float64) ([][]float64, error) {
	layers := network.GetLayers()
	outputs := make([][]float64, len(layers))
	previousLayerOutputs := inputs
	for i := range outputs {
		layerOutputs, err := gonet.ActivateLayer(layers[i], previousLayerOutputs...)
		if err != nil {
			return nil, errors.Wrap(err, "Could not activate layer")
		}
		outputs[i] = layerOutputs
		previousLayerOutputs = layerOutputs
	}
	return outputs, nil
}

func addNeuronGradients(gradients ...neuronGradient) neuronGradient {
	biasSum := 0.0
	weightsSum := make([]float64, len(gradients[0].weightsGradients))
	inputsSum := make([]float64, len(gradients[0].inputsGradients))

	for _, gradient := range gradients {
		biasSum += gradient.biasGradient
		for i, weight := range gradient.weightsGradients {
			weightsSum[i] += weight
		}
		for i, input := range gradient.inputsGradients {
			inputsSum[i] += input
		}
	}
	return neuronGradient{
		biasGradient:     biasSum,
		weightsGradients: weightsSum,
		inputsGradients:  inputsSum,
	}
}

func addLayerGradients(layerGradients ...layerGradient) layerGradient {
	neuronGradients := make([][]neuronGradient, len(layerGradients))
	for i := range neuronGradients {
		neuronGradients[i] = layerGradients[i].neuronGradients
	}
	summedNeuronGradients := make([]neuronGradient, len(layerGradients[0].neuronGradients))
	for i := range summedNeuronGradients {
		summedNeuronGradients[i] = addNeuronGradients(layerGradients[i].neuronGradients...)
	}
	return layerGradient{
		neuronGradients: summedNeuronGradients,
	}
}

func addNetworkGradients(networkGradients ...networkGradient) networkGradient {
	layerGradients := make([][]layerGradient, len(networkGradients))

	for i := range layerGradients {
		layerGradients[i] = networkGradients[i].layerGradients
	}

	summedLayerGradients := make([]layerGradient, len(networkGradients[0].layerGradients))
	for i := range summedLayerGradients {
		summedLayerGradients[i] = addLayerGradients(layerGradients[i]...)
	}

	return networkGradient{
		layerGradients: summedLayerGradients,
	}
}

// func avgInputToZRatio(neurons ...gonet.Neuron) []float64 {
// 	numWeights := len(neurons[0].GetWeigths())
// 	total := make([]float64, numWeights)
// 	average := make([]float64, numWeights)
// 	for _, neuron := range neurons {
// 		for w, weight := range neuron.GetWeigths() {
// 			total[w] += previousActivationToZRatio(weight)
// 		}
// 	}
// 	for a := range average {
// 		average[a] = total[a] / float64(numWeights)
// 	}
// 	return average
// }
