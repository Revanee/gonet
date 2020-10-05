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

func getNeuronGradient(weights []float64, bias, cost float64, inputs []float64) neuronGradient {
	biasToZGradient := biasToZRatio()
	zToActivationGradient := zToActivationRatio(gonet.WeightedSumAndBias(weights, bias, inputs))

	inputsGradients := make([]float64, len(inputs))
	for i := range inputsGradients {
		inputToZGradient := previousActivationToZRatio(weights[i])
		inputsGradients[i] = getInputGradient(inputToZGradient, zToActivationGradient, cost)
	}

	weigthsGradients := make([]float64, len(weights))
	weightToZGradients := weightsToZRatios(inputs)
	for i := range weigthsGradients {
		weigthsGradients[i] = getWeightGradient(weightToZGradients[i], zToActivationGradient, cost)
	}
	biasGradient := getBiasGradient(biasToZGradient, zToActivationGradient, cost)
	return neuronGradient{
		biasGradient:     biasGradient,
		inputsGradients:  inputsGradients,
		weightsGradients: weigthsGradients,
	}
}

func getLayerGradient(layer gonet.Layer, costs, inputs []float64) layerGradient {
	neurons := layer.GetNeurons()
	neuronGradients := make([]neuronGradient, len(neurons))
	for i, neuron := range neurons {
		neuronGradients[i] = getNeuronGradient(neuron.GetWeigths(), neuron.GetBias(), costs[i], inputs)
	}
	return layerGradient{
		neuronGradients: neuronGradients,
	}
}

func getNetworkGradient(network gonet.Network, inputs, expectedOutput []float64) networkGradient {
	layers := network.GetLayers()

	layerOutputs, _ := getLayerOutputs(network, inputs)
	layerInputs := append([][]float64{inputs}, layerOutputs...)

	layerGradients := make([]layerGradient, len(layers))
	layersActivationCosts := make([][]float64, len(layers))
	for l := len(layers) - 1; l >= 0; l-- {
		neurons := layers[l].GetNeurons()

		layersActivationCosts[l] = make([]float64, len(neurons))
		if l == len(layers)-1 {
			for n := range neurons {
				layersActivationCosts[l][n] = activationToCostRatio(layerOutputs[l][n], expectedOutput[n])
			}
		} else {
			expectedOutputGradients := addNeuronGradients(layerGradients[l+1].neuronGradients...).inputsGradients
			for n := range neurons {
				layersActivationCosts[l][n] = expectedOutputGradients[n]
			}
		}
		layerGradients[l] = getLayerGradient(layers[l], layersActivationCosts[l], layerInputs[l])
	}
	return networkGradient{
		layerGradients: layerGradients,
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
