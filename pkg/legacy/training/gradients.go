package training

import (
	gonet "github.com/Revanee/gonet/pkg/legacy"
	"github.com/Revanee/gonet/pkg/legacy/backprop"
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

func getBiasRatio(biasToZGradient, zToActivationGradient, activationToCostGradient float64) float64 {
	return biasToZGradient * zToActivationGradient * activationToCostGradient
}

func getWeightRatio(weightToZGradient, zToActivationGradient, activationToCostGradient float64) float64 {
	return weightToActivationCostRatio(weightToZGradient, zToActivationGradient, activationToCostGradient)
}

func getNeuronGradient(weights []float64, bias, activationToCostRatio, cost float64, inputs []float64) neuronGradient {
	biasToZRatio := biasToZRatio()
	zToActivationRatio := zToActivationRatio(gonet.WeightedSumAndBias(weights, bias, inputs))

	inputsRatios := make([]float64, len(inputs))
	for i := range inputsRatios {
		inputToZGradient := previousActivationToZRatio(weights[i])
		inputsRatios[i] = getInputGradient(inputToZGradient, zToActivationRatio, activationToCostRatio)
	}

	weigthsRatios := make([]float64, len(weights))
	weightToZGradients := weightsToZRatios(inputs)
	for i := range weigthsRatios {
		weigthsRatios[i] = getWeightRatio(weightToZGradients[i], zToActivationRatio, activationToCostRatio)
	}
	biasRatio := getBiasRatio(biasToZRatio, zToActivationRatio, activationToCostRatio)
	return neuronGradient{
		biasGradient:     biasRatio,
		inputsGradients:  inputsRatios,
		weightsGradients: weigthsRatios,
	}
}

func getNetworkGradient(network gonet.Network, input, expectedNetworkOutput []float64) networkGradient {
	layers := network.GetLayers()

	layerOutputs, _ := getLayerOutputs(network, input)
	layerInputs := append([][]float64{input}, layerOutputs...)[:len(layerOutputs)]

	weights := make([][][]float64, len(layers))
	biases := make([][]float64, len(layers))
	for l := range layers {
		neurons := layers[l].GetNeurons()
		weights[l] = make([][]float64, len(neurons))
		biases[l] = make([]float64, len(neurons))
		for n := range neurons {
			weights[l][n] = neurons[n].GetWeigths()
			biases[l][n] = neurons[n].GetBias()
		}
	}

	weigthsGradient, biasGradient := backprop.NetworkGradient(layerOutputs[len(layers)-1], expectedNetworkOutput, weights, biases, layerInputs)

	layerGradients := make([]layerGradient, len(layers))
	for l := range layers {
		neuronGradients := make([]neuronGradient, len(layers[l].GetNeurons()))
		for n := range neuronGradients {
			neuronGradients[n] = neuronGradient{
				biasGradient:     biasGradient[l][n],
				weightsGradients: weigthsGradient[l][n],
			}
		}
		layerGradients[l] = layerGradient{
			neuronGradients: neuronGradients,
		}
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

	summedNeuronGradients := make([]neuronGradient, len(neuronGradients[0]))
	copy(summedNeuronGradients, neuronGradients[0])
	for i := 1; i < len(neuronGradients); i++ {
		for g := range summedNeuronGradients {
			summedNeuronGradients[g] = addNeuronGradients(summedNeuronGradients[g], neuronGradients[i][g])
		}
	}

	return layerGradient{
		neuronGradients: summedNeuronGradients,
	}
}

func addNetworkGradients(networkGradients ...networkGradient) networkGradient {
	numberOfNetworkGradients := float64(len(networkGradients))
	layerGradients := make([][]layerGradient, len(networkGradients))

	for i := range layerGradients {
		layerGradients[i] = networkGradients[i].layerGradients
	}

	summedLayerGradients := make([]layerGradient, len(layerGradients[0]))
	copy(summedLayerGradients, layerGradients[0])
	for i := 1; i < len(layerGradients); i++ {
		for g := range summedLayerGradients {
			summedLayerGradients[g] = addLayerGradients(summedLayerGradients[g], layerGradients[i][g])
		}
	}

	for l := 0; l < len(summedLayerGradients); l++ {
		for n := range summedLayerGradients[l].neuronGradients {
			summedLayerGradients[l].neuronGradients[n].biasGradient = summedLayerGradients[l].neuronGradients[n].biasGradient / numberOfNetworkGradients
			for w := range summedLayerGradients[l].neuronGradients[n].weightsGradients {
				summedLayerGradients[l].neuronGradients[n].weightsGradients[w] = summedLayerGradients[l].neuronGradients[n].weightsGradients[w] / numberOfNetworkGradients
			}
		}
	}

	return networkGradient{
		layerGradients: summedLayerGradients,
	}
}
