package training

import (
	gonet "github.com/Revanee/gonet/pkg"
	"github.com/pkg/errors"
)

func TrainLayer(layer gonet.Layer, inputs, actualOutputs, expectedOutputs []float64) gonet.Layer {
	return StepLayer(layer, actualOutputs, expectedOutputs, inputs)
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

func getDesiredNeuronInputs(neuron gonet.Neuron, inputs []float64, actualOutput, expetedOutput float64) []float64 {
	weights := neuron.GetWeigths()
	activationCost := activationCost(actualOutput, expetedOutput)
	z := gonet.WeightedSumAndBias(neuron.GetWeigths(), neuron.GetBias(), inputs)
	zToActivationRatio := zToActivationRatio(z)
	activationToCostRatio := activationToCostRatio(actualOutput, expetedOutput)

	desiredInputs := make([]float64, len(inputs))
	for i := range desiredInputs {
		previousActivationToZRatio := previousActivationToZRatio(weights[i])
		previousActivationsToCostRatio := previousActivationsToCostRatio(previousActivationToZRatio, zToActivationRatio, activationToCostRatio)
		desiredInputs[i] = previousActivationsToCostRatio * activationCost
	}
	return desiredInputs
}

func getDesiredLayerInputs(layer gonet.Layer, inputs, actualOutputs, expectedOutputs []float64) []float64 {
	neurons := layer.GetNeurons()
	desiredNeuronsInputs := make([][]float64, len(neurons))
	for i := range neurons {
		desiredNeuronsInputs[i] = getDesiredNeuronInputs(neurons[i], inputs, actualOutputs[i], expectedOutputs[i])
	}
	desiredInputs := make([]float64, len(inputs))
	for i := range desiredInputs {
		desiredInputSum := 0.0
		for j := range desiredNeuronsInputs {
			desiredInputSum += desiredNeuronsInputs[j][i]
		}
		desiredInputs[i] = desiredInputSum / float64(len(desiredNeuronsInputs))
	}
	return desiredInputs
}

func getDesiredLayersOutputs(layers []gonet.Layer, layeredInputs [][]float64, expectedOutputs []float64) [][]float64 {
	desiredLayersInputs := make([][]float64, len(layers))
	previousExpectedOutputs := expectedOutputs
	for i := len(layers) - 1; i >= 0; i-- {
		desiredLayersInputs[i] = getDesiredLayerInputs(layers[i], layeredInputs[i], layeredInputs[i+1], previousExpectedOutputs)
		previousExpectedOutputs = desiredLayersInputs[i]
	}
	return append(desiredLayersInputs, expectedOutputs)
}

func TrainNetwork(network gonet.Network, inputs, expectedOutputs []float64) (gonet.Network, error) {
	layers := network.GetLayers()
	layerOutputs, err := getLayerOutputs(network, inputs)
	if err != nil {
		return nil, errors.Wrap(err, "Could not get layer outputs")
	}
	layeredData := append([][]float64{inputs}, layerOutputs...)
	desiredLayersOutputs := getDesiredLayersOutputs(layers, layeredData, expectedOutputs)

	newLayers := make([]gonet.Layer, len(layers))
	for i := range newLayers {
		newLayers[i] = TrainLayer(layers[i], layeredData[i], layeredData[i+1], desiredLayersOutputs[i+1])
	}
	return gonet.NewNetwork(newLayers...), nil
}
