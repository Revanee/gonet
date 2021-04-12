package training

import gonet "github.com/Revanee/gonet/pkg/legacy"

func stepNeuron(neuron gonet.Neuron, neuronGradient neuronGradient) gonet.Neuron {
	stepSize := .3
	weights := neuron.GetWeigths()
	newWeights := make([]float64, len(weights))
	for i := range newWeights {
		newWeights[i] = weights[i] - neuronGradient.weightsGradients[i]*stepSize
	}
	newBias := neuron.GetBias() - neuronGradient.biasGradient*stepSize
	return gonet.NewNeuron(newBias, newWeights...)
}

func stepLayer(layer gonet.Layer, layerGradient layerGradient) gonet.Layer {
	neurons := layer.GetNeurons()
	newNeurons := make([]gonet.Neuron, len(neurons))
	for i, neuron := range neurons {
		newNeurons[i] = stepNeuron(neuron, layerGradient.neuronGradients[i])
	}
	return gonet.NewLayer(newNeurons...)
}

func stepNetwork(network gonet.Network, networkGradient networkGradient) gonet.Network {
	layers := network.GetLayers()
	newLayers := make([]gonet.Layer, len(layers))
	for i, layer := range layers {
		newLayers[i] = stepLayer(layer, networkGradient.layerGradients[i])
	}
	return gonet.NewNetwork(newLayers...)
}
