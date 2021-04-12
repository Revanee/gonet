package network

import "github.com/pkg/errors"

type Network interface {
	GetLayers() []Layer
	GetTensor() ([][][]float64, [][]float64)
	Activate(...float64) ([]float64, error)
	ActivateTransparent(...float64) ([][]float64, error)
}

type networkData struct {
	layers []Layer
}

var _ Network = (*networkData)(nil)

func (n networkData) GetLayers() []Layer {
	return n.layers
}

func (n networkData) GetTensor() (weights [][][]float64, biases [][]float64) {
	weights = make([][][]float64, len(n.layers))
	biases = make([][]float64, len(n.layers))
	for l := range n.layers {
		weights[l], biases[l] = n.layers[l].GetTensor()
	}
	return weights, biases
}

func (n *networkData) Activate(inputs ...float64) ([]float64, error) {
	previousActivations := inputs
	for _, l := range n.GetLayers() {
		activations, err := l.Activate(previousActivations...)
		if err != nil {
			return nil, errors.Wrap(err, "A layer failed to compute")
		}
		previousActivations = activations
	}
	return previousActivations, nil
}

func (n *networkData) ActivateTransparent(inputs ...float64) (allActivations [][]float64, error error) {
	allActivations = make([][]float64, len(n.layers))
	previousActivations := inputs
	for l := range n.layers {
		activations, err := n.layers[l].Activate(previousActivations...)
		if err != nil {
			return nil, errors.Wrap(err, "A layer failed to compute")
		}
		allActivations[l] = activations
		previousActivations = activations
	}
	return allActivations, nil
}

func NetworkWithNewWeigths(network Network, weigths ...[][]float64) Network {
	layers := make([]Layer, len(network.GetLayers()))
	for i, layer := range layers {
		layerWeights := weigths[i]
		layers[i] = LayerWithNewWeigths(layer, layerWeights...)
	}
	return NewNetwork(layers...)
}

func NetworkWithNewBiases(network Network, biases ...[]float64) Network {
	layers := make([]Layer, len(network.GetLayers()))
	for i, layer := range layers {
		layers[i] = LayerWithNewBiases(layer, biases[i])
	}
	return NewNetwork(layers...)
}

func NewNetworkFromTensor(weights [][][]float64, biases [][]float64) Network {
	layers := make([]Layer, len(weights))
	for l := range layers {
		layers[l] = NewLayerFromTensor(weights[l], biases[l])
	}
	return NewNetwork(layers...)
}

func NewNetwork(layers ...Layer) Network {
	return &networkData{
		layers: layers,
	}
}
