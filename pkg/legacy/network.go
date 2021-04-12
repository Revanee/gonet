package pkg

import "github.com/pkg/errors"

type Network interface {
	GetLayers() []Layer
}

type NetworkData struct {
	layers []Layer
}

var _ Network = (*NetworkData)(nil)

func (n NetworkData) GetLayers() []Layer {
	return n.layers
}

func ActivateNetwork(network Network, inputs ...float64) ([]float64, error) {
	previousActivations := inputs
	for _, l := range network.GetLayers() {
		activations, err := ActivateLayer(l, previousActivations...)
		if err != nil {
			return nil, errors.Wrap(err, "A layer failed to compute")
		}
		previousActivations = activations
	}
	return previousActivations, nil
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

func NewNetwork(layers ...Layer) Network {
	return &NetworkData{
		layers: layers,
	}
}
