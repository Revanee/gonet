package pkg

import (
	"github.com/pkg/errors"
)

type Layer interface {
	GetNeurons() []Neuron
}

type LayerData struct {
	neurons []Neuron
}

var _ Layer = (*LayerData)(nil)

func (l LayerData) GetNeurons() []Neuron {
	return l.neurons
}

func ActivateLayer(layer Layer, values ...float64) ([]float64, error) {
	activations := make([]float64, len(layer.GetNeurons()))
	for i, n := range layer.GetNeurons() {
		var err error = nil
		activations[i], err = ActivateNeuron(n, values...)
		if err != nil {
			return nil, errors.Wrap(err, "One or more of the neurons in layer got mismatchign inputs and weigths")
		}
	}
	return activations, nil
}

func LayerWithNewWeigths(layer Layer, weigths ...[]float64) Layer {
	neurons := layer.GetNeurons()
	for i, neuron := range neurons {
		neuronWeights := weigths[i]
		NewNeuron(neuron.GetBias(), neuronWeights...)
	}
	return NewLayer(neurons...)
}

func LayerWithNewBiases(layer Layer, biases []float64) Layer {
	neurons := layer.GetNeurons()
	for i, neuron := range neurons {
		neurons[i] = NewNeuron(biases[i], neuron.GetWeigths()...)
	}
	return NewLayer(neurons...)
}

func NewLayer(neurons ...Neuron) Layer {
	return &LayerData{
		neurons: neurons,
	}
}

func NewRandomizedLayer(inputSize int, layerSize int) Layer {
	neurons := make([]Neuron, layerSize)
	for i := range neurons {
		neurons[i] = NewRandomizedNeuron(inputSize)
	}
	return NewLayer(neurons...)
}
