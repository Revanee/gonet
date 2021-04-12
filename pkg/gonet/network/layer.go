package network

import (
	"github.com/pkg/errors"
)

type Layer interface {
	GetNeurons() []Neuron
	GetTensor() ([][]float64, []float64)
	Activate(...float64) ([]float64, error)
}

type layerData struct {
	neurons []Neuron
}

var _ Layer = (*layerData)(nil)

func (l layerData) GetNeurons() []Neuron {
	return l.neurons
}

func (l layerData) GetTensor() (weights [][]float64, biases []float64) {
	weights = make([][]float64, len(l.neurons))
	biases = make([]float64, len(l.neurons))
	for n := range l.neurons {
		weights[n], biases[n] = l.neurons[n].GetTensor()
	}
	return weights, biases
}

func (l *layerData) Activate(values ...float64) ([]float64, error) {
	activations := make([]float64, len(l.GetNeurons()))
	for i, n := range l.GetNeurons() {
		var err error = nil
		activations[i], err = n.Activate(values...)
		if err != nil {
			return nil, errors.Wrap(err, "One or more of the neurons in layer got mismatchign inputs and weigths")
		}
	}
	return activations, nil
}

func (l *layerData) ActivateParallel(values ...float64) ([]float64, error) {
	activations := make([]float64, len(l.GetNeurons()))
	ch := make(chan []float64)
	for i, n := range l.GetNeurons() {
		var err error = nil
		go activateNeuron(float64(i), values, n, ch)
		if err != nil {
			return nil, errors.Wrap(err, "One or more of the neurons in layer got mismatchign inputs and weigths")
		}
	}
	for range l.GetNeurons() {
		res := <-ch
		activations[int(res[0])] = res[1]
	}
	close(ch)
	return activations, nil
}

func activateNeuron(index float64, input []float64, neuron Neuron, ch chan []float64) {
	res, _ := neuron.Activate(input...)
	ch <- []float64{index, res}
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

func NewLayerFromTensor(weights [][]float64, biases []float64) Layer {
	neurons := make([]Neuron, len(weights))
	for n := range neurons {
		neurons[n] = NewNeuron(biases[n], weights[n]...)
	}
	return NewLayer(neurons...)
}

func NewLayer(neurons ...Neuron) Layer {
	return &layerData{
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
