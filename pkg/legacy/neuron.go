package pkg

import (
	"errors"
	"math/rand"
)

type NeuronData struct {
	weigths []float64
	bias    float64
}

var _ Neuron = (*NeuronData)(nil)

type Neuron interface {
	GetWeigths() []float64
	GetBias() float64
}

func (n NeuronData) GetWeigths() []float64 {
	return n.weigths
}

func (n NeuronData) GetBias() float64 {
	return n.bias
}

func ActivateNeuron(neuron Neuron, values ...float64) (float64, error) {
	if len(values) != len(neuron.GetWeigths()) {
		return 0, errors.New("Mismatching number of inputs and weights")
	}
	return LogisticCurve(WeightedSumAndBias(neuron.GetWeigths(), neuron.GetBias(), values), 1), nil
}

func WeightedSumAndBias(weigths []float64, bias float64, values []float64) float64 {
	return WeightedSum(weigths, values) + bias
}

func WeightedSum(weigths, values []float64) float64 {
	weightedSum := 0.0
	for i, v := range values {
		weightedSum +=
			v * weigths[i]
	}
	return weightedSum
}

func NewNeuron(bias float64, weigths ...float64) Neuron {
	return &NeuronData{
		bias:    bias,
		weigths: weigths,
	}
}

func NewRandomizedNeuron(inputSize int) Neuron {
	weigths := make([]float64, inputSize)
	for i := range weigths {
		weigths[i] = rand.Float64()
	}
	return NewNeuron(rand.Float64(), weigths...)
}
