package network

import (
	"errors"
	"math/rand"

	"github.com/Revanee/gonet/pkg/gonet/functions"
)

type neuronData struct {
	weigths []float64
	bias    float64
}

var _ Neuron = (*neuronData)(nil)

type Neuron interface {
	GetWeigths() []float64
	GetBias() float64
	GetTensor() ([]float64, float64)
	Activate(...float64) (float64, error)
}

func (n neuronData) GetWeigths() []float64 {
	return n.weigths
}

func (n neuronData) GetBias() float64 {
	return n.bias
}

func (n neuronData) GetTensor() ([]float64, float64) {
	return n.weigths, n.bias
}

func (n *neuronData) Activate(values ...float64) (float64, error) {
	if len(values) != len(n.weigths) {
		return 0, errors.New("Mismatching number of inputs and weights")
	}
	return functions.LogisticCurve(WeightedSumAndBias(n.weigths, n.bias, values), 1), nil
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
	return &neuronData{
		bias:    bias,
		weigths: weigths,
	}
}

func NewRandomizedNeuron(inputSize int) Neuron {
	weigths := make([]float64, inputSize)
	for i := range weigths {
		weigths[i] = rand.Float64()*2 - 1
	}
	return NewNeuron(rand.Float64()*2-1, weigths...)
}
