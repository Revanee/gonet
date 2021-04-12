package backprop

import (
	"math"
)

func ActivationCost(activation, desired float64) float64 {
	return math.Pow(activation-desired, 2)
}

func ActivationsCosts(activations, desired []float64) []float64 {
	costs := make([]float64, len(activations))
	for i := range activations {
		costs[i] = ActivationCost(activations[i], desired[i])
	}
	return costs
}
