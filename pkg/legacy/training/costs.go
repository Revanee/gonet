package training

import (
	"math"
)

func activationCost(activation, desired float64) float64 {
	return math.Pow(activation-desired, 2)
}

func activationsCosts(activations, desired []float64) []float64 {
	costs := make([]float64, len(activations))
	for i := range activations {
		costs[i] = activationCost(activations[i], desired[i])
	}
	return costs
}
