package functions

import "math"

func ActivationCost(activation, desired float64) float64 {
	return math.Pow(activation-desired, 2)
}

func ActivationsCostsAverage(activations, desired []float64) float64 {
	costTotal := .0
	for i := range activations {
		costTotal += ActivationCost(activations[i], desired[i])
	}
	return costTotal / float64(len(activations))
}
