package functions

import "math"

func ActivationCost(activation, desired float64) float64 {
	return math.Pow(activation-desired, 2)
}

func ActivationsCostsSum(activations, desired []float64) float64 {
	cost := .0
	for i := range activations {
		cost += ActivationCost(activations[i], desired[i])
	}
	return cost
}
