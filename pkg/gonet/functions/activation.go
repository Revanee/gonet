package functions

import "math"

func LogisticCurveDerivative(x float64) float64 {
	return LogisticCurve(-x, 1)
}

func LogisticCurve(x, k float64) float64 {
	return 1 / (1 + math.Pow(math.E, (-k*x)))
}

func ReLU(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func ReLUDerivative(x float64) float64 {
	if x < 0 {
		return 0
	}
	return 1
}

func WeightedSumAndBias(input, weights []float64, bias float64) float64 {
	z := bias
	for w, weight := range weights {
		z += weight * input[w]
	}
	return z
}
