package functions

import "math"

func LogisticCurveDerivative(x float64) float64 {
	return LogisticCurve(-x, 1)
}

func LogisticCurve(x, k float64) float64 {
	return 1 / (1 + math.Pow(math.E, (-k*x)))
}
