package pkg

import "math"

func LogisticCurve(x, k float64) float64 {
	return 1 / (1 + math.Pow(math.E, (-k*x)))
}
