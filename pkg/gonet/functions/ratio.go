package functions

func WeightToActivationCostRatio(weightToZRatio, zToActivationRatio, activationToCostRatio float64) float64 {
	return weightToZRatio * zToActivationRatio * activationToCostRatio
}

func BiasToActivationCostRatio(biasToZRatio, zToActivationRatio, activationToCostRatio float64) float64 {
	return biasToZRatio * zToActivationRatio * activationToCostRatio
}

func InputToCostRatio(inputToZRatio, zToActivationRatio, activationToCostRatio float64) float64 {
	return inputToZRatio * zToActivationRatio * activationToCostRatio
}

func ActivationToCostRatio(activation, desired float64) float64 {
	return 2 * (activation - desired)
}

func ZToActivationRatio(z float64) float64 {
	return LogisticCurveDerivative(z)
}

func WeightsToZRatios(previousActivations []float64) []float64 {
	ratios := make([]float64, len(previousActivations))
	copy(ratios, previousActivations)
	return ratios
}

func BiasToZRatio() float64 {
	return 1
}

func InputToZRatio(weight float64) float64 {
	return weight
}

func InputsToZsRatios(weights [][]float64) []float64 {
	ratios := make([]float64, len(weights[0]))
	for w := range ratios {
		ratio := 0.0
		for n := range weights {
			ratio += weights[n][w]
		}
		ratios[w] = ratio
	}
	return ratios
}

func WeightsToActivationCostRatios(activationToCostRatio, zToActivationRatio float64, weights, inputs []float64) []float64 {
	ratios := make([]float64, len(weights))
	weightsToZRatios := WeightsToZRatios(inputs)
	for w := range ratios {
		ratios[w] = weightsToZRatios[w] *
			zToActivationRatio *
			activationToCostRatio
	}
	return ratios
}
