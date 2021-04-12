package backprop

import gonet "github.com/Revanee/gonet/pkg/legacy"

func ActivationToCostRatio(activation, desired float64) float64 {
	return 2 * (activation - desired)
}

func ActivationsToCostsRatios(activations, desired []float64) []float64 {
	costs := make([]float64, len(activations))
	for i := range costs {
		costs[i] = ActivationToCostRatio(activations[i], desired[i])
	}
	return costs
}

func ZToActivationRatio(z float64) float64 {
	return LogisticCurveDerivative(z)
}

func ZsToActivationsRatios(zs []float64) []float64 {
	ratios := make([]float64, len(zs))
	for i := range ratios {
		ratios[i] = ZToActivationRatio(zs[i])
	}
	return ratios
}

func WeightsToZRatios(previousActivations []float64) []float64 {
	ratios := make([]float64, len(previousActivations))
	for i := range ratios {
		ratios[i] = previousActivations[i]
	}
	return ratios
}

func WeightsToZsRatios(previousActivations []float64, weigths [][]float64) [][]float64 {
	ratios := make([][]float64, len(weigths))
	for i := range ratios {
		ratios[i] = WeightsToZRatios(previousActivations)
	}
	return ratios
}

func BiasToZRatio() float64 {
	return 1
}

func BiasesToZsRatios(biases []float64) []float64 {
	ratios := make([]float64, len(biases))
	for i := range ratios {
		ratios[i] = BiasToZRatio()
	}
	return ratios
}

func PreviousActivationToZRatio(weight float64) float64 {
	return weight
}

func PreviousActivationsToZRatios(weights []float64) []float64 {
	return weights
}

func PreviousActivationsToZsRatios(previousActivations []float64, weights [][]float64) []float64 {
	ratios := make([]float64, len(previousActivations))
	for i := range ratios {
		ratio := 0.0
		for j := range weights[i] {
			ratio += weights[i][j]
		}
		ratios[i] = ratio
	}
	return ratios
}

func PreviousActivationToCostRatio(previousActivationToZRatio, zToActivationRatio, activationToCostRatio float64) float64 {
	return previousActivationToZRatio * zToActivationRatio * activationToCostRatio
}

func WeightToActivationCostRatio(weightToZRatio, zToActivationRatio, activationToCostRatio float64) float64 {
	return weightToZRatio * zToActivationRatio * activationToCostRatio
}

func WeightsToActivationCostRatios(activationToCostRatio, zToActivationRatio float64, weights, inputs []float64) []float64 {
	ratios := make([]float64, len(weights))
	weightsToZRatios := WeightsToZRatios(inputs)
	for i := range ratios {
		ratios[i] = weightsToZRatios[i] *
			zToActivationRatio *
			activationToCostRatio
	}
	return ratios
}

func WeightsToActivationCostsRatios(biases, desireds, activations, previousActivations, zs []float64, weights [][]float64) [][]float64 {
	ratios := make([][]float64, len(weights), len(previousActivations))
	for i := range ratios {
		activationToCostRatio := ActivationToCostRatio(activations[i], desireds[i])
		z := gonet.WeightedSumAndBias(weights[i], biases[i], previousActivations)
		zToActivationRatio := ZToActivationRatio(z)
		ratios[i] = WeightsToActivationCostRatios(activationToCostRatio, zToActivationRatio, weights[i], previousActivations)
	}
	return ratios
}

func BiasToActivationCostRatio(biasToZRatio, zToActivationRatio, activationToCostRatio float64) float64 {
	return biasToZRatio * zToActivationRatio * activationToCostRatio
}

func BiasesToActivationCostsRatios(biases, zs, activations, desireds []float64) []float64 {
	ratios := make([]float64, len(biases))
	for i := range ratios {
		ratios[i] = BiasToActivationCostRatio(activations[i], desireds[i], zs[i])
	}
	return ratios
}

func LogisticCurveDerivative(x float64) float64 {
	return gonet.LogisticCurve(-x, 1)
}
