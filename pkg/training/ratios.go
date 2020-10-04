package training

import gonet "github.com/Revanee/gonet/pkg"

func activationToCostRatio(activation, desired float64) float64 {
	return 2 * (activation - desired)
}

func activationsToCostsRatios(activations, desired []float64) []float64 {
	costs := make([]float64, len(activations))
	for i := range costs {
		costs[i] = activationToCostRatio(activations[i], desired[i])
	}
	return costs
}

func zToActivationRatio(z float64) float64 {
	return logisticCurveDerivative(z)
}

func zsToActivationsRatios(zs []float64) []float64 {
	ratios := make([]float64, len(zs))
	for i := range ratios {
		ratios[i] = zToActivationRatio(zs[i])
	}
	return ratios
}

func weightsToZRatios(previousActivations []float64) []float64 {
	ratios := make([]float64, len(previousActivations))
	for i := range ratios {
		ratios[i] = previousActivations[i]
	}
	return ratios
}

func weightsToZsRatios(previousActivations []float64, weigths [][]float64) [][]float64 {
	ratios := make([][]float64, len(weigths), len(previousActivations))
	for i := range ratios {
		ratios[i] = weightsToZRatios(previousActivations)
	}
	return ratios
}

func biasToZRatio() float64 {
	return 1
}

func biasesToZsRatios(biases []float64) []float64 {
	ratios := make([]float64, len(biases))
	for i := range ratios {
		ratios[i] = biasToZRatio()
	}
	return ratios
}

func previousActivationToZRatio(weight float64) float64 {
	return weight
}

func previousActivationsToZRatios(weights []float64) []float64 {
	return weights
}

func previousActivationsToZsRatios(previousActivations []float64, weights [][]float64) []float64 {
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

func previousActivationsToCostRatio(previousActivationToZRatio, zToActivationRatio, activationToCostRatio float64) float64 {
	return previousActivationToZRatio * zToActivationRatio * activationToCostRatio
}

func weightsToActivationCostRatios(activationToCostRatio, zToActivationRatio float64, weights, inputs []float64) []float64 {
	ratios := make([]float64, len(weights))
	weightsToZRatios := weightsToZRatios(inputs)
	for i := range ratios {
		ratios[i] = weightsToZRatios[i] *
			zToActivationRatio *
			activationToCostRatio
	}
	return ratios
}

func weightsToActivationCostsRatios(biases, desireds, activations, previousActivations, zs []float64, weights [][]float64) [][]float64 {
	ratios := make([][]float64, len(weights), len(previousActivations))
	for i := range ratios {
		activationToCostRatio := activationToCostRatio(activations[i], desireds[i])
		z := gonet.WeightedSumAndBias(weights[i], biases[i], previousActivations)
		zToActivationRatio := zToActivationRatio(z)
		ratios[i] = weightsToActivationCostRatios(activationToCostRatio, zToActivationRatio, weights[i], previousActivations)
	}
	return ratios
}

func biasToActivationCostRatio(biasToZRatio, zToActivationRatio, activationToCostRatio float64) float64 {
	return biasToZRatio * zToActivationRatio * activationToCostRatio
}

func biasesToActivationCostsRatios(biases, zs, activations, desireds []float64) []float64 {
	ratios := make([]float64, len(biases))
	for i := range ratios {
		ratios[i] = biasToActivationCostRatio(activations[i], desireds[i], zs[i])
	}
	return ratios
}

func logisticCurveDerivative(x float64) float64 {
	return gonet.LogisticCurve(-x, 1)
}
