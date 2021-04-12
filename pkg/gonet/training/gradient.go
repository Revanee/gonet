package training

func InputGradient(inputToCostRatio, cost float64) float64 {
	return inputToCostRatio * cost
}

func BiasGradient(biasToCostRatio, cost float64) float64 {
	return biasToCostRatio * cost
}

func WeightGradient(weightToCostRatio, cost float64) float64 {
	return weightToCostRatio * cost
}

func NeuronGradient(weightsToCostRatio []float64, biasToCostRatio, cost float64) (weightsGradient []float64, biasGradient float64) {
	weightsGradient = make([]float64, len(weightsToCostRatio))
	for w := range weightsGradient {
		weightsGradient[w] = weightsToCostRatio[w] * cost
	}
	biasGradient = biasToCostRatio * cost
	return weightsGradient, biasGradient
}

func LayerGradient(weightsToCostRatio [][]float64, biasesToCostRatio []float64, cost float64) (weightsGradient [][]float64, biasesGradient []float64) {
	weightsGradient = make([][]float64, len(weightsToCostRatio))
	biasesGradient = make([]float64, len(biasesToCostRatio))
	for n := range weightsToCostRatio {
		weightsGradient[n], biasesGradient[n] = NeuronGradient(weightsToCostRatio[n], biasesToCostRatio[n], cost)
	}
	return weightsGradient, biasesGradient
}

func NetworkGradient(weightsToCostRatio [][][]float64, biasesToCostRatio [][]float64, cost float64) (weightsGradient [][][]float64, biasesGradient [][]float64) {
	weightsGradient = make([][][]float64, len(weightsToCostRatio))
	biasesGradient = make([][]float64, len(biasesToCostRatio))
	for l := range weightsToCostRatio {
		weightsGradient[l], biasesGradient[l] = LayerGradient(weightsToCostRatio[l], biasesToCostRatio[l], cost)
	}
	return weightsGradient, biasesGradient
}
