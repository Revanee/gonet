package backprop_test

import (
	"testing"

	"github.com/Revanee/gonet/pkg/backprop"
)

func TestBiasGradient(t *testing.T) {
	if backprop.BiasRatio(0, 1)-.5 > 0 {
		t.Errorf("Bias gradient should be negative")
	}
	if backprop.BiasRatio(1, 0)-.5 < 0 {
		t.Errorf("Bias gradient should be postitive")
	}
}

func TestWeightsGradient(t *testing.T) {
	weigthsGradient := backprop.WeightsRatio(0, 1, []float64{0}, []float64{1})
	if weigthsGradient[0]-.5 > 0 {
		t.Errorf("Weigths gradient should be negative")
	}

	weigthsGradient = backprop.WeightsRatio(1, 0, []float64{1}, []float64{1})
	if weigthsGradient[0]-.5 < 0 {
		t.Errorf("Weigths gradient should be positive")
	}
}

func TestInputsGradient(t *testing.T) {
	inputsGradient := backprop.InputsRatio(0, 1, []float64{0}, []float64{1})
	if inputsGradient[0]-.5 > 0 {
		t.Errorf("Inputs gradient should be negative")
	}

	inputsGradient = backprop.InputsRatio(1, 0, []float64{1}, []float64{1})
	if inputsGradient[0]-.5 < 0 {
		t.Errorf("Inputs gradient should be positive")
	}
}
