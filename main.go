package main

import (
	"fmt"

	gonet "github.com/Revanee/gonet/pkg"
	"github.com/Revanee/gonet/pkg/training"
)

func main() {

	l1 := gonet.NewRandomizedLayer(2, 2)
	l2 := gonet.NewRandomizedLayer(2, 2)
	l3 := gonet.NewRandomizedLayer(2, 2)
	l4 := gonet.NewRandomizedLayer(2, 2)

	network := gonet.NewNetwork(l1, l2, l3, l4)

	trainingInputs := [][]float64{
		{1, 0},
		{0, 1},
		{1, 1},
		{0, 0},
	}
	trainingOutputs := [][]float64{
		{0, 1},
		{1, 0},
		{0, 0},
		{1, 1},
	}

	inputs := trainingInputs[0]
	expectedOutputs := trainingOutputs[0]

	newNetwork := trainOnData(network, trainingInputs, trainingOutputs)
	actualOutputs, _ := gonet.ActivateNetwork(newNetwork, inputs...)

	showResult(actualOutputs, expectedOutputs, inputs)
}

func trainOnData(network gonet.Network, inputs, outputs [][]float64) gonet.Network {
	newNetwork := network
	for i := 0; i < 10000; i++ {
		for j := range inputs {
			newNetwork = train(newNetwork, inputs[j], outputs[j])
		}
	}
	return newNetwork
}

func train(network gonet.Network, inputs, expectedOutputs []float64) gonet.Network {
	newNetwork := network
	n, err := training.TrainNetwork(newNetwork, inputs, expectedOutputs)
	newNetwork = n
	if err != nil {
		panic(err)
	}
	return newNetwork
}

func showResult(outputs, expectedOutputs, inputs []float64) {
	fmt.Printf("Inputs:\t\t%s\n", printTabbed(inputs...))
	fmt.Printf("Activations:\t%s\n", printTabbed(outputs...))
	fmt.Printf("Expected:\t%s\n", printTabbed(expectedOutputs...))
	// fmt.Printf("Difference:\t%f\n", outputs[0]-expectedOutputs[0])
}

func printTabbed(numbers ...float64) string {
	result := ""
	for _, number := range numbers {
		result += fmt.Sprintf("%f\t", number)
	}
	return result
}
