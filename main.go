package main

import (
	"fmt"
	"math/rand"
	"os"
	"runtime/pprof"

	"github.com/Revanee/gonet/pkg/gonet/network"
	"github.com/Revanee/gonet/pkg/gonet/training"
)

func main() {
	f, err := os.Create("p.pprof")
	if err != nil {
		fmt.Println(err)
	}
	pprof.StartCPUProfile(f)
	defer pprof.StopCPUProfile()

	network := network.NewNetwork(
		// network.NewRandomizedLayer(2, 2),
		// network.NewRandomizedLayer(2, 2),
		// network.NewRandomizedLayer(2, 2),
		// network.NewRandomizedLayer(2, 2),
		// network.NewRandomizedLayer(2, 2),
		network.NewRandomizedLayer(2, 6),
		// network.NewRandomizedLayer(10, 10),
		network.NewRandomizedLayer(6, 6),
		network.NewRandomizedLayer(6, 6),
		network.NewRandomizedLayer(6, 2),
	)

	trainingInputs := make([][]float64, 1000)
	for i := range trainingInputs {
		trainingInputs[i] = []float64{rand.Float64(), rand.Float64()}
	}
	trainingOutputs := generateTrainingOutputs(trainingInputs)

	newNetwork := trainOnData(network, trainingInputs, trainingOutputs)

	testInputs := []float64{0, 1}
	networkActivations, _ := newNetwork.Activate(testInputs...)
	showResult(networkActivations, generateTrainingOutput(testInputs), testInputs)
	fmt.Println("---")
	testInputs = []float64{1, 0}
	networkActivations, _ = newNetwork.Activate(testInputs...)
	showResult(networkActivations, generateTrainingOutput(testInputs), testInputs)
	fmt.Println("---")
	testInputs = []float64{.5, .5}
	networkActivations, _ = newNetwork.Activate(testInputs...)
	showResult(networkActivations, generateTrainingOutput(testInputs), testInputs)
}

func trainOnData(network network.Network, inputs, outputs [][]float64) network.Network {
	newNetwork := network
	for i := 0; i < 1000; i++ {
		if i%10 == 0 {
			fmt.Println("-------")
			fmt.Println(i/10, "%")
			testInputs := []float64{0, 1}
			networkActivations, _ := newNetwork.Activate(testInputs...)
			showResult(networkActivations, generateTrainingOutput(testInputs), testInputs)
			fmt.Println("---")
			testInputs = []float64{1, 0}
			networkActivations, _ = newNetwork.Activate(testInputs...)
			showResult(networkActivations, generateTrainingOutput(testInputs), testInputs)
			fmt.Println("---")
			testInputs = []float64{.5, .5}
			networkActivations, _ = newNetwork.Activate(testInputs...)
			showResult(networkActivations, generateTrainingOutput(testInputs), testInputs)
		}
		newNetwork = training.BackPropNetworkBatch(newNetwork, inputs, outputs)
	}
	return newNetwork
}

// func train(network network.Network, inputs, expectedOutputs []float64) network.Network {
// 	newNetwork := training.TrainNetworkSingle(network, inputs, expectedOutputs)
// 	return newNetwork
// }

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

func generateTrainingOutputs(trainingInputs [][]float64) [][]float64 {
	trainingOutputs := make([][]float64, len(trainingInputs))
	for i := range trainingOutputs {
		trainingOutputs[i] = generateTrainingOutput(trainingInputs[i])
	}
	return trainingOutputs
}

func generateTrainingOutput(inputs []float64) []float64 {
	output := []float64{inputs[1], inputs[0]}
	return output
}
