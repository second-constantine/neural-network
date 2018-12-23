package io.secondconstantine.neural.network

import java.math.BigDecimal
import java.util.*

class NeuralNetwork(
        val hiddenLayer: NeuronLayer = NeuronLayer(),
        val outputLayer: NeuronLayer = NeuronLayer()
) {

    fun prediction(inputs: MutableList<Double>): String {
        return String.format("%.2f", feedForward(inputs)[0] * 100) + "%"
    }

    fun feedForward(inputs: MutableList<Double>): List<Double> {
        return outputLayer.feedForward(hiddenLayer.feedForward(inputs))
    }

    fun train(trainingInputs: MutableList<Double>, trainingOutputs: MutableList<Double>) {
        feedForward(trainingInputs)
        val errorInputForOutputLayer = ArrayList<Double>()
        for (i in 0 until outputLayer.neurons.size) {
            errorInputForOutputLayer.add(outputLayer.neurons[i]
                    .calculateErrorInput(trainingOutputs[i]))
        }
        val errorInputForHiddenLayer = ArrayList<Double>()
        for (i in 0 until hiddenLayer.neurons.size) {
            var error = 0.0
            for (j in 0 until outputLayer.neurons.size) {
                error += errorInputForOutputLayer[j] * outputLayer.neurons[j].weights[i]
            }
            errorInputForHiddenLayer.add(error * hiddenLayer.neurons[i].calculate())
        }
        for (i in 0 until outputLayer.neurons.size) {
            for (w in 0 until outputLayer.neurons[i].weights.size) {
                var weight = outputLayer.neurons[i].weights[w]
                weight -= LEARNING_RATE * (errorInputForOutputLayer[i] * outputLayer.neurons[i].inputs[w])
                outputLayer.neurons[i].weights[w] = weight
            }
        }
        for (i in 0 until hiddenLayer.neurons.size) {
            for (w in 0 until hiddenLayer.neurons[i].weights.size) {
                var weight = hiddenLayer.neurons[i].weights[w]
                weight -= w * (errorInputForHiddenLayer[i] * hiddenLayer.neurons[i].inputs[w])
                hiddenLayer.neurons[i].weights[w] = weight
            }
        }
    }

    fun calculateError(trainingSets: Array<Array<Array<Double>>>): BigDecimal {
        var totalError = 0.0
        for (a in trainingSets.indices) {
            val trainingInputs = trainingSets[a][0]
            val trainingOutputs = trainingSets[a][1]
            feedForward(Arrays.asList(*trainingInputs))
            for (i in trainingOutputs.indices) {
                totalError += outputLayer.neurons[i].calculateError(trainingOutputs[i])
            }
        }
        return BigDecimal.valueOf(totalError)
    }

    companion object {

        val LEARNING_RATE = 0.5
        private val random = Random()

        @JvmStatic
        fun create(numInputs: Int, numHidden: Int, numOutputs: Int): NeuralNetwork {
            val hiddenLayer = NeuronLayer.create(numHidden, random.nextDouble())
            val outputLayer = NeuronLayer.create(numOutputs, random.nextDouble())
            for (neuron in outputLayer.neurons) {
                for (i in 0 until hiddenLayer.neurons.size) {
                    neuron.weights.add(random.nextDouble())
                }
            }
            for (neuron in hiddenLayer.neurons) {
                for (i in 0 until numInputs) {
                    neuron.weights.add(random.nextDouble())
                }
            }
            return NeuralNetwork(
                    hiddenLayer = hiddenLayer,
                    outputLayer = outputLayer
            )
        }
    }
}
