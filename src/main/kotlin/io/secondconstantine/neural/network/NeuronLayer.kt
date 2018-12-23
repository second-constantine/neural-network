package io.secondconstantine.neural.network

import java.util.*

class NeuronLayer(
        val neurons: List<Neuron> = ArrayList(),
        val bias: Double = 0.0
) {

    fun feedForward(inputs: MutableList<Double>): MutableList<Double> {
        val ouputs = mutableListOf<Double>()
        for (neuron in neurons) {
            ouputs.add(neuron.calculateOutput(inputs))
        }
        return ouputs
    }

    companion object {
        fun create(neuronAmount: Int, bias: Double): NeuronLayer {
            val neurons = arrayListOf<Neuron>()
            for (i in 0 until neuronAmount) {
                neurons.add(Neuron())
            }
            return NeuronLayer(
                    bias = bias,
                    neurons = neurons
            )
        }
    }
}
