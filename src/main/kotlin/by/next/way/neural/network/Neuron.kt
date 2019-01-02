package by.next.way.neural.network

class Neuron(
        var output: Double = 0.0,
        var inputs: MutableList<Double> = mutableListOf(),
        var weights: MutableList<Double> = mutableListOf()
) {

    fun calculateInput(): Double {
        var total = 0.0
        for (i in inputs.indices) {
            total += inputs[i] * weights[i]
        }
        return total
    }

    fun calculateError(targetOutput: Double): Double {
        return 0.5 * Math.pow(targetOutput - output, 2.0)
    }

    fun calculateOutput(inputs: MutableList<Double>): Double {
        this.inputs = inputs
        this.output = squash(calculateInput())
        return output
    }

    fun calculateErrorInput(targetOutput: Double): Double {
        return diffOutput(targetOutput) * calculate()
    }

    fun calculate(): Double {
        return output * (1 - output)
    }

    private fun diffOutput(targetOutput: Double): Double {
        return -(targetOutput - output)
    }

    companion object {

        fun squash(totalNetInput: Double): Double {
            return 1 / (1 + Math.exp(-totalNetInput))
        }
    }
}
