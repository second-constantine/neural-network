package by.next.way.neural.network;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

class CalculatorMultiplyTest {

    private static Logger log = LogManager.getLogger();

    private Random random;
    private Double[][][] trainingSets;


    @BeforeEach
    void setUp() {
        random = new Random();
        trainingSets = new Double[][][]{
                {{1.0, 0.0}, {1.0}},
                {{0.0, 1.0}, {1.0}}
        };
    }

    @Test
    void example() {
        log.info("NN training start!");
        NeuralNetwork neuralNetwork = NeuralNetwork.create(trainingSets[0][0].length, 20, trainingSets[0][1].length);
        for (int i = 0; i < 100000; i++) {
            double first = random.nextDouble();
            double second = random.nextDouble();
            double sum = first * second;
            neuralNetwork.train(Arrays.asList(first, second), Arrays.asList(sum));
        }
        log.info("NN training finish!");
        print(neuralNetwork);
        log.warn("error: " + neuralNetwork.calculateError(trainingSets));
    }

    static void print(NeuralNetwork neuralNetwork) {
        log.info(
                "0.32, 0.11 -> " + neuralNetwork.feedForward(Arrays.asList(0.32, 0.11)) + "\n" +
                        "0.22, 0.11 -> " + neuralNetwork.feedForward(Arrays.asList(0.22, 0.11)) + "\n" +
                        "0.2, 0.1 -> " + neuralNetwork.feedForward(Arrays.asList(0.2, 0.1)) + "\n" +
                        "0.42, 0.11 -> " + neuralNetwork.feedForward(Arrays.asList(0.42, 0.11)) + "\n" +
                        "0.1, 0.1 -> " + neuralNetwork.feedForward(Arrays.asList(0.1, 0.1)) + "\n" +
                        "0.4, 0.13 -> " + neuralNetwork.feedForward(Arrays.asList(0.4, 0.13)) + "\n" +
                        "0.7, 0.11 -> " + neuralNetwork.feedForward(Arrays.asList(0.7, 0.11))
        );
    }
}
