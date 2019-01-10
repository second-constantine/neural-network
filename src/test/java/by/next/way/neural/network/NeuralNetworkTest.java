package by.next.way.neural.network;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

class NeuralNetworkTest {

    private static Logger log = LogManager.getLogger();

    private Random random;
    private Double[][][] trainingSets;

    @BeforeEach
    void setUp() {
        random = new Random();
        trainingSets = new Double[][][]{
                {{0.0, 0.0}, {0.0}},
                {{0.0, 1.0}, {1.0}},
                {{1.0, 0.0}, {1.0}},
                {{1.0, 1.0}, {0.0}}
        };
    }

    @Test
    void example() {
        log.info("NN training start!");
        NeuralNetwork neuralNetwork = NeuralNetwork.create(trainingSets[0][0].length, 20, trainingSets[0][1].length);
        for (int i = 0; i < 100000; i++) {
            int choice = random.nextInt(trainingSets.length);
            neuralNetwork.train(Arrays.asList(trainingSets[choice][0]),
                    Arrays.asList(trainingSets[choice][1]));
        }
        log.info("NN training finish!");
        print(neuralNetwork);
        log.warn("error: " + neuralNetwork.calculateError(trainingSets));
    }

    static void print(NeuralNetwork neuralNetwork) {
        log.info("test[0, 0]: " + neuralNetwork.prediction(Arrays.asList(0.0, 0.0))
                + "\ntest[0, 1]: " + neuralNetwork.prediction(Arrays.asList(0.0, 1.0))
                + "\ntest[1, 0]: " + neuralNetwork.prediction(Arrays.asList(1.0, 0.0))
                + "\ntest[1, 1]: " + neuralNetwork.prediction(Arrays.asList(1.0, 1.0))
                + "\ntest[2.0, 0.0]: " + neuralNetwork.prediction(Arrays.asList(2.0, 0.0))
                + "\ntest[2.0, 1.0]: " + neuralNetwork.prediction(Arrays.asList(2.0, 1.0))
                + "\ntest[2.0, 1.9]: " + neuralNetwork.prediction(Arrays.asList(2.0, 1.9))
                + "\ntest[1.1, 1.0]: " + neuralNetwork.prediction(Arrays.asList(1.1, 1.0))
                + "\ntest[0.9, 1.0]: " + neuralNetwork.prediction(Arrays.asList(0.9, 1.0))
                + "\ntest[1.0, 0.1]: " + neuralNetwork.prediction(Arrays.asList(1.0, 0.1)));
    }
}
