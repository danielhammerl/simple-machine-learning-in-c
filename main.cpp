#include <iostream>
#include <algorithm>
#include <array>
#include <random>
#include <chrono>
#include <filesystem>
#include <cmath>

static double sigmoidFunction(double x) {
    return 1.f / (1.f + std::exp(-x));
}

static double sigmoidDerivativeFunction(double x) {
    return x * (1 - x);
}

static double getRandomDouble(double min, double max) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<> dist(min, max);

    return dist(rng);
}

static double randomBetweenZeroAndOne() {
    return getRandomDouble(0, 1);
}

int main(int argc, char *argv[]) {

    std::cout << "start" << std::endl;

    const unsigned numberOfInputLayers = 2;
    const unsigned numberOfHiddenLayers = 2;
    const unsigned numberOfOutputLayers = 1;

    const double learningFactor = 0.5;
    double hiddenLayers[numberOfHiddenLayers];
    double outputLayers[numberOfOutputLayers];

    double inputToHiddenWeights[numberOfInputLayers][numberOfHiddenLayers] = {
            {randomBetweenZeroAndOne(), randomBetweenZeroAndOne(),},
            {randomBetweenZeroAndOne(), randomBetweenZeroAndOne()}};
    double hiddenToOutputWeights[numberOfHiddenLayers][numberOfOutputLayers] = {
            {randomBetweenZeroAndOne()},
            {randomBetweenZeroAndOne()}};;

    double globalBias = 0.1;

    unsigned int trainingData[4][numberOfInputLayers] = {{0, 0},
                                                         {1, 0},
                                                         {0, 1},
                                                         {1, 1}};

    unsigned int trainingOutput[4][numberOfOutputLayers] = {{0},
                                                            {1},
                                                            {1},
                                                            {0}};

    std::array<int, 4> trainingSetOrder{0, 1, 2, 3};
    int numberOfEpochs = 15000;
    int numberOfVerificationSteps = 100;

    for (int epoch = 0; epoch < numberOfEpochs; epoch++) {
        // obtain a time-based seed:
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        shuffle(trainingSetOrder.begin(), trainingSetOrder.end(), std::default_random_engine(seed));

        for (int x = 0; x < 4; x++) {
            int trainingSet = trainingSetOrder[x];

            for (int hiddenNodeIndex = 0; hiddenNodeIndex < numberOfHiddenLayers; hiddenNodeIndex++) {
                double activation = 0;//hiddenLayerBias[hiddenNodeIndex];

                for (int inputIndex = 0; inputIndex < numberOfInputLayers; inputIndex++) {
                    activation +=
                            trainingData[trainingSet][inputIndex] * inputToHiddenWeights[inputIndex][hiddenNodeIndex];
                }

                hiddenLayers[hiddenNodeIndex] = sigmoidFunction(activation - globalBias);
            }

            for (int outputNodeIndex = 0; outputNodeIndex < numberOfOutputLayers; outputNodeIndex++) {
                double activation = 0;//outputLayerBias[outputNodeIndex];

                for (int hiddenIndex = 0; hiddenIndex < numberOfHiddenLayers; hiddenIndex++) {
                    activation +=
                            hiddenLayers[hiddenIndex] * hiddenToOutputWeights[hiddenIndex][outputNodeIndex];
                }

                outputLayers[outputNodeIndex] = sigmoidFunction(activation - globalBias);
            }

            //std::cout << "input: " << trainingData[trainingSet][0] << "-" << trainingData[trainingSet][1] << " output: "
            //         << round(outputLayers[0]) << " expected output: " << trainingOutput[trainingSet][0] << std::endl;

            // backpropagation

            double deltaOutput[numberOfOutputLayers];
            for (int output = 0; output < numberOfOutputLayers; output++) {
                double error = trainingOutput[trainingSet][output] - outputLayers[output];
                deltaOutput[output] = error * sigmoidDerivativeFunction(outputLayers[output]);
            }

            double deltaHidden[numberOfHiddenLayers];
            for (int hidden = 0; hidden < numberOfHiddenLayers; hidden++) {
                double error = 0.0;
                for (int output = 0; output < numberOfOutputLayers; output++) {
                    error += deltaOutput[output] * hiddenToOutputWeights[hidden][output];
                }
                deltaHidden[hidden] = error * sigmoidDerivativeFunction(hiddenLayers[hidden]);
            }

            std::cout << inputToHiddenWeights[0][0] << " " << inputToHiddenWeights[0][1] << " "
                      << inputToHiddenWeights[1][0] << " " << inputToHiddenWeights[1][1] << std::endl;
            std::cout << hiddenToOutputWeights[0][0] << " " << hiddenToOutputWeights[1][0] << std::endl;
            std::cout << "-" << std::endl;

            //apply changes in weights
            for (int output = 0; output < numberOfOutputLayers; output++) {
                //outputLayerBias[output] += deltaOutput[output] * learningFactor;
                for (int hidden = 0; hidden < numberOfHiddenLayers; hidden++) {
                    hiddenToOutputWeights[hidden][output] +=
                            hiddenLayers[hidden] * deltaOutput[output] * learningFactor;
                }
            }

            for (int hidden = 0; hidden < numberOfHiddenLayers; hidden++) {
                //hiddenLayerBias[hidden] += deltaHidden[hidden] * learningFactor;
                for (int input = 0; input < numberOfInputLayers; input++) {
                    inputToHiddenWeights[input][hidden] +=
                            trainingData[trainingSet][input] * deltaHidden[hidden] * learningFactor;
                }
            }
        }
    }

    bool works = true;
    int errorAt = 0;
    for (int verificationStep = 0; verificationStep < numberOfVerificationSteps && works; verificationStep++) {
        // obtain a time-based seed:
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        shuffle(trainingSetOrder.begin(), trainingSetOrder.end(), std::default_random_engine(seed));

        for (int x = 0; x < 4; x++) {
            int trainingSet = trainingSetOrder[x]; // i

            for (int hiddenNodeIndex = 0; hiddenNodeIndex < numberOfHiddenLayers; hiddenNodeIndex++) {
                double activation = 0;//hiddenLayerBias[hiddenNodeIndex];

                for (int inputIndex = 0; inputIndex < numberOfInputLayers; inputIndex++) {
                    activation +=
                            trainingData[trainingSet][inputIndex] * inputToHiddenWeights[inputIndex][hiddenNodeIndex];
                }

                hiddenLayers[hiddenNodeIndex] = sigmoidFunction(activation);
            }

            for (int outputNodeIndex = 0; outputNodeIndex < numberOfOutputLayers; outputNodeIndex++) {
                double activation = 0;//outputLayerBias[outputNodeIndex];

                for (int hiddenIndex = 0; hiddenIndex < numberOfHiddenLayers; hiddenIndex++) {
                    activation +=
                            hiddenLayers[hiddenIndex] * hiddenToOutputWeights[hiddenIndex][outputNodeIndex];
                }

                outputLayers[outputNodeIndex] = sigmoidFunction(activation);
            }

            if (round(outputLayers[0]) != trainingOutput[trainingSet][0]) {
                works = false;
                errorAt = verificationStep;
                break;
            }
        }
    }

    std::cout << (works ? "OK" : "Not ok") << std::endl;
    if (works) {
        std::cout << inputToHiddenWeights[0][0] << " " << inputToHiddenWeights[0][1] << " "
                  << inputToHiddenWeights[1][0] << " " << inputToHiddenWeights[1][1] << std::endl;
        std::cout << hiddenToOutputWeights[0][0] << " " << hiddenToOutputWeights[1][0] << std::endl;

        for (auto &x: trainingData) {
            for (int hiddenNodeIndex = 0; hiddenNodeIndex < numberOfHiddenLayers; hiddenNodeIndex++) {
                double activation = 0;//hiddenLayerBias[hiddenNodeIndex];

                for (int inputIndex = 0; inputIndex < numberOfInputLayers; inputIndex++) {
                    activation +=
                            x[inputIndex] * inputToHiddenWeights[inputIndex][hiddenNodeIndex];
                }

                hiddenLayers[hiddenNodeIndex] = sigmoidFunction(activation);
            }

            for (int outputNodeIndex = 0; outputNodeIndex < numberOfOutputLayers; outputNodeIndex++) {
                double activation = 0;//outputLayerBias[outputNodeIndex];

                for (int hiddenIndex = 0; hiddenIndex < numberOfHiddenLayers; hiddenIndex++) {
                    activation +=
                            hiddenLayers[hiddenIndex] * hiddenToOutputWeights[hiddenIndex][outputNodeIndex];
                }

                outputLayers[outputNodeIndex] = sigmoidFunction(activation);
            }

            std::cout << "input " << x[0] << " and " << x[1] << " results to "
                      << round(outputLayers[0]) << std::endl;
        }


    } else {
        std::cout << errorAt << std::endl;
    }
}
