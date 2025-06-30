#include "NeuralNetwork.h"
#include "TrainingDataGenerator.h"
#include "Visualization.h"
#include <iostream>
#include <thread>

int main() {
    std::vector<std::vector<double>> X;
    std::vector<double> Y;
    DataGenerator::generate_training_data(X, Y);

    NeuralNetwork nn(4, 15, 1, 0.001);

    if (!Visualizer::initialize()) {
        std::cerr << "Failed to initialize visualization!" << std::endl;
        return 1;
    }

    std::thread renderThread([&]() {
        Visualizer::render(nn, X, Y);
    });

    double new_A, new_B;
    std::cout << "Enter new amplitudes (A and B): ";
    std::cin >> new_A >> new_B;
    auto sin_points = DataGenerator::generate_sin_cos_points(new_A, new_B);
    double predicted_A = nn.predict_from_sin_cos_points(sin_points);
    std::cout << "Predicted A from sine points: " << predicted_A << std::endl;

    renderThread.join();
    Visualizer::shutdown();
    return 0;
}