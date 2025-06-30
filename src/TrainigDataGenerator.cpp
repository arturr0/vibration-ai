#include "TrainingDataGenerator.h"
#include "constants.h"
#include <cmath>

void DataGenerator::generate_training_data(std::vector<std::vector<double>>& X, std::vector<double>& Y) {
    const int steps = static_cast<int>(XMAX / STEP);
    const double threshold = 1;

    for (int A = -5; A <= 5; ++A) {
        for (int B = -5; B <= 5; ++B) {
            double norm_A = normalize(A);
            double norm_B = normalize(B);
            double magnitude = std::sqrt(A * A + B * B);
            if (magnitude == 0)
                continue;

            double x_end = -2.0 / DAMPING * std::log(threshold / magnitude);

            for (int i = 0; i <= steps; ++i) {
                double x = i * STEP;
                double decay = exp(-DAMPING * x);
                X.push_back({ norm_A, decay * sin(FREQUENCY * x), norm_B, decay * cos(FREQUENCY * x) });
                Y.push_back(x_end);
            }
        }
    }
}

std::vector<std::vector<double>> DataGenerator::generate_sin_cos_points(double A, double B) {
    double norm_A = normalize(A);
    double norm_B = normalize(B);
    std::vector<std::vector<double>> points;
    for (double x = 0; x <= 4 * M_PI; x += 0.1) {
        points.push_back({ norm_A, sin(x), norm_B, cos(x) });
    }
    return points;
}