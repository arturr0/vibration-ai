#pragma once
#include <vector>
#include <mutex>
#include <random>
#include "constants.h"

class NeuralNetwork {
public:
    NeuralNetwork(int in_size, int hid_size, int out_size, double lr = 0.1);
    void train(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, int epochs);
    std::vector<double> forward(const std::vector<double>& input, std::vector<double>& hidden_out, bool apply_sigmoid = false);
    double predict_from_sin_cos_points(const std::vector<std::vector<double>>& sin_values);

private:
    void initialize_weights();
    void train_sample(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, int sample_start, int sample_end);
    void store_last_parameters();
    void restore_last_parameters();
    
    std::vector<std::vector<double>> W1, W2, last_W1, last_W2;
    std::vector<double> b1, b2, last_b1, last_b2;
    int input_size, hidden_size, output_size;
    double learning_rate;
    std::mutex mtx;

    inline double relu(double x) { return (x > 0) ? x : 0.0; }
    inline double relu_derivative(double x) { return (x > 0) ? 1.0 : 0.0; }
};