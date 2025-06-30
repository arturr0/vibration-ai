#include "NeuralNetwork.h"
#include <iostream>
#include <algorithm>
#include <thread>

NeuralNetwork::NeuralNetwork(int in_size, int hid_size, int out_size, double lr) 
    : input_size(in_size), hidden_size(hid_size), output_size(out_size), learning_rate(lr) {
    initialize_weights();
}

void NeuralNetwork::initialize_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    W1.resize(hidden_size, std::vector<double>(input_size));
    W2.resize(output_size, std::vector<double>(hidden_size));
    b1.resize(hidden_size);
    b2.resize(output_size);

    for (auto& row : W1)
        for (auto& w : row)
            w = dist(gen);
    for (auto& row : W2)
        for (auto& w : row)
            w = dist(gen);
    for (auto& b : b1)
        b = dist(gen);
    for (auto& b : b2)
        b = dist(gen);

    last_W1 = W1;
    last_W2 = W2;
    last_b1 = b1;
    last_b2 = b2;
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& input, std::vector<double>& hidden_out, bool apply_sigmoid) {
    std::vector<double> hidden(hidden_size, 0.0);
    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < input_size; j++)
            hidden[i] += W1[i][j] * input[j];
        hidden[i] += b1[i];
        hidden[i] = relu(hidden[i]);
    }

    std::vector<double> output(output_size, 0.0);
    for (int i = 0; i < output_size; i++) {
        output[i] = 0.0;
        for (int j = 0; j < hidden_size; j++)
            output[i] += W2[i][j] * hidden[j];
        output[i] += b2[i];
    }

    hidden_out = hidden;
    return output;
}

void NeuralNetwork::store_last_parameters() {
    last_W1 = W1;
    last_W2 = W2;
    last_b1 = b1;
    last_b2 = b2;
}

void NeuralNetwork::restore_last_parameters() {
    W1 = last_W1;
    W2 = last_W2;
    b1 = last_b1;
    b2 = last_b2;
}

void NeuralNetwork::train_sample(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, int sample_start, int sample_end) {
    double local_loss = 0.0;

    for (size_t sample = sample_start; sample < sample_end; sample++) {
        auto x = X[sample];
        auto y_true = Y[sample];
        std::vector<double> h;
        auto y_pred = forward(x, h, false);

        std::vector<double> error(output_size);
        double loss = 0.0;

        for (int i = 0; i < output_size; i++) {
            error[i] = y_pred[i] - y_true;
            loss += error[i] * error[i];
        }
        local_loss += loss / output_size;

        std::vector<std::vector<double>> dW2(output_size, std::vector<double>(hidden_size, 0.0));
        std::vector<double> db2(output_size, 0.0);
        std::vector<double> dH(hidden_size, 0.0);

        for (int i = 0; i < output_size; i++) {
            double delta_out = error[i];
            for (int j = 0; j < hidden_size; j++) {
                dW2[i][j] = delta_out * h[j];
                dH[j] += W2[i][j] * delta_out;
            }
            db2[i] = delta_out;
        }

        std::vector<std::vector<double>> dW1(hidden_size, std::vector<double>(input_size, 0.0));
        std::vector<double> db1(hidden_size, 0.0);

        for (int i = 0; i < hidden_size; i++) {
            double delta_hidden = dH[i] * relu_derivative(h[i]);
            for (int j = 0; j < input_size; j++) {
                dW1[i][j] = delta_hidden * x[j];
            }
            db1[i] = delta_hidden;
        }

        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                W2[i][j] -= learning_rate * dW2[i][j];
            }
            b2[i] -= learning_rate * db2[i];
        }

        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < input_size; j++) {
                W1[i][j] -= learning_rate * dW1[i][j];
            }
            b1[i] -= learning_rate * db1[i];
        }
    }

    std::lock_guard<std::mutex> guard(mtx);
    total_loss += local_loss;
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, int epochs) {
    double last_loss = std::numeric_limits<double>::max();

    for (int epoch = 0; epoch < epochs; epoch++) {
        total_loss = 0.0;
        std::vector<std::thread> threads;
        size_t batch_size = X.size() / 4;

        for (int i = 0; i < 4; i++) {
            size_t sample_start = i * batch_size;
            size_t sample_end = (i == 3) ? X.size() : (i + 1) * batch_size;
            threads.push_back(std::thread(&NeuralNetwork::train_sample, this, std::ref(X), std::ref(Y), sample_start, sample_end));
        }

        for (auto& t : threads)
            t.join();

        if (isnan(total_loss) || total_loss > 1e10) {
            restore_last_parameters();
            learning_rate *= 0.9;
        }
        else {
            if (total_loss > last_loss)
                learning_rate *= 0.95;
            else
                store_last_parameters();

            last_loss = total_loss;
        }

        train_epoch++;
        if (train_epoch > max_epoch)
            break;
    }
}

double NeuralNetwork::predict_from_sin_cos_points(const std::vector<std::vector<double>>& sin_values) {
    double sum_A = 0.0;
    for (const auto& x : sin_values) {
        std::vector<double> h;
        std::vector<double> pred = forward(x, h, true);
        sum_A += pred[0];
    }
    double avg_A = sum_A / sin_values.size();
    std::cout << "Average predicted A: " << denormalize(avg_A) << std::endl;
    return denormalize(avg_A);
}