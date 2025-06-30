#include "Visualization.h"
#include <iostream>

int train_epoch = 0;
double total_loss = 0;
int max_epoch = 10000000000;

ALLEGRO_DISPLAY* Visualizer::display = nullptr;
ALLEGRO_EVENT_QUEUE* Visualizer::event_queue = nullptr;
ALLEGRO_FONT* Visualizer::font = nullptr;
ALLEGRO_FONT* Visualizer::font_axis = nullptr;

bool Visualizer::initialize() {
    if (!al_init() || !al_init_primitives_addon() || !al_install_keyboard())
        return false;

    ALLEGRO_MONITOR_INFO info;
    al_get_monitor_info(0, &info);
    int screen_width = info.x2 - info.x1;
    int screen_height = info.y2 - info.y1;

    display = al_create_display(screen_width, screen_height);
    event_queue = al_create_event_queue();
    al_init_font_addon();
    al_init_ttf_addon();

    font = al_load_ttf_font("VeraBd.ttf", 14, 0);
    font_axis = al_load_ttf_font("VeraBd.ttf", 11, 0);

    if (!display || !font || !font_axis) {
        std::cerr << "Failed to initialize display or font!" << std::endl;
        return false;
    }

    al_register_event_source(event_queue, al_get_keyboard_event_source());
    return true;
}

void Visualizer::draw_heatmap(int x_offset, int y_offset, int cell_w, int cell_h, NeuralNetwork& nn) {
    const int grid_size = 100;
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            double A = (i / (double)(grid_size - 1)) * 2 - 1;
            double B = (j / (double)(grid_size - 1)) * 2 - 1;
            double norm = sqrt(A * A + B * B);

            std::vector<double> input = { A, sin(norm), B, cos(norm) };
            std::vector<double> h;
            std::vector<double> output = nn.forward(input, h, false);

            double value = output[0];
            double norm_val = (value + 12.57) / (2 * 12.57);
            norm_val = std::clamp(norm_val, 0.0, 1.0);

            ALLEGRO_COLOR color = al_map_rgb_f(norm_val, 0.2, 1.0 - norm_val);

            float px = x_offset + (i / (float)grid_size) * cell_w;
            float py = y_offset + (j / (float)grid_size) * cell_h;
            al_draw_filled_rectangle(px, py, px + cell_w / (float)grid_size, py + cell_h / (float)grid_size, color);
        }
    }
    al_draw_text(font, al_map_rgb(255, 255, 255), x_offset + 10, y_offset + 10, 0, "Heatmap: f(A,B)");
}

void Visualizer::draw_sample_plot(int idx, int x_offset, int y_offset, int cell_w, int cell_h, 
                                NeuralNetwork& nn, const std::vector<std::vector<double>>& X, 
                                const std::vector<double>& Y, size_t sample_idx) {
    size_t start_index = sample_idx * (Y.size() / 120);
    size_t end_index = start_index + (Y.size() / 120);

    for (int i = 0; i < 16; ++i) {
        int gx = x_offset + (i * cell_w) / 10;
        int gy = y_offset + (i * cell_h) / 16;
        al_draw_line(gx, y_offset, gx, y_offset + cell_h, al_map_rgb(40, 40, 60), 1);
        al_draw_line(x_offset, gy, x_offset + cell_w, gy, al_map_rgb(40, 40, 60), 1);
    }

    al_draw_line(x_offset, y_offset + cell_h / 2, x_offset + cell_w, y_offset + cell_h / 2, al_map_rgb(100, 100, 100), 2);
    al_draw_line(x_offset + cell_w / 2, y_offset, x_offset + cell_w / 2, y_offset + cell_h, al_map_rgb(100, 100, 100), 2);

    for (int i = -5; i <= 5; ++i) {
        if (i == 5 && idx == 0) continue;
        double x_val = ((i + 5) * cell_w) / 10.0;
        al_draw_textf(font_axis, al_map_rgb(255, 255, 255), x_offset + x_val, y_offset + cell_h / 2 + 10, ALLEGRO_ALIGN_CENTER, "%.2f", (i / 5.0) * 4 * M_PI);
    }

    for (int i = -8; i <= 8; ++i) {
        double y_val = (cell_h / 2) - i * (cell_h / 16.0);
        if (i != 0)
            al_draw_textf(font_axis, al_map_rgb(255, 255, 255), x_offset + 10 + cell_w / 2, y_offset + y_val, ALLEGRO_ALIGN_LEFT, "%.2f", (i / 8.0) * 28);
    }

    double prev_x_real = -1, prev_y_real = -1;
    for (size_t i = start_index; i < end_index; i++) {
        std::vector<double> h;
        std::vector<double> pred = nn.forward(X[i], h, false);

        double f_pred = pred[0];
        double x_value = (i - start_index) * 0.05;
        double norm_x = x_value / (4 * M_PI);
        double x = x_offset + norm_x * cell_w;
        double x0 = x_offset + cell_w / 2 + ((cell_w / 2) / 12.57) * Y[i];
        double x_pred = x_offset + cell_w / 2 + ((cell_w / 2) / 12.57) * pred[0];

        double y_real = y_offset + (cell_h / 2) - (X[i][0] * X[i][1] + X[i][2] * X[i][3]) * (cell_h / 2.8);
        double y_predicted = y_offset + (cell_h / 2) - f_pred * (cell_h / 2.8);

        if (prev_x_real != -1)
            al_draw_line(prev_x_real, prev_y_real, x, y_real, al_map_rgb(255, 255, 255), 2);

        al_draw_filled_circle(x0, y_offset + (cell_h / 2), 5, al_map_rgb(255, 0, 0));
        al_draw_filled_circle(x_pred, y_offset + (cell_h / 2), 5, al_map_rgb(0, 255, 0));

        prev_x_real = x;
        prev_y_real = y_real;
    }

    al_draw_textf(font, al_map_rgb(180, 180, 255), x_offset + 10, y_offset + 20, 0, "Sample %d", idx);
}

void Visualizer::render(NeuralNetwork& nn, const std::vector<std::vector<double>>& X, const std::vector<double>& Y) {
    ALLEGRO_MONITOR_INFO info;
    al_get_monitor_info(0, &info);
    int screen_width = info.x2 - info.x1;
    int screen_height = info.y2 - info.y1;

    const int cols = 2, rows = 2;
    const int top_margin = 3;
    int cell_w = screen_width / cols;
    int cell_h = (screen_height - top_margin - 50) / rows;

    while (train_epoch <= max_epoch) {
        ALLEGRO_EVENT event;
        while (al_get_next_event(event_queue, &event)) {
            if (event.type == ALLEGRO_EVENT_KEY_DOWN) {
                if (event.keyboard.keycode == ALLEGRO_KEY_ESCAPE)
                    return;
                else if (event.keyboard.keycode == ALLEGRO_KEY_UP)
                    nn.learning_rate *= 1.1;
                else if (event.keyboard.keycode == ALLEGRO_KEY_DOWN)
                    nn.learning_rate *= 0.9;
            }
        }

        al_clear_to_color(al_map_rgb(15, 15, 20));
        al_draw_textf(font, al_map_rgb(255, 255, 255), 10, 3, 0,
            "Epoch: %d | Avg Loss: %.6f | Learning Rate: %.4f", train_epoch, total_loss / X.size(), nn.learning_rate);

        for (int idx = 0; idx < 4; ++idx) {
            int row = idx / cols;
            int col = idx % cols;
            int x_offset = col * cell_w;
            int y_offset = row * cell_h + top_margin;

            if (idx == 0) {
                draw_heatmap(x_offset, y_offset, cell_w, cell_h, nn);
            } else {
                draw_sample_plot(idx, x_offset, y_offset, cell_w, cell_h, nn, X, Y, idx);
            }
        }

        al_flip_display();
    }
}

void Visualizer::shutdown() {
    if (font) al_destroy_font(font);
    if (font_axis) al_destroy_font(font_axis);
    if (event_queue) al_destroy_event_queue(event_queue);
    if (display) al_destroy_display(display);
}