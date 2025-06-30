#pragma once
#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>
#include <allegro5/allegro_font.h>
#include <allegro5/allegro_ttf.h>
#include "NeuralNetwork.h"

extern int train_epoch;
extern double total_loss;
extern int max_epoch;

class Visualizer {
public:
    static bool initialize();
    static void render(NeuralNetwork& nn, 
                      const std::vector<std::vector<double>>& X, 
                      const std::vector<double>& Y);
    static void shutdown();

private:
    static ALLEGRO_DISPLAY* display;
    static ALLEGRO_EVENT_QUEUE* event_queue;
    static ALLEGRO_FONT* font;
    static ALLEGRO_FONT* font_axis;
    
    static void draw_heatmap(int x_offset, int y_offset, int cell_w, int cell_h, NeuralNetwork& nn);
    static void draw_sample_plot(int idx, int x_offset, int y_offset, int cell_w, int cell_h, 
                               NeuralNetwork& nn, const std::vector<std::vector<double>>& X, 
                               const std::vector<double>& Y, size_t sample_idx);
};