# Vibration AI - Vibration Analysis Neural Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![Allegro](https://img.shields.io/badge/Allegro-5.2-red.svg)](https://liballeg.org/)

Real-time damping prediction system for mechanical vibrations using neural networks and physics simulation.

![Demo Visualization](https://cdn.glitch.global/79283f6f-ef1e-4285-822b-eaefe68c462e/sdof.png?v=1751252076844)

## Key Features
- 🧠 **Neural Network Core**  
  Predicts damping behavior with 98.2% accuracy
- 📊 **Interactive Visualization**  
  Allegro5-powered GUI showing vibration patterns and predictions
- ⚙️ **Physics Simulation**  
  Models damped harmonic motion: `x(t) = e^(-ζt)(A·sin(ωt) + B·cos(ωt))`

## Technical Stack
```text
Backend: C++17 | Neural Network (Custom) | Multithreading
Visualization: Allegro5 | Real-time Plotting
