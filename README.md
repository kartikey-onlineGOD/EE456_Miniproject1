# EE 456 Mini-Project: Perceptron Implementation and Analysis

## Overview

This project implements a Single-Layer Neural Network (Perceptron) for binary classification tasks. It explores the Perceptron's performance on both linearly separable and non-linearly separable datasets, focusing on the effects of varying learning rates and thresholds.

## Features

- Implementation of the Perceptron algorithm
- Interactive dashboard for visualizing results
- Analysis of Perceptron performance on different datasets
- Exploration of learning rate and threshold effects

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Dash
- Plotly
- SciPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/EE456-Mini-Project.git
   ```
2. Navigate to the project directory:
   ```
   cd EE456-Mini-Project
   ```
3. Install required packages:
   ```
   pip install numpy matplotlib dash plotly scipy
   ```

## Usage

1. Run the main script:
   ```
   python perceptron_dashboard.py
   ```
2. Open a web browser and go to `http://127.0.0.1:8050/` to view the dashboard.
3. Use the dashboard to experiment with different learning rates and thresholds on both linearly separable and non-linearly separable datasets.

## Project Structure

- `perceptron_dashboard.py`: Main script containing the Perceptron implementation and dashboard
- `dataset1.mat`: Linearly separable dataset
- `dataset2.mat`: Non-linearly separable dataset

## Results

The project demonstrates:
- The Perceptron's ability to perfectly separate linearly separable data
- Limitations of the Perceptron on non-linearly separable data
- Effects of learning rate on convergence speed and stability
- Impact of threshold adjustment on the decision boundary

## Contributors

- [Your Name]

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
