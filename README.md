
# Physics-Informed LSTM for Methane Production Prediction

This repository contains the implementation of a Physics-Informed LSTM (PI-LSTM) model developed to predict methane production in a continuous anaerobic co-digestion process using experimental process data.

## Project Overview

The goal of this model is to integrate domain knowledge—mass balance constraints—into a data-driven LSTM architecture to improve predictive performance and interpretability in environmental biotechnology systems.

## Key Features

- Uses a sequence-based LSTM to model time-series data related to biogas production.
- Integrates a physics-based constraint into the loss function that enforces mass conservation:
  `Methane = alpha × (Q_s + Q_F - Q_d)`
- Trains on real process data with normalized input features such as:
  - Retention time
  - Operation time
  - Flow rates of sludge (Q_s), food waste (Q_F), and digestate (Q_d)
- Visualizes performance metrics and feature importance using Captum for interpretability.

## Requirements

- Python 3.7+
- PyTorch
- scikit-learn
- pandas, numpy, matplotlib

## Repository Contents

- `cleaned_pi_lstm.py`: Main model and training script
- `data/`: Placeholder path for input CSV file
- `notebooks/`: Optional space for exploration or visualizations

## Usage

1. Place your time-series CSV file with appropriate headers in the correct path.
2. Run the Python script to preprocess, train, and evaluate the model.
3. Visualize training/testing loss, R² values, and feature importances.

## Example Output

- R² and RMSE on train/test sets
- Scatter plot of actual vs predicted methane production
- Integrated gradients for feature attribution
- Learned `alpha` values over training epochs

## Author

Maryam Ghazizade Fard  
PhD Candidate, Queen’s University  
Email: m.ghazizadefard@queensu.ca
