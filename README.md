# House Price Prediction using a Neural Network

A deep learning project to predict the median value of homes in Boston suburbs using the Keras API with a TensorFlow backend. This project follows the tutorial by FreeCodeCamp, demonstrating fundamental concepts of building and training a neural network for a regression task.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [License](#license)

## Project Overview

This project implements a simple feedforward neural network to solve a regression problem: predicting house prices. The key steps include:
- Loading the Boston housing dataset.
- Preprocessing the data by splitting and scaling (normalization).
- Building a `Sequential` model in Keras with two hidden layers.
- Compiling and training the model.
- Evaluating the model's performance using Mean Squared Error (MSE) and R-squared ($R^2$).

## Technologies Used

- Python 3.9
- TensorFlow / Keras
- Pandas
- Scikit-learn
- Jupyter Notebook

## Dataset

The project uses the Boston Housing Price dataset, sourced from the StatLib library at Carnegie Mellon University. It contains 506 samples with 13 feature variables, such as crime rate, number of rooms, and distance to employment centers. The target variable is the median value of owner-occupied homes (MEDV).

## Model Architecture

The neural network is a `Sequential` model with the following structure:

1.  **Input Layer:** Expects 13 input features.
2.  **Hidden Layer 1:** A `Dense` layer with 128 neurons and a `ReLU` activation function.
3.  **Hidden Layer 2:** A `Dense` layer with 64 neurons and a `ReLU` activation function.
4.  **Output Layer:** A `Dense` layer with a single neuron and a `linear` activation function to predict the continuous price value.

The model is compiled with the `Adam` optimizer and `mean_squared_error` as the loss function.

## Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Keras-House-Price-Prediction.git](https://github.com/YourUsername/Keras-House-Price-Prediction.git)
    cd Keras-House-Price-Prediction
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create --name house-price-predictor python=3.9
    conda activate house-price-predictor
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  Ensure your Conda environment (`house-price-predictor`) is active.
2.  Launch Jupyter Notebook from your terminal:
    ```bash
    jupyter notebook
    ```
3.  In the Jupyter interface that opens in your browser, navigate to and open the `boston_house_price_prediction.ipynb` file.
4.  Run the cells sequentially from top to bottom.

## Results

After training for 100 epochs, the model was evaluated on the unseen test set.

- **Test Set Mean Squared Error (MSE):** `21.53` (Your value may vary slightly)
- **R-squared ($R^2$):** `0.71` (Your value may vary slightly)

An $R^2$ value of 0.71 indicates that the model can explain approximately 71% of the variance in the test set's house prices, which is a reasonable result for this simple model.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
