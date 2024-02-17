# Traffic Prediction with Neural Networks

This repository contains code for predicting traffic volume using various neural network architectures including Multi-Layer Perceptron (MLP), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU). The models are trained and evaluated on historical traffic data obtained from multiple junctions.

## Overview

Traffic prediction plays a crucial role in urban planning, transportation management, and resource allocation. Accurate forecasts of traffic volume enable authorities to optimize traffic flow, improve safety, and minimize congestion. This project aims to develop machine learning models capable of predicting future traffic volume based on historical patterns and environmental factors.

## Dataset

The dataset used in this project is sourced from [provide data source or describe how it was collected]. It consists of historical traffic data recorded at various junctions over a specific period. Each record includes attributes such as DateTime, Number of Vehicles, and Junction ID. Prior to model training, the dataset undergoes preprocessing steps including:

- Feature engineering: Extracting relevant features such as year, month, day, hour, and day of the week from the DateTime attribute.
- Normalization: Scaling numerical features to a standard range to ensure consistent model training.
- Data splitting: Segmenting the dataset into training and testing sets to facilitate model evaluation.

## Usage

To run the project:

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/traffic-prediction.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the main script:

    ```bash
    python traffic_prediction.py
    ```

## Models

### MLP

The Multi-Layer Perceptron (MLP) model is a feedforward neural network architecture consisting of multiple layers of neurons. In this project, the MLP model is employed to learn complex relationships between input features and traffic volume. The model architecture includes:

- Fully connected layers with ReLU activation functions.
- Dropout layers for regularization to prevent overfitting.
- Stochastic Gradient Descent (SGD) optimizer with a learning rate schedule.

### GRU

The Gated Recurrent Unit (GRU) is a type of recurrent neural network (RNN) architecture designed to capture temporal dependencies in sequential data. The GRU model employed in this project comprises:

- Multiple GRU layers to learn temporal patterns in the traffic data.
- Dropout layers for regularization to enhance generalization.
- SGD optimizer with a learning rate schedule for model training.

### LSTM

The Long Short-Term Memory (LSTM) model is another variant of recurrent neural networks known for its ability to capture long-term dependencies in sequential data. The LSTM model used in this project consists of:

- LSTM layers with memory cells to retain and update information over time.
- Dropout layers to prevent overfitting by randomly disabling neurons during training.
- SGD optimizer with a learning rate schedule for efficient model convergence.

## Evaluation

Model performance is evaluated using standard metrics including Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). These metrics quantify the disparity between predicted and actual traffic volumes, providing insights into the models' accuracy and robustness. Additionally, comparative plots are generated to visualize the predicted traffic volume alongside the actual values, facilitating qualitative assessment.

## Contributing

Contributions to this project are welcome! If you have suggestions for improvements, encounter any issues, or wish to contribute new features, feel free to open an issue or submit a pull request. Together, we can enhance the accuracy and effectiveness of traffic prediction models.

## License

This project is licensed under the MIT License. You are free to modify, distribute, and use the code for both commercial and non-commercial purposes. See the [LICENSE](LICENSE) file for details.
