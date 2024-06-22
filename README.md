# Sentiment Analysis of IMDB Movie Reviews

This repository contains the implementation and evaluation of various machine learning and deep learning models for sentiment analysis on IMDB movie reviews dataset. The goal is to predict whether a review is positive or negative based on the text content.

## Dataset
The IMDB movie reviews dataset consists of 50,000 movie reviews labeled as positive or negative. It is commonly used for sentiment analysis tasks and provides a balanced distribution of reviews across classes.

## Models Implemented
- **Machine Learning Models:**
  - Multinomial Naive Bayes
  - Logistic Regression

- **Deep Learning Models:**
  - Recurrent Neural Network (RNN)
  - Long Short-Term Memory (LSTM)
  - Stacked LSTM
  - Gated Recurrent Unit (GRU)

## Evaluation Metrics
Each model was evaluated based on the following metrics:
- Accuracy
- Precision
- Recall
- F1-score

## Results
The models were trained and evaluated using standard machine learning and deep learning techniques. Here are the accuracy results obtained:

- Multinomial Naive Bayes: 86.54%
- Logistic Regression: 88.97%
- RNN: 77.57%
- LSTM: 82.14%
- Stacked LSTM: 82.83%
- GRU: 82.91%

## Conclusion
Based on the accuracy results, Logistic Regression outperformed the deep learning models on this specific task. However, the choice of model can depend on various factors such as interpretability, computational resources, and the need for further improvements. The repository provides insights into the performance and implementation of each model for sentiment analysis.

## Usage
You can clone this repository to reproduce the results or further experiment with different models and hyperparameters. Each model's implementation is provided in separate notebooks or scripts for clarity and ease of understanding.

## Requirements
- Python (version)
- Libraries: TensorFlow, Keras, Scikit-learn, NLTK, etc.

---
