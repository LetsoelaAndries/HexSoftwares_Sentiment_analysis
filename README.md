 # Restaurant Review Sentiment Analysis

## Overview

This project is aimed at analyzing sentiment from restaurant reviews using natural language processing (NLP) techniques. The goal is to determine whether a given review expresses a positive or negative sentiment, providing useful insights for both restaurant owners and customers. The sentiment analysis is done using machine learning algorithms and NLP models.

## Features

- **Sentiment Classification**: The system classifies restaurant reviews into positive, negative, or neutral categories based on the text provided.
- **Data Preprocessing**: Includes text cleaning, tokenization, and lemmatization to prepare the review data for analysis.
- **Model Training**: Utilizes machine learning models such as Logistic Regression, Naive Bayes Machines, and RandomForestClassifier classify the reviews.
- **Accuracy Metrics**: Evaluates the model's performance using metrics such as accuracy, precision, recall, and F1-score.

## Technologies Used

- **Python**: Programming language used to build the project.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For implementing machine learning models and evaluation metrics.
- **NLTK/Spacy**: For natural language processing tasks like tokenization, stopword removal, and lemmatization.
  
- **Matplotlib/Seaborn**: For data visualization (Word Cloud).
- **Jupyter Notebook**: For developing and testing the models interactively.

## Getting Started

### Prerequisites

To run this project locally, ensure you have the following installed:

- Python 3.x
- Jupyter Notebook
- Libraries: Pandas, NumPy, NLTK, Scikit-learn, Matplotlib, Seaborn

You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn nltk seaborn matplotlib
### Cloning repository
git clone https://github.com/LetsoelaAndries/HexSoftwares_Sentiment_analysis.git

## Usages

-  Preprocessing Data: Clean and prepare the text data by removing unwanted characters, tokenizing the words, and applying lemmatization.
-  Training the Model: You can use models like Logistic Regression, SVM, or deep learning-based approaches like LSTM for sentiment classification.
-  Evaluating the Model: The model will output metrics such as accuracy, confusion matrix, precision, recall, and F1 score to evaluate its performance
-   Predicting Sentiments: Once the model is trained, you can input new restaurant reviews and get predictions on whether they are positive, negative, or neutral

## Results
-  Accuracy: 76%
-  Precision: 
-  Recall:
-  F1 Score:
## Future Work
-  Improve Model: You can experiment with other models such as XGBoost, or fine-tune deep learning models for better accuracy.
-  Deploy the Model: Consider deploying the model as a web service or integrating it with a restaurant review website.
