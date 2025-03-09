## NLP Project: Sentiment Analysis Using Logistic Regression and Random Forest

### Project Overview

This project focuses on sentiment analysis using text classification techniques, specifically logistic regression and random forest classifiers. The goal is to predict whether a given tweet is indicative of an actual disaster or not. The project is based on the Kaggle competition called "Disaster Tweets," where the dataset and additional information can be found.

### Methods Used

    Lemmatization: The text data is preprocessed using lemmatization, which reduces words to their base form. This helps in standardizing the vocabulary and improving the accuracy of the models.

    Data Processing: The text data is converted to lowercase and spaces are replaced with underscores to ensure consistency and facilitate further analysis.

    Feature Extraction: The TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique is used to convert the text data into numerical features. This approach calculates the importance of each word in the tweet based on its frequency and rarity across the entire corpus.

    Model Training and Evaluation:
        Logistic Regression: A logistic regression model is trained on the preprocessed text data and corresponding target labels. This model is well-suited for binary classification tasks like sentiment analysis.
        Random Forest: A random forest classifier is trained on the same preprocessed data. Random forests are an ensemble learning method that combines multiple decision trees to make predictions.

    Model Comparison: The classification reports are generated for both logistic regression and random forest models to evaluate their performance. The metrics in the reports provide insights into precision, recall, and F1-score for each class, helping assess the models' effectiveness.

### Best Performing Models

From the various algorithms used, both logistic regression and random forest classifiers demonstrated the best performance.
