import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# Sample data generation
def generate_sample_data():
    # Sample texts with sentiment labels
    data = {
        'text': [
            "This product is absolutely amazing! I love it.",
            "Great experience, would recommend to everyone.",
            "Not bad, but could be better.",
            "Mediocre at best, wouldn't buy again.",
            "Terrible product, complete waste of money.",
            "I'm so disappointed with this purchase.",
            "This exceeded my expectations!",
            "Average product, nothing special.",
            "The quality is outstanding, very pleased!",
            "I regret buying this, it's awful.",
            "The service was excellent and prompt.",
            "It broke after a week, avoid at all costs.",
            "Decent value for money but not great.",
            "This is the worst product I've ever used.",
            "Pretty good overall, minor issues only.",
            "I'm very satisfied with my purchase.",
            "Not worth the price, wouldn't recommend.",
            "Absolutely love it, would buy again!",
            "It's okay, does the job but nothing special.",
            "Total disappointment, don't waste your money."
        ],
        'sentiment': [
            'positive',
            'positive',
            'neutral',
            'negative',
            'negative',
            'negative',
            'positive',
            'neutral',
            'positive',
            'negative',
            'positive',
            'negative',
            'neutral',
            'negative',
            'positive',
            'positive',
            'negative',
            'positive',
            'neutral',
            'negative'
        ]
    }
    return pd.DataFrame(data)

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, keeping only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Apply preprocessing to the data
def prepare_data(df):
    # Apply text preprocessing
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df['sentiment'], 
        test_size=0.3, 
        random_state=42,
        stratify=df['sentiment']
    )
    
    return X_train, X_test, y_train, y_test

# Train sentiment analysis model
def train_model(X_train, y_train):
    # Create pipeline with TF-IDF and Logistic Regression
    sentiment_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8)),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Train the model
    sentiment_pipeline.fit(X_train, y_train)
    
    return sentiment_pipeline

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = sorted(y_test.unique())
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return accuracy

# Test with new examples
def test_with_examples(model):
    examples = [
        "This is the best purchase I've made all year!",
        "It's not terrible, but I wouldn't buy it again.",
        "Absolute garbage, avoid at all costs!",
        "It works fine for what I need it for.",
        "I'm so happy with this product, it's perfect!"
    ]
    
    # Preprocess examples
    processed_examples = [preprocess_text(ex) for ex in examples]
    
    # Make predictions
    predictions = model.predict(processed_examples)
    
    # Print results
    print("\nTesting with new examples:")
    for i, (example, prediction) in enumerate(zip(examples, predictions)):
        print(f"Example {i+1}: \"{example}\"")
        print(f"Predicted sentiment: {prediction}\n")

def main():
    print("Generating sample sentiment data...")
    df = generate_sample_data()
    
    print("\nPreparing data...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    print("\nTraining sentiment analysis model...")
    model = train_model(X_train, y_train)
    
    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)
    
    test_with_examples(model)
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main() 