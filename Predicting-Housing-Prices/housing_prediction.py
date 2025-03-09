import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load housing data
# In a real scenario, replace with actual data path
def load_data(filepath='housing_data.csv'):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        # Sample data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic data
        X = np.random.normal(size=(n_samples, 5))
        y = 3 + 2*X[:, 0] - 1*X[:, 1] + 0.5*X[:, 2] + np.random.normal(size=n_samples)
        
        df = pd.DataFrame(X, columns=['size', 'bedrooms', 'bathrooms', 'age', 'location_score'])
        df['price'] = y
        return df

# Prepare data
def prepare_data(df):
    X = df.drop('price', axis=1)
    y = df['price']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression model
def train_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Random forest model
def train_rf_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model evaluation results:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    return mse, rmse, r2

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Housing Prices')
    plt.tight_layout()
    plt.savefig('prediction_plot.png')
    plt.close()

def main():
    print("Loading housing data...")
    df = load_data()
    
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    print("Training linear regression model...")
    linear_model = train_linear_model(X_train, y_train)
    linear_mse, linear_rmse, linear_r2 = evaluate_model(linear_model, X_test, y_test)
    
    print("\nTraining random forest model...")
    rf_model = train_rf_model(X_train, y_train)
    rf_mse, rf_rmse, rf_r2 = evaluate_model(rf_model, X_test, y_test)
    
    # Plot predictions for the better model
    if rf_r2 > linear_r2:
        print("\nRandom Forest performs better!")
        plot_predictions(y_test, rf_model.predict(X_test))
    else:
        print("\nLinear Regression performs better!")
        plot_predictions(y_test, linear_model.predict(X_test))

if __name__ == "__main__":
    main() 