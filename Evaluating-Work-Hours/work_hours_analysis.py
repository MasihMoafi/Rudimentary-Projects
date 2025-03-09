import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Generate sample work hours data
def generate_sample_data():
    np.random.seed(42)
    days = np.arange(1, 101)
    
    # Generate hours worked with some pattern
    hours = 8 + 2 * np.sin(days / 10) + np.random.normal(0, 1, size=len(days))
    hours = np.clip(hours, 4, 12)  # Realistic work hour bounds
    
    # Generate productivity based on hours with diminishing returns
    base_productivity = -0.5 * (hours - 8) ** 2 + 10
    day_effect = -np.sin(days / 7 * 2 * np.pi)  # Weekly cycle
    productivity = base_productivity + day_effect + np.random.normal(0, 1, size=len(days))
    productivity = np.clip(productivity, 0, 15)
    
    # Create DataFrame
    df = pd.DataFrame({
        'day': days,
        'day_of_week': (days % 7) + 1,
        'hours_worked': hours,
        'productivity': productivity,
        'tasks_completed': np.round(productivity / 2 + np.random.normal(0, 0.5, size=len(days)))
    })
    
    return df

# Linear regression model for hours vs productivity
def analyze_linear_relationship(df):
    X = df[['hours_worked']]
    y = df['productivity']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Linear Regression Results:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"Coefficient: {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5)
    
    # Plot regression line
    x_line = np.array([df['hours_worked'].min(), df['hours_worked'].max()]).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, 'r-', linewidth=2)
    
    plt.xlabel('Hours Worked')
    plt.ylabel('Productivity')
    plt.title('Linear Relationship: Hours Worked vs Productivity')
    plt.savefig('linear_relationship.png')
    plt.close()
    
    return model, mse, r2

# Polynomial regression for better modeling of productivity
def analyze_polynomial_relationship(df, degree=2):
    X = df[['hours_worked']]
    y = df['productivity']
    
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nPolynomial Regression (degree={degree}) Results:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5)
    
    # Plot polynomial curve
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    x_line_poly = poly.transform(x_line)
    y_line = model.predict(x_line_poly)
    plt.plot(x_line, y_line, 'r-', linewidth=2)
    
    plt.xlabel('Hours Worked')
    plt.ylabel('Productivity')
    plt.title(f'Polynomial Relationship (degree={degree}): Hours Worked vs Productivity')
    plt.savefig(f'polynomial_relationship_degree{degree}.png')
    plt.close()
    
    return model, mse, r2

# Analyze productivity by day of week
def analyze_day_of_week(df):
    day_productivity = df.groupby('day_of_week')['productivity'].mean()
    day_hours = df.groupby('day_of_week')['hours_worked'].mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    x = np.arange(len(days))
    
    ax1.bar(x, day_productivity)
    ax1.set_xticks(x)
    ax1.set_xticklabels(days)
    ax1.set_ylabel('Average Productivity')
    ax1.set_title('Productivity by Day of Week')
    
    ax2.bar(x, day_hours)
    ax2.set_xticks(x)
    ax2.set_xticklabels(days)
    ax2.set_ylabel('Average Hours Worked')
    ax2.set_title('Hours Worked by Day of Week')
    
    plt.tight_layout()
    plt.savefig('day_of_week_analysis.png')
    plt.close()

def main():
    print("Generating sample work hours data...")
    df = generate_sample_data()
    
    print("\nAnalyzing linear relationship between hours worked and productivity...")
    linear_model, linear_mse, linear_r2 = analyze_linear_relationship(df)
    
    print("\nAnalyzing polynomial relationship between hours worked and productivity...")
    poly_model, poly_mse, poly_r2 = analyze_polynomial_relationship(df, degree=2)
    
    print("\nAnalyzing productivity by day of week...")
    analyze_day_of_week(df)
    
    print("\nAnalysis complete. Visualizations saved as PNG files.")

if __name__ == "__main__":
    main() 