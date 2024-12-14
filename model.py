import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Attempt to download historical stock price data
try:
    data = yf.download('AAPL', start='2022-01-01', end='2023-01-01')
    data['Price'] = data['Adj Close']
except Exception as e:
    print("Data download failed, generating simulated data instead.")
    # Generate simulated data
    date_range = pd.date_range(start="2022-01-01", end="2023-01-01", freq='D')
    data = pd.DataFrame({
        'Date': date_range,
        'Price': 100 + np.cumsum(np.random.randn(len(date_range)) * 2)  # Simulated price path
    })
    data.set_index('Date', inplace=True)

# Check data
print(data.head())

# Calculate first and second derivatives
data.loc[:, 'Velocity'] = data['Price'].diff()
data.loc[:, 'Acceleration'] = data['Velocity'].diff()

# Drop rows with NaN values
data.dropna(inplace=True)

# Confirm there's enough data for splitting
if len(data) > 1:
    # Prepare features and target
    X = data[['Velocity', 'Acceleration']]
    y = np.where(data['Price'].shift(-1) > data['Price'], 1, 0)  # Binary classification: next move up/down

    # Remove the last row as it has NaN in target due to shift(-1)
    X = X[:-1]
    y = y[:-1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check if there's enough data after split
    if len(X_train) > 0 and len(X_test) > 0:
        # Initialize and train the model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate and print accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print("Model Accuracy:", accuracy)
    else:
        print("Not enough data after split. Please check the dataset size.")
else:
    print("Insufficient data available for analysis.")
