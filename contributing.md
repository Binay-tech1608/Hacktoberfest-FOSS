This is my contribution for the hacktoberfest 2023
//I am very interested in web development and ML/AI algorithms. This is also my strongest part in career. so please give me a chance to contribute to the project. 
#The given program is for the prediction of the stock prices. 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load historical stock price data from a CSV file
# Replace 'stock_data.csv' with your actual data file
data = pd.read_csv('stock_data.csv')

# Extract the 'Date' and 'Price' columns
dates = data['Date'].values.reshape(-1, 1)
prices = data['Price'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dates, prices, test_size=0.2, random_state=0)

# Create a linear regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Visualize the results
plt.scatter(X_test, y_test, color='b', label='Actual Prices')
plt.plot(X_test, y_pred, color='r', label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.title('Stock Price Prediction')
plt.show()

# Make a prediction for a future date
future_date = np.array([[20231025]])  # Replace with your desired date
predicted_price = model.predict(future_date)
print(f'Predicted stock price on 20231025 is: {predicted_price[0]:.2f}')
