import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Fetch the California housing dataset
# This dataset contains data from the 1990 California census.
# It includes features like MedInc (median income), HouseAge (median house age), AveRooms (average number of rooms), etc.
# The target variable is the median house value for California districts, expressed in hundreds of thousands of dollars.
california = fetch_california_housing()

# Selecting a single feature for prediction: the average number of rooms per household
# This simplifies the example to focus on the effect of using linear vs. polynomial regression.
X = california.data[:, np.newaxis, 3]  # Extracts the AveRooms feature as an independent variable
y = california.target  # Median house value (in $100,000s) as the dependent variable

# Split the dataset into training (80%) and testing (20%) sets
# This is a common practice to evaluate the performance of machine learning models on unseen data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
# Assumes a linear relationship between the average number of rooms and the median house value.
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)  # Train the model using the training data
y_pred_linear = lin_reg.predict(X_test)  # Make predictions on the test data

# Polynomial Regression Model
# Enhances the model by considering not just the linear term, but also the square (degree 2) of the number of rooms.
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly_features.fit_transform(X_train)  # Transforms the training data to include polynomial features
X_poly_test = poly_features.transform(X_test)  # Transforms the test data similarly

poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_train)  # Train the model on the polynomial-transformed training data
y_pred_poly = poly_reg.predict(X_poly_test)  # Make predictions on the polynomial-transformed test data


# Predicting the median house value for a hypothetical household with an average of 5 rooms
new_input = [[5]]  # Example input for prediction
linear_pred = lin_reg.predict(new_input)  # Prediction using linear regression
poly_pred = poly_reg.predict(poly_features.transform(new_input))  # Prediction using polynomial regression

# Print predictions
print(f"Linear Regression Prediction for input {new_input[0][0]} avg rooms: ${linear_pred[0]*1000:.2f}")
print(f"Polynomial Regression Prediction for input {new_input[0][0]} avg rooms: ${poly_pred[0]*1000:.2f}")