import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

tips = sns.load_dataset('tips')
print(tips.head())
print(tips.describe())
print(tips.nunique())
print(tips['size'].unique())


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the Tips dataset
tips = sns.load_dataset("tips")

# Display the first few rows of the dataframe
print(tips.head())

# Check for skewness in 'tip' column
plt.hist(tips['tip'], bins=20)
plt.title('Distribution of Tips')
plt.show()

tips_encoded = pd.get_dummies(tips, columns=['sex', 'smoker', 'day', 'time'])

# Select features and target
X = tips[['total_bill']]
y = tips['tip']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)


# Predict on test set
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R2 Score: {r2}")



from sklearn.linear_model import QuantileRegressor

# Initialize the Quantile Regression model
quantile_model = QuantileRegressor(alpha=0.5)  # Choosing median (alpha=0.5)

# Train the model
quantile_model.fit(X_train, y_train)

# Predict on the test set
y_pred_quantile = quantile_model.predict(X_test)

# Evaluate the model
#quantile_loss = mean_absolute_error(y_test, y_pred_quantile)
#print("Mean Absolute Error (Quantile Loss):", quantile_loss)
mse = mean_squared_error(y_test, y_pred_quantile)
rmse = mse**0.5
mae = mean_absolute_error(y_test, y_pred_quantile)
r2 = r2_score(y_test, y_pred_quantile)

print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R2 Score: {r2}")

from sklearn.linear_model import HuberRegressor

# Initialize the Huber Regression model
huber_model = HuberRegressor()

# Train the model
huber_model.fit(X_train, y_train)

# Predict on the test set
y_pred_huber = huber_model.predict(X_test)

# Evaluate the model
#huber_loss = mean_absolute_error(y_test, y_pred_huber)
#print("Mean Absolute Error (Huber Loss):", huber_loss)
mse = mean_squared_error(y_test, y_pred_huber)
rmse = mse**0.5
mae = mean_absolute_error(y_test, y_pred_huber)
r2 = r2_score(y_test, y_pred_huber)

print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R2 Score: {r2}")