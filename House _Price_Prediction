import pandas as pd
from sklearn.linear_model import LinearRegression

# Training data: [Square Footage, Bedrooms, Bathrooms]
X = [[1500, 3, 2], [2000, 4, 3], [1200, 2, 1], [2500, 4, 3], [1800, 3, 2], [3000, 5, 4]]
y = [400000, 500000, 320000, 610000, 460000, 750000]

model = LinearRegression()
model.fit(X, y)

# Prediction
sample_house = [[2100, 3, 2]]
prediction = model.predict(sample_house)

print(f"Predicted Price for 2100sqft, 3BR, 2BA: ${prediction[0]:,.2f}")
