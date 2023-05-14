import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


remote_data = "https://raw.githubusercontent.com/ageron/data/main/lifesat/lifesat.csv"
lifesat = pd.read_csv(remote_data)

X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]]

model = KNeighborsRegressor(n_neighbors=3)

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[37_655.2]]
y_new = model.predict(X_new)
print(f"Estimated life satisfaction index using k-nn regression: {y_new}")
