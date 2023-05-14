import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


remote_data = "https://raw.githubusercontent.com/ageron/data/main/lifesat/lifesat.csv"
lifesat = pd.read_csv(remote_data)

X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]]


lifesat.plot(
    kind="scatter",
    grid=True,
    x="GDP per capita (USD)",
    y="Life satisfaction"
)

plt.axis([23_500, 62_500, 4, 9])
#plt.show()

model = LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[37_655.2]]
y_new = model.predict(X_new)
print(f"Estimated life satisfaction index using linear regression: {y_new}")
