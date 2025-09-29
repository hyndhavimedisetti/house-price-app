from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
X, y = load_boston(return_X_y=True)

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model as a proper binary pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("model.pkl created successfully!")
