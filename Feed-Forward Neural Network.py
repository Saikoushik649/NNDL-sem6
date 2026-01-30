import numpy as np

# Input data
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Output labels
y = np.array([[0],[1],[1],[0]])

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize weights
np.random.seed(1)
W1 = np.random.randn(2, 3)
W2 = np.random.randn(3, 1)

# Training
for _ in range(10000):
    # Forward pass
    h = sigmoid(np.dot(X, W1))
    y_pred = sigmoid(np.dot(h, W2))

    # Backpropagation
    error = y - y_pred
    d2 = error * y_pred * (1 - y_pred)
    d1 = d2.dot(W2.T) * h * (1 - h)

    # Update weights
    W2 += h.T.dot(d2) * 0.1
    W1 += X.T.dot(d1) * 0.1

# Output
print("Predicted Output:")
print(np.round(y_pred))
