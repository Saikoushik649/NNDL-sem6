 1A. Without Package Using Numpy

import numpy as np

X = np.array([1, 2, 3, 4, 5])   # Independent variable
y = np.array([2, 4, 6, 8, 10]) # Dependent variable

m = 0   # slope
c = 0   # intercept
lr = 0.01  # learning rate
epochs = 1000

n = len(X)
for _ in range(epochs):
    y_pred = m * X + c
    dm = (-2/n) * np.sum(X * (y - y_pred))
    dc = (-2/n) * np.sum(y - y_pred)
    m = m - lr * dm
    c = c - lr * dc
X_test = 6
y_test_pred = m * X_test + c
print("Slope (m):", m)
print("Intercept (c):", c)
print("Predicted value for X = 6:", y_test_pred)


 1B. With TensorFlow

import tensorflow as tf
import numpy as np
X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([2, 4, 6, 8, 10], dtype=float)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])  # Single neuron for linear regression
])
model.compile(
    optimizer='sgd',        # Stochastic Gradient Descent
    loss='mean_squared_error'
)
model.fit(X, Y, epochs=500, verbose=0)
X_test = np.array([6.0])
prediction = model.predict(X_test)
print("Predicted value for X = 6 is:", prediction[0][0])
