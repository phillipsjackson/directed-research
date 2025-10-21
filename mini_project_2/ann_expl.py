import numpy as np

class ANN(object):
    def __init__(self, input_size, hidden_size, output_size, eta=0.01, n_iter=100):
        self.eta = eta
        self.n_iter = n_iter
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        self.W1 = np.random.rand(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.rand(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def show_weight(self):
        print(self.W1)
        print(self.W2)

    def fit(self, X, y):
        for i in range(self.n_iter):
            # Forward pass
            hidden_input = np.dot(X, self.W1) + self.b1
            hidden_output = self.sigmoid(hidden_input)

            final_input = np.dot(hidden_output, self.W2) + self.b2
            final_output = self.sigmoid(final_input)

            # Calculate the error
            error = y - final_output

            # Backward pass
            d_final_output = error * self.sigmoid_derivative(final_output)
            error_hidden_layer = d_final_output.dot(self.W2.T)
            d_hidden_output = error_hidden_layer * self.sigmoid_derivative(hidden_output)

            # Update weights and biases
            self.W2 += hidden_output.T.dot(d_final_output) * self.eta
            self.b2 += np.sum(d_final_output, axis=0, keepdims=True) * self.eta
            self.W1 += X.T.dot(d_hidden_output) * self.eta
            self.b1 += np.sum(d_hidden_output, axis=0, keepdims=True) * self.eta

    def predict(self, X):
        hidden_input = np.dot(X, self.W1) + self.b1
        hidden_output = self.sigmoid(hidden_input)

        final_input = np.dot(hidden_output, self.W2) + self.b2
        final_output = self.sigmoid(final_input)

        predictions = np.argmax(final_output, axis=1)
        return predictions

import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Assuming the dataset has been loaded and preprocessed as in your initial code
df = pd.read_csv('iris_dataset.csv')
df = shuffle(df)
X = df.iloc[:, 0:4].values
y = df[['target']].values
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

train_data, test_data, train_labels, test_labels = train_test_split(
                                    X, y, test_size=0.25)

print("Create and fit ANN: ")
ann = ANN(input_size=4, hidden_size=4, output_size=3, eta=0.1, n_iter=1000)
ann.fit(train_data, train_labels)
print("done")

test_preds = ann.predict(test_data)
print(test_preds)
test_labels_indices = np.argmax(test_labels, axis=1)
print(test_labels_indices)
test_accuracy = np.mean(test_preds == test_labels_indices)
print("Accuracy on test data: ", round(test_accuracy * 100, 2), "%")

ann.show_weight()
