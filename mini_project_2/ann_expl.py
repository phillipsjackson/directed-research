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
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

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
ann = ANN(input_size=4, hidden_size=4, output_size=3, eta=0.05, n_iter=1000)
ann.fit(train_data, train_labels)
print("done")

test_preds = ann.predict(test_data)
print(test_preds)
test_labels_indices = np.argmax(test_labels, axis=1)
print(test_labels_indices)
test_accuracy = np.mean(test_preds == test_labels_indices)
print("Accuracy on test data: ", round(test_accuracy * 100, 2), "%")

ann.show_weight()

print("Create and fit ANN: ")

# --- Replacement: collect accuracy results and plot them ---
n_iter_values = [10,100,500,1000,1500,2500,5000,10000]
accuracies_pct = []

for ni in n_iter_values:
    ann = ANN(input_size=4, hidden_size=4, output_size=3, eta=0.05, n_iter=ni)
    ann.fit(train_data, train_labels)
    print(f"done (n_iter={ni})")

    test_preds = ann.predict(test_data)
    test_labels_indices = np.argmax(test_labels, axis=1)
    acc = np.mean(test_preds == test_labels_indices)
    accuracies_pct.append(acc * 100.0)  # store percent
    print("Test predictions:", test_preds)
    print("True labels:", test_labels_indices)
    print("Accuracy on test data: ", round(acc * 100, 2), "%")

# Plot results
plt.figure(figsize=(8,5))
plt.plot(n_iter_values, accuracies_pct, marker='o', linestyle='-')
plt.xscale('log')  # optional: makes wide range easier to read
plt.xticks(n_iter_values, labels=[str(x) for x in n_iter_values], rotation=45)
plt.xlabel('Number of training iterations (n_iter)')
plt.ylabel('Test accuracy (%)')
plt.title('ANN test accuracy vs n_iter')
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_vs_niter.png', dpi=150)
plt.show()


ann.show_weight()
