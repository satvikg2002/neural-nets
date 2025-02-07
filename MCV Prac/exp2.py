import numpy as np


class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(
            self.input_size, self.hidden_size)
        self.bias_input_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(
            self.hidden_size, self.output_size)
        self.bias_hidden_output = np.zeros((1, self.output_size))
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Forward pass
        self.hidden_input = np.dot(
            X, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = np.dot(
            self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output
        return self.output

    def backward(self, X, y, output):
        # Backpropagation
        self.output_error = y - output
        self.output_delta = self.output_error
        self.hidden_error = self.output_delta.dot(self.weights_hidden_output.T)
        self.hidden_delta = self.hidden_error * \
            self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(
            self.output_delta) * self.learning_rate
        self.bias_hidden_output += np.sum(self.output_delta,
                                          axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += X.T.dot(self.hidden_delta) * \
            self.learning_rate
        self.bias_input_hidden += np.sum(self.hidden_delta,
                                         axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f'Epoch {epoch}, Loss: {loss}')


# Generate synthetic dataset
np.random.seed(0)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train MLP
mlp = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
mlp.train(X, y, epochs=10000)

# Test the trained model
print("Test Predictions:")
print(mlp.forward(X))
