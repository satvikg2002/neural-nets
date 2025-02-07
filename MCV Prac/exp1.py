import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

# Generate some random data
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Input feature
y = 2 * X + 1 + np.random.randn(100, 1)  # Output with noise

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully connected layer
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # ReLU activation function for the hidden layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


input_size = 1
hidden_size = 10
output_size = 1

model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()  # Mean Squared Error loss
# Stochastic Gradient Descent
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
