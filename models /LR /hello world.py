import torch
import torch.nn as nn

# 1. Define the Model Architecture
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # A linear layer: y = xW^T + b
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)

# 2. Initialize the model, loss function, and optimizer
model = SimpleNet()
criterion = nn.MSELoss() # Mean Squared Error
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # Stochastic Gradient Descent
