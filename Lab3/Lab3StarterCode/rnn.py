import torch
from torch import nn
from torch.utils.data import DataLoader

# Sample data generation (replace with your actual data)
def generate_data(seq_len):
  data = torch.randn(seq_len)  # Random data with sequence length
  return data

# Define the RNN model (LSTM in this case)
class RNNModel(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(RNNModel, self).__init__()
    self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    # Initialize hidden and cell states for batch size 1
    h0 = torch.zeros(1, 1, self.lstm.hidden_size).to(x.device)
    c0 = torch.zeros(1, 1, self.lstm.hidden_size).to(x.device)
    
    # Forward pass through LSTM
    out, _ = self.lstm(x, (h0, c0))
    
    # Use the output of the last time step
    out = self.fc(out[:, -1, :])
    return out

# Hyperparameters
input_dim = 1  # Feature size
hidden_dim = 16  # Number of hidden units
output_dim = 1  # Output size (assuming prediction of next value)
seq_len = 10  # Sequence length

model = RNNModel(input_dim, hidden_dim, output_dim)

# Generate sample data
data = generate_data(seq_len)

# Reshape data to have a batch dimension of 1
data = data.unsqueeze(0)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean squared error for regression task
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(100):  # Adjust number of epochs as needed
  # Forward pass
  output = model(data)

  # Calculate loss
  loss = criterion(output, data[:, 1:])  # Use all except first element as target

  # Backward pass and update weights
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  # Print loss (optional)
  if epoch % 10 == 0:
    print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')
