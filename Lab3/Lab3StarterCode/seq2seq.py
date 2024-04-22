import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(output_size, hidden_size)
        self.output_size = output_size
        
    def forward(self, input_seq, target_seq_len):
        hidden, cell = self.encoder(input_seq)
        # Start with an initial input (usually zero) to the decoder
        decoder_input = torch.zeros((input_seq.size(0), 1, self.output_size), device=input_seq.device)
        outputs = []
        
        for _ in range(target_seq_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(output)
            decoder_input = output.unsqueeze(1)
        
        return torch.cat(outputs, dim=1)

# Parameters
input_size = 1  # Size of each input item
output_size = 1  # Size of each output item
hidden_size = 128  # Number of features in the hidden state

# Model initialization
model = Seq2Seq(input_size, output_size, hidden_size)

# Example input
input_seq = torch.randn(1, 10, input_size)  # Batch size of 1, sequence length of 10
print(input_seq.size())

# Forward pass
output_seq = model(input_seq, 5)  # We want to generate a sequence of length 5
print(output_seq)
