import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        """
        lstm_output: (batch, seq_len, hidden_dim)
        """
        attention_scores = self.attn_weights(lstm_output).squeeze(-1)  # (batch, seq_len)
        attention_weights = torch.softmax(attention_scores, dim=1)  # Normalize over sequence length
        context_vector = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)  # Weighted sum

        return context_vector, attention_weights

class CustomLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Weight matrices for input, forget, cell, and output gates
        self.W_i = nn.Linear(input_dim + hidden_dim, hidden_dim)  # Input gate
        self.W_f = nn.Linear(input_dim + hidden_dim, hidden_dim)  # Forget gate
        self.W_c = nn.Linear(input_dim + hidden_dim, hidden_dim)  # Cell gate
        self.W_o = nn.Linear(input_dim + hidden_dim, hidden_dim)  # Output gate

    def forward(self, x, h_prev, c_prev):
        """
        x: (batch, input_dim)    -> Current input
        h_prev: (batch, hidden_dim)  -> Previous hidden state
        c_prev: (batch, hidden_dim)  -> Previous cell state
        """
        combined = torch.cat((x, h_prev), dim=1)  # Concatenate input and hidden state

        i = torch.sigmoid(self.W_i(combined))  # Input gate
        f = torch.sigmoid(self.W_f(combined))  # Forget gate
        o = torch.sigmoid(self.W_o(combined))  # Output gate
        c_tilde = torch.tanh(self.W_c(combined))  # Candidate cell state

        c_next = f * c_prev + i * c_tilde  # New cell state
        h_next = o * torch.tanh(c_next)  # New hidden state

        return h_next, c_next
    
class CustomLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(CustomLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Create multiple LSTM layers
        self.lstm_layers = nn.ModuleList([CustomLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) 
                                          for i in range(num_layers)])

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)  -> Input sequence
        """
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)  # Hidden states
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)  # Cell states

        outputs = []
        for t in range(seq_len):  # Process sequence step by step
            x_t = x[:, t, :]  # Extract time step
            
            for layer in range(self.num_layers):  # Pass through layers
                h[layer], c[layer] = self.lstm_layers[layer](x_t, h[layer], c[layer])
                x_t = h[layer]  # Pass hidden state to next layer

            outputs.append(h[-1])  # Collect last layer's output

        return torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_dim)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attention = SelfAttention(hidden_dim * 2)  # Bidirectional doubles hidden size
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, text_input):
        embedded = self.embedding(text_input)  # (batch, seq_len, embedding_dim)
        lstm_output, _ = self.lstm(embedded)  # (batch, seq_len, hidden_dim*2)
        context_vector, attn_weights = self.attention(lstm_output)  # (batch, hidden_dim*2)
        output = self.fc(context_vector)  # (batch, output_dim)
        return output


batch_size = 4
seq_len = 10
input_dim = 32
hidden_dim = 64
num_layers = 2

model = CustomLSTM(input_dim, hidden_dim, num_layers)

x = torch.randn(batch_size, seq_len, input_dim)  # Random sequence input
output = model(x)  # Forward pass

print(output.shape)  # Expected: (batch_size, seq_len, hidden_dim)
