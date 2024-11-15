import torch
from torch import nn
import argparse
import torch.optim as optim

# --- BayesianLSTM models and distributions ---

class BayesianModule(nn.Module):
    def __init__(self):
        super(BayesianModule, self).__init__()

    def forward(self, *inputs):
        raise NotImplementedError

class TrainableRandomDistribution:
    def __init__(self, mu, rho):
        self.mu = mu
        self.rho = rho

    def sample(self):
        epsilon = torch.randn_like(self.mu)
        return self.mu + self.rho.exp() * epsilon

    def log_posterior(self):
        return -0.5 * torch.sum((self.mu / self.rho.exp())**2 + 2 * torch.log(self.rho.exp()))

class PriorWeightDistribution:
    def __init__(self, pi, sigma1, sigma2):
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def log_prior(self, x):
        prior1 = torch.exp(-0.5 * x**2 / self.sigma1**2) / (self.sigma1 * (2 * torch.pi)**0.5)
        prior2 = torch.exp(-0.5 * x**2 / self.sigma2**2) / (self.sigma2 * (2 * torch.pi)**0.5)
        return torch.log(self.pi * prior1 + (1 - self.pi) * prior2)

class BayesianLSTM(BayesianModule):
    def __init__(self, in_features, out_features, bias=True, posterior_mu_init=0, posterior_rho_init=-6.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Input weight parameters
        self.weight_ih_mu = nn.Parameter(torch.Tensor(in_features, out_features * 4).normal_(posterior_mu_init, 0.1))
        self.weight_ih_rho = nn.Parameter(torch.Tensor(in_features, out_features * 4).normal_(posterior_rho_init, 0.1))
        self.weight_ih_sampler = TrainableRandomDistribution(self.weight_ih_mu, self.weight_ih_rho)
        
        # Hidden state weight parameters
        self.weight_hh_mu = nn.Parameter(torch.Tensor(out_features, out_features * 4).normal_(posterior_mu_init, 0.1))
        self.weight_hh_rho = nn.Parameter(torch.Tensor(out_features, out_features * 4).normal_(posterior_rho_init, 0.1))
        self.weight_hh_sampler = TrainableRandomDistribution(self.weight_hh_mu, self.weight_hh_rho)
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features * 4).normal_(posterior_mu_init, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features * 4).normal_(posterior_rho_init, 0.1))
        self.bias_sampler = TrainableRandomDistribution(self.bias_mu, self.bias_rho)
        
        # Prior distribution
        self.weight_ih_prior_dist = PriorWeightDistribution(1, 0.1, 0.002)
        self.weight_hh_prior_dist = PriorWeightDistribution(1, 0.1, 0.002)
        self.bias_prior_dist = PriorWeightDistribution(1, 0.1, 0.002)
    
    def sample_weights(self):
        # Sample weights from the defined distributions
        weight_ih = self.weight_ih_sampler.sample()
        weight_hh = self.weight_hh_sampler.sample()
        bias = self.bias_sampler.sample() if self.use_bias else None
        return weight_ih, weight_hh, bias

    def forward(self, x, hidden_states=None):
        weight_ih, weight_hh, bias = self.sample_weights()
        bs, seq_sz, _ = x.size()
        HS = self.out_features
        
        if hidden_states is None:
            h_t, c_t = (torch.zeros(bs, HS).to(x.device), torch.zeros(bs, HS).to(x.device))
        else:
            h_t, c_t = hidden_states

        hidden_seq = []
        
        for t in range(seq_sz):
            x_t = x[:, t, :]
            gates = x_t @ weight_ih + h_t @ weight_hh + bias
            i_t, f_t, o_t = (
                torch.sigmoid(gates[:, :HS]), 
                torch.sigmoid(gates[:, HS:HS*2]), 
                torch.sigmoid(gates[:, HS*3:])
            )
            c_t = f_t * c_t + i_t * torch.tanh(gates[:, HS*2:HS*3])
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        
        hidden_seq = torch.cat(hidden_seq, dim=0).transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

class MyBayesianModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyBayesianModel, self).__init__()
        self.bayesian_lstm = BayesianLSTM(in_features=input_size, out_features=hidden_size, bias=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden_states=None):
        lstm_out, _ = self.bayesian_lstm(x, hidden_states)
        out = self.fc(lstm_out[:, -1, :])
        return out

# --- Argument parsing and model training ---

def train_model(args):
    # Create model instance
    model = MyBayesianModel(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size)
    
    # Choose optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()  # Use MSELoss for regression, or CrossEntropy for classification
    
    # Random data for testing purposes
    x_train = torch.randn(args.batch_size, args.seq_len, args.input_size)
    y_train = torch.randn(args.batch_size, args.output_size)
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Bayesian LSTM model")
    
    # Adding arguments
    parser.add_argument('--input_size', type=int, default=10, help='Number of input features')
    parser.add_argument('--hidden_size', type=int, default=20, help='Number of hidden units in the LSTM')
    parser.add_argument('--output_size', type=int, default=1, help='Number of output features')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--seq_len', type=int, default=15, help='Sequence length for input data')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')

    # Additional parameters to match the command you gave
    parser.add_argument('--data', type=str, required=True, help='Path to the training data')
    parser.add_argument('--scratch', type=str, required=True, help='Path to the scratch data')
    parser.add_argument('--dataset', type=str, default='PTB', help='Dataset name')
    parser.add_argument('--scheduler', type=str, default='cosine', help='Scheduler type')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay rate')
    parser.add_argument('--dropout_emb', type=float, default=0.6, help='Dropout for embedding layer')
    parser.add_argument('--dropout_forward', type=float, default=0.25, help='Dropout for forward pass')
    parser.add_argument('--grad_clip', type=float, default=0.1, help='Gradient clipping')

    args = parser.parse_args()
    
    # Execute training process
    train_model(args)
