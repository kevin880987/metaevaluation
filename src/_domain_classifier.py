

import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss



class Network(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_layers)
        self.linear2 = nn.Linear(hidden_layers, output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x

net = Network(X.shape[1], 25, Y.nunique())

def train(X, Y):
    n_epochs = 2
    batch_size = 100
    learning_rate = 0.01
    n_samples = X.shape[0]
    device = 'cpu'
    for epoch in range(n_epochs):
        epoch_loss = 0.0

        # Index for batches
        permutation = np.arange(0, n_samples)
        np.random.shuffle(permutation)
        if n_samples>batch_size:
            permutations = permutation[:-(n_samples%batch_size)]
            permutations = permutations.reshape((-1, batch_size)).tolist()
            permutations.append(permutation[-(n_samples%batch_size):])
        else:
            permutations = [permutation]
        
        for i, idx in enumerate(permutations, 0):
            input, target = X.astype(float).values[idx], Y.astype(float).values[idx]
            input = torch.from_numpy(input).type(torch.FloatTensor).to(device)
            target = torch.from_numpy(target).type(torch.FloatTensor).to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
            # set optimizer to zero grad to remove previous epoch gradients
            optimizer.zero_grad()
            # forward propagation
            output = net(input)
            # _, pred = torch.max(output, dim=1)
            loss = CrossEntropyLoss()
            loss(output, target)
            # backward propagation
            loss.backward()
            # optimize
            optimizer.step()
            epoch_loss += loss.item()
        # display statistics
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')