# Adapted from: https://github.com/DiTEC-project/semantic-association-rule-learning

import torch
import os
from torch import nn


class AutoEncoder(nn.Module):
    """
    This autoencoder is used to create a numerical representation for the categorical values.
    """

    def __init__(self, data_size):
        """
        :param data_size: size of the categorical features in the knowledge graph, after one-hot encoding
        """
        super().__init__()
        self.data_size = data_size
        print("Data size: ", data_size)
        self.encoder = nn.Sequential(
            nn.Linear(self.data_size, int(1 * self.data_size / 2)),
            nn.Tanh(),
            nn.Linear(int(1 * self.data_size / 2), int(1 * self.data_size / 4)),
            nn.Tanh(),
            nn.Linear(int(1 * self.data_size / 4), int(1 * self.data_size / 16)),
            nn.Tanh(),
            nn.Linear(int(1 * self.data_size / 16), int(1 * self.data_size / 32)),
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(1 * self.data_size / 32), int(1 * self.data_size / 16)),
            nn.Tanh(),
            nn.Linear(int(1 * self.data_size / 16), int(1 * self.data_size / 4)),
            nn.Tanh(),
            nn.Linear(int(1 * self.data_size / 4), int(1 * self.data_size / 2)),
            nn.Tanh(),
            nn.Linear(int(1 * self.data_size / 2), self.data_size)
        )

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)
        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(p=0.2)

    @staticmethod
    def init_weights(m):
        """
        all weights are initialized with values sampled from uniform distributions with the Xavier initialization
        and the biases are set to 0, as described in the paper by Delong et al. (2023)
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.zero_()

    def save(self, p):
        torch.save(self.encoder.state_dict(), p + 'cat_encoder.pt')
        torch.save(self.decoder.state_dict(), p + 'cat_decoder.pt')

    def load(self, p):
        if os.path.isfile(p + 'cat_encoder.pt') and os.path.isfile(p + 'cat_decoder.pt'):
            self.encoder.load_state_dict(torch.load(p + 'cat_encoder.pt'))
            self.decoder.load_state_dict(torch.load(p + 'cat_decoder.pt'))
            self.encoder.eval()
            self.decoder.eval()
            return True
        else:
            return False

    def forward(self, x, input_vector_category_indices):
        y = self.encoder(x)
        y = self.decoder(y)

        for category_index in range(len(input_vector_category_indices)):
            category_range = input_vector_category_indices[category_index]
            y[category_range['start']:category_range['end']] = \
                self.softmax(y[category_range['start']:category_range['end']])

        return y
