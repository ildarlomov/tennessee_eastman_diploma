import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """
    What is this for??
    This simply truncates chomp_size dots from the end of the sequence
    to make it the original length after Conv1d with
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU(0.1)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):  # [2, 1, 500]
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
 
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
          self.linear.weight.data.normal_(0, 0.01)
    
    def forward(self, x, channel_last=True):
        # If channel_last, the expected format is (batch_size, seq_len, features)
        y1 = self.tcn(x.transpose(1, 2) if channel_last else x)
        return self.linear(y1.transpose(1, 2))


class CausalConvDiscriminatorMultitask(nn.Module):
    """Discriminator using casual dilated convolution, outputs a probability for each time step

    Args:
        input_size (int): dimensionality (channels) of the input
        n_layers (int): number of hidden layers
        n_channels (int): number of channels in the hidden layers (it's always the same)
        kernel_size (int): kernel size in all the layers
        dropout: (float in [0-1]): dropout rate

    Input: (batch_size, seq_len, input_size)
    Output: (batch_size, seq_len, 1)
    """

    def __init__(self, input_size, n_layers, n_channel, class_count, kernel_size, dropout=0):
        super().__init__()
        # Assuming same number of channels layerwise
        num_channels = [n_channel] * n_layers
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)

        self.fault_type_head_fc1 = nn.Linear(num_channels[-1], 64)
        self.fault_type_head_fc2 = nn.Linear(64, 32)
        self.fault_type_head_fc3 = nn.Linear(32, class_count)

        self.real_fake_head_fc1 = nn.Linear(num_channels[-1], 64)
        self.real_fake_head_fc2 = nn.Linear(64, 1)

        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x, _, channel_last=True):
        common = self.tcn(x.transpose(1, 2) if channel_last else x)
        type_logits = self.activation(self.fault_type_head_fc1(common))
        type_logits = self.activation(self.fault_type_head_fc2(type_logits))
        type_logits = self.fault_type_head_fc3(type_logits)

        real_fake_logits = self.activation(self.real_fake_head_fc1(common))
        real_fake_logits = self.activation(self.real_fake_head_fc2(real_fake_logits))

        return type_logits, real_fake_logits

    def zero_state(self, _):
        # just to make it compatible with LSTM architecture
        return torch.randn(1), torch.randn(1)


class CausalConvDiscriminator(nn.Module):
    """Discriminator using casual dilated convolution, outputs a probability for each time step

    Args:
        input_size (int): dimensionality (channels) of the input
        n_layers (int): number of hidden layers
        n_channels (int): number of channels in the hidden layers (it's always the same)
        kernel_size (int): kernel size in all the layers
        dropout: (float in [0-1]): dropout rate
        
    Input: (batch_size, seq_len, input_size)
    Output: (batch_size, seq_len, 1)
    """
    def __init__(self, input_size, n_layers, n_channel, kernel_size, dropout=0):
        super().__init__()
        #Assuming same number of channels layerwise
        num_channels = [n_channel] * n_layers
        self.tcn = TCN(input_size, 1, num_channels, kernel_size, dropout)
        
    def forward(self, x, channel_last=True):
        return torch.sigmoid(self.tcn(x, channel_last))


class CausalConvGenerator(nn.Module):
    """Generator using casual dilated convolution, expecting a noise vector for each timestep as input

    Args:
        noise_size (int): dimensionality (channels) of the input noise
        output_size (int): dimenstionality (channels) of the output sequence
        n_layers (int): number of hidden layers
        n_channels (int): number of channels in the hidden layers (it's always the same)
        kernel_size (int): kernel size in all the layers
        dropout: (float in [0-1]): dropout rate
        
    Input: (batch_size, seq_len, input_size)
    Output: (batch_size, seq_len, outputsize)
    """ 
    def __init__(self, noise_size, output_size, n_layers, n_channel, kernel_size, dropout=0):
        super().__init__()
        num_channels = [n_channel] * n_layers
        self.tcn = TCN(noise_size, output_size, num_channels, kernel_size, dropout)
        
    def forward(self, x, _, channel_last=True):
        return torch.tanh(self.tcn(x, channel_last))

    def zero_state(self, _):
        # just to make it compatible with LSTM architecture
        return torch.randn(1), torch.randn(1)


class CNN1D2D(nn.Module):
    """
    CNN for TEP dataset.
    It combines stacked 1d and 2d convolution layers in order to capture the delays in sensor data.
    """

    def __init__(self, class_count):
        super(CNN1D2D, self).__init__()
        # todo: experimenting with groups parameter is an interesting thing
        self.conv1 = nn.Conv1d(in_channels=52, out_channels=52*2, kernel_size=5, groups=52)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(in_channels=52*2, out_channels=52*2*2, kernel_size=5, groups=52*2)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.conv3 = nn.Conv2d(1, 6, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 103, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, class_count)

    def forward(self, x):
        x = x.squeeze()
        x = x.transpose(1, 2)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 1, 52*2*2, 4)
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 6*103)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    #30-dimensional noise
    input = torch.randn(8, 32, 30)

    gen = CausalConvGenerator(noise_size=30, output_size=1, n_layers=8, n_channel=10, kernel_size=8, dropout=0)
    dis = CausalConvDiscriminator(input_size=1, n_layers=8, n_channel=10, kernel_size=8, dropout=0)

    print("Input shape:", input.size())
    fake = gen(input)
    print("Generator output shape:", fake.size())
    dis_out = dis(fake)
    print("Discriminator output shape:", dis_out.size())
