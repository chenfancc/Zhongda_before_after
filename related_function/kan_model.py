import torch
from kan import KANLayer
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        self.relu2 = nn.ReLU()

        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu2(out)

        return out


class BiLSTM_small_kan(nn.Module):
    from KAN.train_kan import layer_nums_small
    def __init__(self, Feature_number):
        super(BiLSTM_small_kan, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=64, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.kan = nn.ModuleList(
            [KANLayer(in_dim=self.layer_nums_small[i], out_dim=self.layer_nums_small[i + 1], device='cuda') for i in
             range(len(self.layer_nums_small) - 1)])

    def __str__(self):
        return 'BiLSTM_small_kan'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(4, x.size(0), 64).to(x.device)
        c0 = torch.zeros(4, x.size(0), 64).to(x.device)

        output, _ = self.lstm(x, (h0, c0))
        # print(output.shape)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        # print(output.shape)
        for layer in self.kan:
            output = layer(output)[0]
        output_sigmoid = torch.sigmoid(output)
        return output_sigmoid.squeeze(1).to(x.device)


class BiLSTM_large_kan(nn.Module):
    from KAN.train_kan import layer_nums_large
    def __init__(self, Feature_number):
        super(BiLSTM_large_kan, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=256, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.kan = nn.ModuleList(
            [KANLayer(in_dim=self.layer_nums_large[i], out_dim=self.layer_nums_large[i + 1], device='cuda') for i in
             range(len(self.layer_nums_large) - 1)])

    def __str__(self):
        return 'BiLSTM_large_kan'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(4, x.size(0), 256).to(x.device)
        c0 = torch.zeros(4, x.size(0), 256).to(x.device)

        output, _ = self.lstm(x, (h0, c0))
        # print(output.shape)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        # print(output.shape)
        for layer in self.kan:
            output = layer(output)[0]
        output_sigmoid = torch.sigmoid(output)
        return output_sigmoid.squeeze(1).to(x.device)


class BiLSTM_kan(nn.Module):
    from KAN.train_kan import layer_nums
    def __init__(self, Feature_number):
        super(BiLSTM_kan, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=128, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.kan = nn.ModuleList(
            [KANLayer(in_dim=self.layer_nums[i], out_dim=self.layer_nums[i + 1], device='cuda') for i in
             range(len(self.layer_nums) - 1)])

    def __str__(self):
        return 'BiLSTM_kan'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(4, x.size(0), 128).to(x.device)
        c0 = torch.zeros(4, x.size(0), 128).to(x.device)

        output, _ = self.lstm(x, (h0, c0))
        # print(output.shape)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        # print(output.shape)
        for layer in self.kan:
            output = layer(output)[0]
        output_sigmoid = torch.sigmoid(output)
        return output_sigmoid.squeeze(1).to(x.device)


class BiLSTM_3layers_kan(nn.Module):
    from KAN.train_kan import layer_nums
    def __init__(self, Feature_number):
        super(BiLSTM_3layers_kan, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=128, num_layers=3, batch_first=True,
                            bidirectional=True)
        self.kan = nn.ModuleList(
            [KANLayer(in_dim=self.layer_nums[i], out_dim=self.layer_nums[i + 1], device='cuda') for i in
             range(len(self.layer_nums) - 1)])

    def __str__(self):
        return 'BiLSTM_3layers_kan'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(6, x.size(0), 128).to(x.device)
        c0 = torch.zeros(6, x.size(0), 128).to(x.device)

        output, _ = self.lstm(x, (h0, c0))
        # print(output.shape)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        # print(output.shape)
        for layer in self.kan:
            output = layer(output)[0]
        output_sigmoid = torch.sigmoid(output)
        return output_sigmoid.squeeze(1).to(x.device)


class BiLSTM_4layers_kan(nn.Module):
    from KAN.train_kan import layer_nums
    def __init__(self, Feature_number):
        super(BiLSTM_4layers_kan, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=128, num_layers=8, batch_first=True,
                            bidirectional=True)
        self.kan = nn.ModuleList(
            [KANLayer(in_dim=self.layer_nums[i], out_dim=self.layer_nums[i + 1], device='cuda') for i in
             range(len(self.layer_nums) - 1)])

    def __str__(self):
        return 'BiLSTM_4layers_kan'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(8, x.size(0), 128).to(x.device)
        c0 = torch.zeros(8, x.size(0), 128).to(x.device)

        output, _ = self.lstm(x, (h0, c0))
        # print(output.shape)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        # print(output.shape)
        for layer in self.kan:
            output = layer(output)[0]
        output_sigmoid = torch.sigmoid(output)
        return output_sigmoid.squeeze(1).to(x.device)


class BiLSTM_Resnet_kan(nn.Module):
    from KAN.train_kan import layer_nums
    def __init__(self, Feature_number):
        super(BiLSTM_Resnet_kan, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=128, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.bn = nn.BatchNorm1d(256)

        self.resblock1 = ResBlock(256, 256)  # 添加残差块
        self.resblock2 = ResBlock(256, 256)  # 添加残差块
        self.resblock3 = ResBlock(256, 256)  # 添加残差块

        self.kan = nn.ModuleList(
            [KANLayer(in_dim=self.layer_nums[i], out_dim=self.layer_nums[i + 1], device='cuda') for i in
             range(len(self.layer_nums) - 1)])

    def __str__(self):
        return 'BiLSTM_Resnet_kan'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(4, x.size(0), 128).to(x.device)
        c0 = torch.zeros(4, x.size(0), 128).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output)

        for layer in self.kan:
            output = layer(output)[0]
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class GRU_kan(nn.Module):
    from KAN.train_kan import layer_nums_nobi
    def __init__(self, Feature_number):
        super(GRU_kan, self).__init__()
        self.gru = nn.GRU(Feature_number, hidden_size=128, num_layers=2, batch_first=True)
        self.kan = nn.ModuleList(
            [KANLayer(in_dim=self.layer_nums_nobi[i], out_dim=self.layer_nums_nobi[i + 1], device='cuda') for i in
             range(len(self.layer_nums_nobi) - 1)])

    def __str__(self):
        return 'GRU_kan'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(2, x.size(0), 128).to(x.device)

        output, _ = self.gru(x, h0)
        # print(output.shape)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        # print(output.shape)
        for layer in self.kan:
            output = layer(output)[0]
        output_sigmoid = torch.sigmoid(output)
        return output_sigmoid.squeeze(1).to(x.device)


class GRU_small_kan(nn.Module):
    from KAN.train_kan import layer_nums_small_nobi
    def __init__(self, Feature_number):
        super(GRU_small_kan, self).__init__()
        self.gru = nn.GRU(Feature_number, hidden_size=64, num_layers=2, batch_first=True)
        self.kan = nn.ModuleList(
            [KANLayer(in_dim=self.layer_nums_small_nobi[i], out_dim=self.layer_nums_small_nobi[i + 1], device='cuda') for i in
             range(len(self.layer_nums_small_nobi) - 1)])

    def __str__(self):
        return 'GRU_small_kan'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(2, x.size(0), 64).to(x.device)

        output, _ = self.gru(x, h0)
        # print(output.shape)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        # print(output.shape)
        for layer in self.kan:
            output = layer(output)[0]
        output_sigmoid = torch.sigmoid(output)
        return output_sigmoid.squeeze(1).to(x.device)


class GRU_large_kan(nn.Module):
    from KAN.train_kan import layer_nums_large_nobi
    def __init__(self, Feature_number):
        super(GRU_large_kan, self).__init__()
        self.gru = nn.GRU(Feature_number, hidden_size=256, num_layers=2, batch_first=True)
        self.kan = nn.ModuleList(
            [KANLayer(in_dim=self.layer_nums_large_nobi[i], out_dim=self.layer_nums_large_nobi[i + 1], device='cuda') for i in
             range(len(self.layer_nums_large_nobi) - 1)])

    def __str__(self):
        return 'GRU_large_kan'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(2, x.size(0), 256).to(x.device)

        output, _ = self.gru(x, h0)
        # print(output.shape)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        # print(output.shape)
        for layer in self.kan:
            output = layer(output)[0]
        output_sigmoid = torch.sigmoid(output)
        return output_sigmoid.squeeze(1).to(x.device)


class GRU_3layers_kan(nn.Module):
    from KAN.train_kan import layer_nums_nobi
    def __init__(self, Feature_number):
        super(GRU_3layers_kan, self).__init__()
        self.gru = nn.GRU(Feature_number, hidden_size=128, num_layers=3, batch_first=True)
        self.kan = nn.ModuleList(
            [KANLayer(in_dim=self.layer_nums_nobi[i], out_dim=self.layer_nums_nobi[i + 1], device='cuda') for i in
             range(len(self.layer_nums_nobi) - 1)])

    def __str__(self):
        return 'GRU_3layers_kan'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(3, x.size(0), 128).to(x.device)

        output, _ = self.gru(x, h0)
        # print(output.shape)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        # print(output.shape)
        for layer in self.kan:
            output = layer(output)[0]
        output_sigmoid = torch.sigmoid(output)
        return output_sigmoid.squeeze(1).to(x.device)


class GRU_4layers_kan(nn.Module):
    from KAN.train_kan import layer_nums_nobi
    def __init__(self, Feature_number):
        super(GRU_4layers_kan, self).__init__()
        self.gru = nn.GRU(Feature_number, hidden_size=128, num_layers=4, batch_first=True)
        self.kan = nn.ModuleList(
            [KANLayer(in_dim=self.layer_nums_nobi[i], out_dim=self.layer_nums_nobi[i + 1], device='cuda') for i in
             range(len(self.layer_nums_nobi) - 1)])

    def __str__(self):
        return 'GRU_4layers_kan'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(4, x.size(0), 128).to(x.device)

        output, _ = self.gru(x, h0)
        # print(output.shape)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        # print(output.shape)
        for layer in self.kan:
            output = layer(output)[0]
        output_sigmoid = torch.sigmoid(output)
        return output_sigmoid.squeeze(1).to(x.device)


class GRU_Resnet_kan(nn.Module):
    from KAN.train_kan import layer_nums_nobi
    def __init__(self, Feature_number):
        super(GRU_Resnet_kan, self).__init__()
        self.gru = nn.GRU(Feature_number, hidden_size=128, num_layers=4, batch_first=True)
        self.bn = nn.BatchNorm1d(128)

        self.resblock1 = ResBlock(128, 128)  # 添加残差块
        self.resblock2 = ResBlock(128, 128)  # 添加残差块
        self.resblock3 = ResBlock(128, 128)  # 添加残差块

        self.kan = nn.ModuleList(
            [KANLayer(in_dim=self.layer_nums_nobi[i], out_dim=self.layer_nums_nobi[i + 1], device='cuda') for i in
             range(len(self.layer_nums_nobi) - 1)])

    def __str__(self):
        return 'GRU_Resnet_kan'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(4, x.size(0), 128).to(x.device)
        output, _ = self.gru(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output)

        for layer in self.kan:
            output = layer(output)[0]
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class RNN_kan(nn.Module):
    from KAN.train_kan import layer_nums_nobi
    def __init__(self, Feature_number):
        super(RNN_kan, self).__init__()
        self.rnn = nn.RNN(Feature_number, hidden_size=128, num_layers=2, batch_first=True)
        self.kan = nn.ModuleList(
            [KANLayer(in_dim=self.layer_nums_nobi[i], out_dim=self.layer_nums_nobi[i + 1], device='cuda') for i in
             range(len(self.layer_nums_nobi) - 1)])

    def __str__(self):
        return 'RNN_kan'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(2, x.size(0), 128).to(x.device)

        output, _ = self.rnn(x, h0)
        # print(output.shape)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        # print(output.shape)
        for layer in self.kan:
            output = layer(output)[0]
        output_sigmoid = torch.sigmoid(output)
        return output_sigmoid.squeeze(1).to(x.device)


class RNN_small_kan(nn.Module):
    from KAN.train_kan import layer_nums_small_nobi
    def __init__(self, Feature_number):
        super(RNN_small_kan, self).__init__()
        self.rnn = nn.RNN(Feature_number, hidden_size=64, num_layers=2, batch_first=True)
        self.kan = nn.ModuleList(
            [KANLayer(in_dim=self.layer_nums_small_nobi[i], out_dim=self.layer_nums_small_nobi[i + 1], device='cuda') for i in
             range(len(self.layer_nums_small_nobi) - 1)])

    def __str__(self):
        return 'RNN_small_kan'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(2, x.size(0), 64).to(x.device)

        output, _ = self.rnn(x, h0)
        # print(output.shape)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        # print(output.shape)
        for layer in self.kan:
            output = layer(output)[0]
        output_sigmoid = torch.sigmoid(output)
        return output_sigmoid.squeeze(1).to(x.device)


class RNN_large_kan(nn.Module):
    from KAN.train_kan import layer_nums_large_nobi
    def __init__(self, Feature_number):
        super(RNN_large_kan, self).__init__()
        self.rnn = nn.RNN(Feature_number, hidden_size=256, num_layers=2, batch_first=True)
        self.kan = nn.ModuleList(
            [KANLayer(in_dim=self.layer_nums_large_nobi[i], out_dim=self.layer_nums_large_nobi[i + 1], device='cuda') for i in
             range(len(self.layer_nums_large_nobi) - 1)])

    def __str__(self):
        return 'RNN_large_kan'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(2, x.size(0), 256).to(x.device)

        output, _ = self.rnn(x, h0)
        # print(output.shape)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        # print(output.shape)
        for layer in self.kan:
            output = layer(output)[0]
        output_sigmoid = torch.sigmoid(output)
        return output_sigmoid.squeeze(1).to(x.device)


class RNN_3layers_kan(nn.Module):
    from KAN.train_kan import layer_nums_nobi
    def __init__(self, Feature_number):
        super(RNN_3layers_kan, self).__init__()
        self.rnn = nn.RNN(Feature_number, hidden_size=128, num_layers=3, batch_first=True)
        self.kan = nn.ModuleList(
            [KANLayer(in_dim=self.layer_nums_nobi[i], out_dim=self.layer_nums_nobi[i + 1], device='cuda') for i in
             range(len(self.layer_nums_nobi) - 1)])

    def __str__(self):
        return 'RNN_3layers_kan'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(3, x.size(0), 128).to(x.device)

        output, _ = self.rnn(x, h0)
        # print(output.shape)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        # print(output.shape)
        for layer in self.kan:
            output = layer(output)[0]
        output_sigmoid = torch.sigmoid(output)
        return output_sigmoid.squeeze(1).to(x.device)


class RNN_4layers_kan(nn.Module):
    from KAN.train_kan import layer_nums_nobi
    def __init__(self, Feature_number):
        super(RNN_4layers_kan, self).__init__()
        self.rnn = nn.RNN(Feature_number, hidden_size=128, num_layers=4, batch_first=True)
        self.kan = nn.ModuleList(
            [KANLayer(in_dim=self.layer_nums_nobi[i], out_dim=self.layer_nums_nobi[i + 1], device='cuda') for i in
             range(len(self.layer_nums_nobi) - 1)])

    def __str__(self):
        return 'RNN_4layers_kan'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(4, x.size(0), 128).to(x.device)

        output, _ = self.rnn(x, h0)
        # print(output.shape)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        # print(output.shape)
        for layer in self.kan:
            output = layer(output)[0]
        output_sigmoid = torch.sigmoid(output)
        return output_sigmoid.squeeze(1).to(x.device)


class RNN_Resnet_kan(nn.Module):
    from KAN.train_kan import layer_nums_nobi
    def __init__(self, Feature_number):
        super(RNN_Resnet_kan, self).__init__()
        self.rnn = nn.RNN(Feature_number, hidden_size=128, num_layers=4, batch_first=True)
        self.bn = nn.BatchNorm1d(128)

        self.resblock1 = ResBlock(128, 128)  # 添加残差块
        self.resblock2 = ResBlock(128, 128)  # 添加残差块
        self.resblock3 = ResBlock(128, 128)  # 添加残差块

        self.kan = nn.ModuleList(
            [KANLayer(in_dim=self.layer_nums_nobi[i], out_dim=self.layer_nums_nobi[i + 1], device='cuda') for i in
             range(len(self.layer_nums_nobi) - 1)])

    def __str__(self):
        return 'RNN_Resnet_kan'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(4, x.size(0), 128).to(x.device)
        output, _ = self.rnn(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output)

        for layer in self.kan:
            output = layer(output)[0]
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)