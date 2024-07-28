import torch
from torch import nn
from related_function.xlstm import xLSTM
import torch.nn.functional as F
class BiLSTM(nn.Module):
    def __init__(self, Feature_number):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=128, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.bn = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 8)
        self.fc5 = nn.Linear(8, 1)
        self.sig = nn.Sigmoid()

    def __str__(self):
        return 'BiLSTM'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(4, x.size(0), 128).to(x.device)
        c0 = torch.zeros(4, x.size(0), 128).to(x.device)

        output, _ = self.lstm(x, (h0, c0))
        # print(output.shape)
        output_hd = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        # print(output.shape)
        output_bn = self.bn(output_hd)  if output_hd.shape[0] != 1 else output_hd
        # print(output.shape)
        output_fc1 = torch.relu(self.fc1(output_bn))
        # print(output.shape)
        output_fc2 = torch.relu(self.fc2(output_fc1))
        # print(output.shape)
        output_fc3 = torch.relu(self.fc3(output_fc2))
        # print(output.shape)
        output_fc4 = torch.relu(self.fc4(output_fc3))
        # print(output.shape)
        output_sigmoid = torch.sigmoid(self.fc5(output_fc4))
        # print(output.shape)
        return output_sigmoid.squeeze(1).to(x.device)


class BiLSTM_BN(nn.Module):
    def __init__(self, Feature_number):
        super(BiLSTM_BN, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=128, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.bn = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(256, 128)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(128)  # 批标准化层

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(64)  # 批标准化层

        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(32)  # 批标准化层

        self.fc4 = nn.Linear(32, 8)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(8)  # 批标准化层

        self.fc5 = nn.Linear(8, 1)

    def __str__(self):
        return 'BiLSTM_BN'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(4, x.size(0), 128).to(x.device)
        c0 = torch.zeros(4, x.size(0), 128).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        # output = self.bn(output) if output.shape[0] != 1 else output
        output = self.bn(output) if output.shape[0] != 1 else output

        output = self.fc1(output)
        output = self.bn1(output) if output.shape[0] != 1 else output
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.bn2(output) if output.shape[0] != 1 else output
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output) if output.shape[0] != 1 else output
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output) if output.shape[0] != 1 else output
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class BiLSTM_BN_larger(nn.Module):
    def __init__(self, Feature_number):
        super(BiLSTM_BN_larger, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=256, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.bn = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 128)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(128)  # 批标准化层

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(64)  # 批标准化层

        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(32)  # 批标准化层

        self.fc4 = nn.Linear(32, 8)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(8)  # 批标准化层

        self.fc5 = nn.Linear(8, 1)

    def __str__(self):
        return 'BiLSTM_BN_larger'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(4, x.size(0), 256).to(x.device)
        c0 = torch.zeros(4, x.size(0), 256).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output) if output.shape[0] != 1 else output

        output = self.fc1(output)
        output = self.bn1(output) if output.shape[0] != 1 else output
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.bn2(output) if output.shape[0] != 1 else output
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output) if output.shape[0] != 1 else output
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output) if output.shape[0] != 1 else output
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class BiLSTM_BN_Resnet(nn.Module):
    def __init__(self, Feature_number):
        super(BiLSTM_BN_Resnet, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=128, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.bn = nn.BatchNorm1d(256)

        self.fc1_1 = nn.Linear(256, 256)
        self.bn1_1 = nn.BatchNorm1d(256)
        self.relu1_1 = nn.ReLU()

        self.fc1_2 = nn.Linear(256, 256)
        self.bn1_2 = nn.BatchNorm1d(256)
        self.relu1_2 = nn.ReLU()

        self.fc1_3 = nn.Linear(256, 256)
        self.bn1_3 = nn.BatchNorm1d(256)
        self.relu1_3 = nn.ReLU()

        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(32, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.relu4 = nn.ReLU()

        self.fc5 = nn.Linear(8, 1)

    def __str__(self):
        return 'BiLSTM_BN_Resnet'

    def forward(self, x):
        h0 = torch.zeros(4, x.size(0), 128).to(x.device)
        c0 = torch.zeros(4, x.size(0), 128).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output) if output.shape[0] != 1 else output

        output_1 = self.fc1_1(output)
        output_1 = self.bn1_1(output_1) if output_1.shape[0] != 1 else output_1
        output_1 = self.relu1_1(output_1)
        output_1 = output_1 + output

        output_2 = self.fc1_2(output_1)
        output_2 = self.bn1_2(output_2) if output_2.shape[0] != 1 else output_2
        output_2 = self.relu1_2(output_2)
        output_2 = output_2 + output_1

        output_3 = self.fc1_3(output_2)
        output_3 = self.bn1_3(output_3) if output_3.shape[0] != 1 else output_3
        output_3 = self.relu1_3(output_3)
        output_3 = output_3 + output_2

        output = self.fc2(output_3)
        output = self.bn2(output) if output.shape[0] != 1 else output
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output) if output.shape[0] != 1 else output
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output) if output.shape[0] != 1 else output
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class BiLSTM_very_large(nn.Module):
    def __init__(self, Feature_number):
        super(BiLSTM_very_large, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=512*4, num_layers=3, batch_first=True,
                            bidirectional=True)
        self.bn = nn.BatchNorm1d(1024*4)

        self.fc1 = nn.Linear(1024*4, 512)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(512)  # 批标准化层

        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(128)  # 批标准化层

        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(64)  # 批标准化层

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def __str__(self):
        return 'BiLSTM_BN_3layers'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(6, x.size(0), 512*4).to(x.device)
        c0 = torch.zeros(6, x.size(0), 512*4).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output) if output.shape[0] != 1 else output

        output = self.fc1(output)
        output = self.bn1(output) if output.shape[0] != 1 else output
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.bn2(output) if output.shape[0] != 1 else output
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output) if output.shape[0] != 1 else output
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output) if output.shape[0] != 1 else output
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)

class BiLSTM_BN_3layers(nn.Module):
    def __init__(self, Feature_number):
        super(BiLSTM_BN_3layers, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=512, num_layers=3, batch_first=True,
                            bidirectional=True)
        self.bn = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(512)  # 批标准化层

        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(128)  # 批标准化层

        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(64)  # 批标准化层

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def __str__(self):
        return 'BiLSTM_BN_3layers'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(6, x.size(0), 512).to(x.device)
        c0 = torch.zeros(6, x.size(0), 512).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output) if output.shape[0] != 1 else output

        output = self.fc1(output)
        output = self.bn1(output) if output.shape[0] != 1 else output
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.bn2(output) if output.shape[0] != 1 else output
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output) if output.shape[0] != 1 else output
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output) if output.shape[0] != 1 else output
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class BiLSTM_BN_4layers(nn.Module):
    def __init__(self, Feature_number):
        super(BiLSTM_BN_4layers, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=512, num_layers=4, batch_first=True,
                            bidirectional=True)
        self.bn = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(512)  # 批标准化层

        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(128)  # 批标准化层

        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(64)  # 批标准化层

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def __str__(self):
        return 'BiLSTM_BN_4layers'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(8, x.size(0), 512).to(x.device)
        c0 = torch.zeros(8, x.size(0), 512).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output) if output.shape[0] != 1 else output

        output = self.fc1(output)
        output = self.bn1(output) if output.shape[0] != 1 else output
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.bn2(output) if output.shape[0] != 1 else output
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output) if output.shape[0] != 1 else output
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output) if output.shape[0] != 1 else output
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class GRU_BN(nn.Module):
    def __init__(self, Feature_number):
        super(GRU_BN, self).__init__()
        self.gru = nn.GRU(Feature_number, hidden_size=1024, num_layers=2, batch_first=True)
        self.bn = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(512)  # 批标准化层

        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(128)  # 批标准化层

        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(64)  # 批标准化层

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def __str__(self):
        return 'GRU_BN'

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 1024).to(x.device)

        output, _ = self.gru(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output) if output.shape[0] != 1 else output

        output = self.fc1(output)
        output = self.bn1(output) if output.shape[0] != 1 else output
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.bn2(output) if output.shape[0] != 1 else output
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output) if output.shape[0] != 1 else output
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output) if output.shape[0] != 1 else output
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class GRU_BN_3layers(nn.Module):
    def __init__(self, Feature_number):
        super(GRU_BN_3layers, self).__init__()
        self.gru = nn.GRU(Feature_number, hidden_size=1024, num_layers=3, batch_first=True)
        self.bn = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(512)  # 批标准化层

        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(128)  # 批标准化层

        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(64)  # 批标准化层

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def __str__(self):
        return 'GRU_BN_3layers'

    def forward(self, x):
        h0 = torch.zeros(3, x.size(0), 1024).to(x.device)

        output, _ = self.gru(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output) if output.shape[0] != 1 else output

        output = self.fc1(output)
        output = self.bn1(output) if output.shape[0] != 1 else output
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.bn2(output) if output.shape[0] != 1 else output
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output) if output.shape[0] != 1 else output
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output) if output.shape[0] != 1 else output
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class GRU_BN_4layers(nn.Module):
    def __init__(self, Feature_number):
        super(GRU_BN_4layers, self).__init__()
        self.gru = nn.GRU(Feature_number, hidden_size=1024, num_layers=4, batch_first=True)
        self.bn = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(512)  # 批标准化层

        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(128)  # 批标准化层

        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(64)  # 批标准化层

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def __str__(self):
        return 'GRU_BN_4layers'

    def forward(self, x):
        h0 = torch.zeros(4, x.size(0), 1024).to(x.device)

        output, _ = self.gru(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output) if output.shape[0] != 1 else output

        output = self.fc1(output)
        output = self.bn1(output) if output.shape[0] != 1 else output
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.bn2(output) if output.shape[0] != 1 else output
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output) if output.shape[0] != 1 else output
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output) if output.shape[0] != 1 else output
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class RNN_BN(nn.Module):
    def __init__(self, Feature_number):
        super(RNN_BN, self).__init__()
        self.rnn = nn.RNN(Feature_number, hidden_size=1024, num_layers=2, batch_first=True)  # 修改此处
        self.bn = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(512)  # 批标准化层

        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(128)  # 批标准化层

        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(64)  # 批标准化层

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def __str__(self):
        return 'RNN_BN'

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 1024).to(x.device)

        output, _ = self.rnn(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output) if output.shape[0] != 1 else output

        output = self.fc1(output)
        output = self.bn1(output) if output.shape[0] != 1 else output
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.bn2(output) if output.shape[0] != 1 else output
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output) if output.shape[0] != 1 else output
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output) if output.shape[0] != 1 else output
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class RNN_BN_3layers(nn.Module):
    def __init__(self, Feature_number):
        super(RNN_BN_3layers, self).__init__()
        self.rnn = nn.RNN(Feature_number, hidden_size=1024, num_layers=3, batch_first=True)  # 修改此处
        self.bn = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(512)  # 批标准化层

        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(128)  # 批标准化层

        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(64)  # 批标准化层

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def __str__(self):
        return 'RNN_BN_3layers'

    def forward(self, x):
        h0 = torch.zeros(3, x.size(0), 1024).to(x.device)

        output, _ = self.rnn(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output) if output.shape[0] != 1 else output

        output = self.fc1(output)
        output = self.bn1(output) if output.shape[0] != 1 else output
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.bn2(output) if output.shape[0] != 1 else output
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output) if output.shape[0] != 1 else output
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output) if output.shape[0] != 1 else output
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class RNN_BN_4layers(nn.Module):
    def __init__(self, Feature_number):
        super(RNN_BN_4layers, self).__init__()
        self.rnn = nn.RNN(Feature_number, hidden_size=1024, num_layers=4, batch_first=True)  # 修改此处
        self.bn = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(512)  # 批标准化层

        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(128)  # 批标准化层

        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()  # ReLU激活函数
        self.bn3 = nn.BatchNorm1d(64)  # 批标准化层

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def __str__(self):
        return 'RNN_BN_4layers'

    def forward(self, x):
        h0 = torch.zeros(4, x.size(0), 1024).to(x.device)

        output, _ = self.rnn(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output) if output.shape[0] != 1 else output

        output = self.fc1(output)
        output = self.bn1(output) if output.shape[0] != 1 else output
        output = self.relu1(output)

        output = self.fc2(output)
        output = self.bn2(output) if output.shape[0] != 1 else output
        output = self.relu2(output)

        output = self.fc3(output)
        output = self.bn3(output) if output.shape[0] != 1 else output
        output = self.relu3(output)

        output = self.fc4(output)
        output = self.bn4(output) if output.shape[0] != 1 else output
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


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
        out = self.bn1(out)  if out.shape[0] != 1 else out
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.bn2(out)  if out.shape[0] != 1 else out

        out += identity
        out = self.relu2(out)

        return out


class BiLSTM_BN_ResBlock(nn.Module):
    def __init__(self, Feature_number):
        super(BiLSTM_BN_ResBlock, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=128, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.bn = nn.BatchNorm1d(256)

        self.resblock1 = ResBlock(256, 128)  # 添加残差块
        self.resblock2 = ResBlock(128, 64)  # 添加残差块
        self.resblock3 = ResBlock(64, 32)  # 添加残差块

        self.fc4 = nn.Linear(32, 8)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(8)  # 批标准化层

        self.fc5 = nn.Linear(8, 1)

    def __str__(self):
        return 'BiLSTM_BN_ResBlock'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(4, x.size(0), 128).to(x.device)
        c0 = torch.zeros(4, x.size(0), 128).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output) if output.shape[0] != 1 else output

        output = self.resblock1(output)  # 通过残差块
        output = self.resblock2(output)  # 通过残差块
        output = self.resblock3(output)  # 通过残差块

        output = self.fc4(output)
        output = self.bn4(output) if output.shape[0] != 1 else output
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class GRU_BN_ResBlock(nn.Module):
    def __init__(self, Feature_number):
        super(GRU_BN_ResBlock, self).__init__()
        self.gru = nn.GRU(Feature_number, hidden_size=1024, num_layers=2, batch_first=True)
        self.bn = nn.BatchNorm1d(1024)

        self.resblock1 = ResBlock(1024, 512)  # 添加残差块
        self.resblock2 = ResBlock(512, 128)  # 添加残差块
        self.resblock3 = ResBlock(128, 64)  # 添加残差块

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def __str__(self):
        return 'GRU_BN_ResBlock'

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 1024).to(x.device)

        output, _ = self.gru(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output) if output.shape[0] != 1 else output

        output = self.resblock1(output)  # 通过残差块
        output = self.resblock2(output)  # 通过残差块
        output = self.resblock3(output)  # 通过残差块

        output = self.fc4(output)
        output = self.bn4(output) if output.shape[0] != 1 else output
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class RNN_BN_ResBlock(nn.Module):
    def __init__(self, Feature_number):
        super(RNN_BN_ResBlock, self).__init__()
        self.rnn = nn.RNN(Feature_number, hidden_size=1024, num_layers=2, batch_first=True)  # 修改此处
        self.bn = nn.BatchNorm1d(1024)

        self.resblock1 = ResBlock(1024, 512)  # 添加残差块
        self.resblock2 = ResBlock(512, 128)  # 添加残差块
        self.resblock3 = ResBlock(128, 64)  # 添加残差块

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def __str__(self):
        return 'RNN_BN_ResBlock'

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 1024).to(x.device)

        output, _ = self.rnn(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output) if output.shape[0] != 1 else output

        output = self.resblock1(output)  # 通过残差块
        output = self.resblock2(output)  # 通过残差块
        output = self.resblock3(output)  # 通过残差块

        output = self.fc4(output)
        output = self.bn4(output) if output.shape[0] != 1 else output
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class BiLSTM_BN_ResBlock_3layers(nn.Module):
    def __init__(self, Feature_number):
        super(BiLSTM_BN_ResBlock_3layers, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=128, num_layers=3, batch_first=True,
                            bidirectional=True)
        self.bn = nn.BatchNorm1d(256)

        self.resblock1 = ResBlock(256, 128)  # 添加残差块
        self.resblock2 = ResBlock(128, 64)  # 添加残差块
        self.resblock3 = ResBlock(64, 32)  # 添加残差块

        self.fc4 = nn.Linear(32, 8)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(8)  # 批标准化层

        self.fc5 = nn.Linear(8, 1)

    def __str__(self):
        return 'BiLSTM_BN_ResBlock_3layers'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(6, x.size(0), 128).to(x.device)
        c0 = torch.zeros(6, x.size(0), 128).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output) if output.shape[0] != 1 else output

        output = self.resblock1(output)  # 通过残差块
        output = self.resblock2(output)  # 通过残差块
        output = self.resblock3(output)  # 通过残差块

        output = self.fc4(output)
        output = self.bn4(output) if output.shape[0] != 1 else output
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class GRU_BN_ResBlock_3layers(nn.Module):
    def __init__(self, Feature_number):
        super(GRU_BN_ResBlock_3layers, self).__init__()
        self.gru = nn.GRU(Feature_number, hidden_size=1024, num_layers=3, batch_first=True)
        self.bn = nn.BatchNorm1d(1024)

        self.resblock1 = ResBlock(1024, 512)  # 添加残差块
        self.resblock2 = ResBlock(512, 128)  # 添加残差块
        self.resblock3 = ResBlock(128, 64)  # 添加残差块

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def __str__(self):
        return 'GRU_BN_ResBlock_3layers'

    def forward(self, x):
        h0 = torch.zeros(3, x.size(0), 1024).to(x.device)

        output, _ = self.gru(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output) if output.shape[0] != 1 else output

        output = self.resblock1(output)  # 通过残差块
        output = self.resblock2(output)  # 通过残差块
        output = self.resblock3(output)  # 通过残差块

        output = self.fc4(output)
        output = self.bn4(output) if output.shape[0] != 1 else output
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class RNN_BN_ResBlock_3layers(nn.Module):
    def __init__(self, Feature_number):
        super(RNN_BN_ResBlock_3layers, self).__init__()
        self.rnn = nn.RNN(Feature_number, hidden_size=1024, num_layers=3, batch_first=True)  # 修改此处
        self.bn = nn.BatchNorm1d(1024)

        self.resblock1 = ResBlock(1024, 512)  # 添加残差块
        self.resblock2 = ResBlock(512, 128)  # 添加残差块
        self.resblock3 = ResBlock(128, 64)  # 添加残差块

        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()  # ReLU激活函数
        self.bn4 = nn.BatchNorm1d(16)  # 批标准化层

        self.fc5 = nn.Linear(16, 1)

    def __str__(self):
        return 'RNN_BN_ResBlock_3layers'

    def forward(self, x):
        h0 = torch.zeros(3, x.size(0), 1024).to(x.device)

        output, _ = self.rnn(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output) if output.shape[0] != 1 else output

        output = self.resblock1(output)  # 通过残差块
        output = self.resblock2(output)  # 通过残差块
        output = self.resblock3(output)  # 通过残差块

        output = self.fc4(output)
        output = self.bn4(output) if output.shape[0] != 1 else output
        output = self.relu4(output)

        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class Custom(nn.Module):
    def __init__(self, Feature_number):
        super(Custom, self).__init__()
        self.fc5 = nn.Linear(5, 1)


    def forward(self, x):
        x = self.fc5(x)
        return torch.zeros(x.shape[0],).to(x.device)


class BiLSTM_BN_single(nn.Module):
    def __init__(self, Feature_number):
        super(BiLSTM_BN_single, self).__init__()
        self.lstm = nn.LSTM(input_size=Feature_number, hidden_size=128, num_layers=3, batch_first=True,
                            bidirectional=True)
        self.bn = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 1)

    def __str__(self):
        return 'BiLSTM_BN_single'

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(6, x.size(0), 128).to(x.device)
        c0 = torch.zeros(6, x.size(0), 128).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output) if output.shape[0] != 1 else output
        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class GRU_BN_single(nn.Module):
    def __init__(self, Feature_number):
        super(GRU_BN_single, self).__init__()
        self.gru = nn.GRU(Feature_number, hidden_size=1024, num_layers=3, batch_first=True)
        self.bn = nn.BatchNorm1d(1024)
        self.fc5 = nn.Linear(1024, 1)

    def __str__(self):
        return 'GRU_BN_single'

    def forward(self, x):
        h0 = torch.zeros(3, x.size(0), 1024).to(x.device)

        output, _ = self.gru(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output) if output.shape[0] != 1 else output
        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class RNN_BN_single(nn.Module):
    def __init__(self, Feature_number):
        super(RNN_BN_single, self).__init__()
        self.rnn = nn.RNN(Feature_number, hidden_size=1024, num_layers=3, batch_first=True)  # 修改此处
        self.bn = nn.BatchNorm1d(1024)
        self.fc5 = nn.Linear(1024, 1)

    def __str__(self):
        return 'RNN_BN_single'

    def forward(self, x):
        h0 = torch.zeros(3, x.size(0), 1024).to(x.device)

        output, _ = self.rnn(x, h0)
        output = output[:, -1, :]  # 取最后一个时间步的隐藏状态
        output = self.bn(output) if output.shape[0] != 1 else output
        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class sLSTM(nn.Module):
    def __init__(self, Feature_number):
        super(sLSTM, self).__init__()
        self.xlstm = xLSTM(Feature_number, hidden_size=120, num_heads=Feature_number, layers=['s', 's', 's', 's'],
                           batch_first=True)
        self.bn = nn.BatchNorm1d(Feature_number)
        self.fc5 = nn.Linear(Feature_number, 1)

    def __str__(self):
        return 'sLSTM'

    def forward(self, x):
        output, _ = self.xlstm(x) # [256, 8, 5]
        output = output[:, -1, :]
        output = self.bn(output) if output.shape[0] != 1 else output
        output = self.fc5(output)
        output = torch.sigmoid(output)

        return output.squeeze(1).to(x.device)


class LSTMAttentionModel(nn.Module):
    def __init__(self, Feature_number=5, hidden_dim=128, output_dim=1, num_layers=1, bidirectional=True):
        super(LSTMAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(Feature_number, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.attention = Attention(hidden_dim * 2 if bidirectional else hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch_size, seq_len, hidden_dim * num_directions)
        attn_out = self.attention(lstm_out)
        # attn_out: (batch_size, hidden_dim * num_directions)
        output = self.fc(attn_out)
        # output: (batch_size, output_dim)
        output = torch.sigmoid(output)
        return output.squeeze(1).to(x.device)


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, lstm_out):
        # lstm_out: (batch_size, seq_len, hidden_dim)
        energy = torch.tanh(self.attn(lstm_out))  # (batch_size, seq_len, hidden_dim)
        energy = energy.transpose(1, 2)  # (batch_size, hidden_dim, seq_len)
        v = self.v.repeat(lstm_out.size(0), 1).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        attention_weights = torch.bmm(v, energy).squeeze(1)  # (batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)  # (batch_size, seq_len)

        attention_weights = attention_weights.unsqueeze(1)  # (batch_size, 1, seq_len)
        context_vector = torch.bmm(attention_weights, lstm_out).squeeze(1)  # (batch_size, hidden_dim)

        return context_vector


class BiLSTM_Attention(nn.Module):
    def __init__(self, Feature_number=5, lstm_units=128):
        super(BiLSTM_Attention, self).__init__()
        self.bilstm = nn.LSTM(input_size=Feature_number, hidden_size=lstm_units, num_layers=1, bidirectional=True,
                              batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(lstm_units * 2, lstm_units * 2),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(lstm_units * 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x, _ = self.bilstm(x)
        attention_weights = self.attention(x)
        x = x * attention_weights
        x = self.fc(x[:, -1, :])  # Taking the last time step
        output = torch.sigmoid(x)
        return output.squeeze(1).to(x.device)


class LSTM_with_Attention(nn.Module):
    def __init__(self, Feature_number=5, hidden_dim=120, output_dim=1, num_layers=2, use_dropout=True):
        super().__init__()
        self.rnn = nn.LSTM(Feature_number, hidden_dim, bidirectional=True, dropout=0.5 if use_dropout else 0.,
                           batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5 if use_dropout else 0.)

    # def attention_net(self, lstm_output, final_state):
    #     lstm_output = lstm_output.permute(1, 0, 2)
    #     hidden = final_state.squeeze(0)
    #     attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
    #     soft_attn_weights = F.softmax(attn_weights, dim=1)
    #     new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
    #                                  soft_attn_weights.unsqueeze(2)).squeeze(2)
    #
    #     return new_hidden_state

    def attention(self, lstm_output, final_state):
        # lstm_output = lstm_output.permute(1, 0, 2)
        merged_state = torch.cat([s for s in final_state[-2:, :, :]], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, x):
        output, (hidden, cell) = self.rnn(x)

        # attn_output = self.attention_net(output, hidden)
        attn_output = self.attention(output, hidden)

        output = self.fc(attn_output.squeeze(0))
        output = torch.sigmoid(output)
        return output.squeeze(1).to(x.device)