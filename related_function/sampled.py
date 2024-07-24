import torch
from imblearn.over_sampling import SMOTE

class BalancedData:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        pass

    def oversample(self):
        """
        过采样
        :return: 平衡后的数据和标签
        """
        # 找到正类和负类的索引
        positive_indices = torch.where(self.labels == 1)[0]
        negative_indices = torch.where(self.labels == 0)[0]

        # 计算正类和负类的数量
        num_positive = len(positive_indices)
        num_negative = len(negative_indices)

        # 过采样，使正类=负类
        if num_positive < num_negative:
            # 对正类进行过采样
            oversampled_indices = torch.randint(num_positive, (num_negative - num_positive,))
            positive_indices = torch.cat((positive_indices, positive_indices[oversampled_indices]))
        elif num_positive > num_negative:
            # 对负类进行过采样
            oversampled_indices = torch.randint(num_negative, (num_positive - num_negative,))
            negative_indices = torch.cat((negative_indices, negative_indices[oversampled_indices]))

        # 确保正类和负类数量相等后，重新排列数据并打乱顺序
        indices = torch.cat((positive_indices, negative_indices))
        shuffled_indices = torch.randperm(len(indices))
        indices = indices[shuffled_indices]
        data_balanced = self.data[indices]
        labels_balanced = self.labels[indices]

        return data_balanced, labels_balanced

    def undersample(self):
        """
        欠采样
        :return: 平衡后的数据和标签
        """
        # 找到正类和负类的索引
        positive_indices = torch.where(self.labels == 1)[0]
        negative_indices = torch.where(self.labels == 0)[0]

        # 计算正类和负类的数量
        num_positive = len(positive_indices)
        num_negative = len(negative_indices)

        # 欠采样，使正类=负类
        if num_positive < num_negative:
            # 对负类进行欠采样
            undersampled_indices = torch.randint(num_negative, (num_positive,))
            negative_indices = negative_indices[undersampled_indices]
        elif num_positive > num_negative:
            # 对正类进行欠采样
            undersampled_indices = torch.randint(num_positive, (num_negative,))
            positive_indices = positive_indices[undersampled_indices]

        # 确保正类和负类数量相等后，重新排列数据并打乱顺序
        indices = torch.cat((positive_indices, negative_indices))
        shuffled_indices = torch.randperm(len(indices))
        indices = indices[shuffled_indices]
        data_balanced = self.data[indices]
        labels_balanced = self.labels[indices]

        return data_balanced, labels_balanced

    def smote(self):
        """
        过采样
        :param data: 数据
        :param label: 标签
        :return: 平衡后的数据和标签
        """
        data_numpy = self.data.numpy()
        labels_numpy = self.labels.numpy()

        # 获取数据的形状
        original_shape = data_numpy.shape

        # 将数据重塑为二维
        data_reshaped = data_numpy.reshape(original_shape[0], -1)

        smote = SMOTE(random_state=42)
        data_resampled, labels_resampled = smote.fit_resample(data_reshaped, labels_numpy)

        # 将数据重塑回原来的形状
        data_resampled = data_resampled.reshape(-1, *original_shape[1:])

        data_balanced = torch.from_numpy(data_resampled)
        labels_balanced = torch.from_numpy(labels_resampled)
        return data_balanced, labels_balanced

    def origin(self):
        data_balanced = self.data
        labels_balanced = self.labels
        return data_balanced, labels_balanced

    def sample(self, method):
        """
        根据指定方法进行采样
        """
        if method == 'oversample':
            return self.oversample()
        elif method == 'undersample':
            return self.undersample()
        elif method == 'smote':
            return self.smote()
        elif method == 'origin':
            return self.origin()
        else:
            raise ValueError(f"Unknown sampling method: {method}")
