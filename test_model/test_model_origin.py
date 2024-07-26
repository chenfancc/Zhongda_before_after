import json
import os
import pickle
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch.optim
from sklearn.metrics import *
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader


# 获取项目根目录的路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from related_function.function import calculate_metrics, plot_confusion_matrix, main_data_loader


warnings.filterwarnings('ignore', category=RuntimeWarning)

class TestModel:
    """
    测试网络的类
    """

    def __init__(self, my_model_name, observe_window, predict_window, model, device, epoch, origin_data, root_dir=None, is_print=True, NUM_FEATURES=5):
        """
        初始化训练
        :param my_model_name: 模型名称
        :param model: 定义好的模型结构，在model.py文件中，直接传入模型的类名，如BiLSTM_BN
        """
        self.batch_size = 1024
        self.observe_window = observe_window
        self.predict_window = predict_window
        self.model = model
        self.epoch = epoch
        self.DEVICE = device
        self.root_dir = root_dir
        self.model_name = my_model_name
        self.data = origin_data
        self.model.to(self.DEVICE)
        self.is_print = is_print
        self.NUM_FEATURES=NUM_FEATURES
        self.dataloader = self.dataloader_origin()

    def test(self):
        """
        训练主函数
        :return: info.json
        """

        if self.root_dir is None:
            model_directory = f"./{self.model_name}/"
        else:
            model_directory = f"./{self.root_dir}/{self.model_name}/"

        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        (threshold_auc, accuracy_auc, specificity_auc, alarm_sen_auc, alarm_acc_auc,
         threshold_prc, accuracy_prc, specificity_prc, alarm_sen_prc, alarm_acc_prc,
         auc, prc, brier, true_labels_flat, predicted_probs_flat,
         acc_list, spe_list, alarm_sen_list, alarm_acc_list, cm_list, PPV_list, NPV_list, F1_list) = self.validate(self.dataloader)

        info = {
            "accuracy_list": acc_list,
            "specificity_list": spe_list,
            "alarm_sen_list": alarm_sen_list,
            "alarm_acc_list": alarm_acc_list,
            "cm_list": cm_list,
            "PPV_list": PPV_list,
            "NPV_list": NPV_list,
            "F1_list": F1_list,
            "threshold_auc": threshold_auc,
            "threshold_prc": threshold_prc,
            "accuracy_auc": accuracy_auc,
            "specificity_auc": specificity_auc,
            "alarm_sen_auc": alarm_sen_auc,
            "alarm_acc_auc": alarm_acc_auc,
            "accuracy_prc": accuracy_prc,
            "specificity_prc": specificity_prc,
            "alarm_sen_prc": alarm_sen_prc,
            "alarm_acc_prc": alarm_acc_prc,
            "auc": auc,
            "prc": prc,
            "brier": brier
        }

        return info, true_labels_flat, predicted_probs_flat

    def _generate_label(self, row, x):
        if 0 < row[-2] <= x:
            return row[-3]
        else:
            return 0
    def dataloader_origin(self):
        unique_ids = np.unique(self.data[:, 0])
        print(unique_ids.shape)

        data_list = []  # 临时存储列表
        label_list = []  # 临时标签列表
        for _, id in enumerate(tqdm(unique_ids)):  # 添加 tqdm 进度条

            patient_data = self.data[self.data[:, 0] == id][:, 1:1 + self.NUM_FEATURES].astype(float)  # 获取特征A和特征B的值
            label_data = np.array([self._generate_label(row, self.predict_window) for row in self.data[self.data[:, 0] == id]]).astype(float)

            for j in range(patient_data.shape[0]):
                if j < self.observe_window - 1:
                    patient_meta_tensor = torch.tensor(patient_data[:j, :]).unsqueeze(0)
                    data_list.append(patient_meta_tensor)
                else:
                    patient_meta_tensor = torch.tensor(patient_data[j - self.observe_window + 1:j, :])
                    data_list.append(patient_meta_tensor)
                label_meta_tensor = torch.tensor(label_data[j]).unsqueeze(0)
                label_list.append(label_meta_tensor)

        # 将列表转换为张量
        data_tensor = torch.cat(data_list, dim=0)
        label_tensor = torch.cat(label_list, dim=0)

        dataset = TensorDataset(data_tensor, label_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        return dataloader

    def validate(self):
        """
        计算每一个epoch结束的模型性能
        :return: valid_loss, valid_accuracy, valid_specificity, valid_alarm_sen, valid_alarm_acc, valid_auc
        """
        self.model.eval()
        self.model.to(self.DEVICE)
        eps = 1e-6
        total_valid_loss = eps
        true_labels = []
        predicted_probs = []
        count = 0

        acc_list = []
        spe_list = []
        alarm_sen_list = []
        alarm_acc_list = []
        cm_list = []
        PPV_list = []
        NPV_list = []
        F1_list = []

        with torch.no_grad():
            for inputs, targets in self.dataloader:
                count += 1
                inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
                outputs = self.model(inputs.float())
                true_labels.append(targets.cpu().numpy())
                predicted_probs.append(outputs.cpu().numpy())

            true_labels_flat = np.concatenate(true_labels)
            predicted_probs_flat = np.concatenate(predicted_probs)

            brier_score = np.mean((predicted_probs_flat - true_labels_flat) ** 2)

            valid_auc, best_threshold_auc = self._plot_roc_curve(true_labels_flat, predicted_probs_flat, self.epoch)
            valid_prc, best_threshold_prc = self._plot_prc_curve(true_labels_flat, predicted_probs_flat, self.epoch)
            valid_accuracy_auc, valid_specificity_auc, valid_alarm_sen_auc, valid_alarm_acc_auc, cm_auc, PPV, NPV, F1 = self._calculate_criterion(
                true_labels_flat, predicted_probs_flat, best_threshold_auc, self.epoch, "auc")
            valid_accuracy_prc, valid_specificity_prc, valid_alarm_sen_prc, valid_alarm_acc_prc, cm_prc, PPV, NPV, F1 = self._calculate_criterion(
                true_labels_flat, predicted_probs_flat, best_threshold_prc, self.epoch, "prc")

            for i in range(0, 1000) if self.is_print else tqdm(range(0, 1000)):
                threshold = i / 1000
                valid_accuracy, valid_specificity, valid_alarm_sen, valid_alarm_acc, cm, PPV, NPV, F1 = self._calculate_criterion(
                    true_labels_flat, predicted_probs_flat, threshold, self.epoch, f"threshold = {threshold}")
                acc_list.append(valid_accuracy)
                spe_list.append(valid_specificity)
                alarm_sen_list.append(valid_alarm_sen)
                alarm_acc_list.append(valid_alarm_acc)
                cm_list.append(cm)
                PPV_list.append(PPV)
                NPV_list.append(NPV)
                F1_list.append(F1)

        return (best_threshold_auc, valid_accuracy_auc, valid_specificity_auc, valid_alarm_sen_auc, valid_alarm_acc_auc,
                best_threshold_prc, valid_accuracy_prc, valid_specificity_prc, valid_alarm_sen_prc, valid_alarm_acc_prc,
                valid_auc, valid_prc, brier_score, true_labels_flat, predicted_probs_flat,
                acc_list, spe_list, alarm_sen_list, alarm_acc_list, cm_list, PPV_list, NPV_list, F1_list)

    def _plot_roc_curve(self, true_labels_flat, predicted_probs_flat, epoch):
        fpr, tpr, thresholds = roc_curve(true_labels_flat, predicted_probs_flat)
        valid_auc = auc(fpr, tpr)
        best_threshold_index = (tpr - fpr).argmax()
        best_threshold = thresholds[best_threshold_index]

        if self.is_print: print(f"AUROC: {valid_auc:.4f}")
        if self.is_print: print(f"Best threshold: {best_threshold:.4f}")
        # 绘制ROC曲线
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {valid_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) - Epoch {epoch}')
        plt.legend(loc="lower right")
        if self.root_dir is None:
            plt.savefig(f"{self.model_name}/{self.model_name}_ROC_EPOCH_{epoch}.png")
        else:
            plt.savefig(f"{self.root_dir}/{self.model_name}/{self.model_name}_ROC_EPOCH_{epoch}.png")

        plt.close()

        return valid_auc, best_threshold

    def _plot_prc_curve(self, true_labels_flat, predicted_probs_flat, epoch):
        precision, recall, thresholds = precision_recall_curve(true_labels_flat, predicted_probs_flat)
        prc_auc = auc(recall, precision)
        best_threshold_index = (precision * recall / (precision + recall + 1e-6)).argmax()
        best_threshold = thresholds[best_threshold_index]
        if self.is_print: print(f"AUPRC: {prc_auc:.4f}")
        if self.is_print: print(f"Best threshold: {best_threshold:.4f}")

        plt.figure()
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PRC curve (area = {prc_auc:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - Epoch {epoch}')
        plt.legend(loc='lower left')
        if self.root_dir is None:
            plt.savefig(f"{self.model_name}/{self.model_name}_PRC_EPOCH_{epoch}.png")
        else:
            plt.savefig(f"{self.root_dir}/{self.model_name}/{self.model_name}_PRC_EPOCH_{epoch}.png")
        plt.grid()
        plt.close()

        return prc_auc, best_threshold

    def _calculate_criterion(self, true_labels_flat, predicted_probs_flat, best_threshold, epoch, name):
        cm, valid_specificity, valid_alarm_sen, valid_alarm_acc, valid_accuracy, PPV, NPV, F1 = calculate_metrics(
            true_labels_flat, predicted_probs_flat, best_threshold)

        if self.is_print: print(name)
        if self.is_print: print("Confusion Matrix:")
        if self.is_print: print(cm)
        if self.is_print: print(f"Specificity: {valid_specificity:.4f}")
        if self.is_print: print(f"Sensitivity: {valid_alarm_sen:.4f}")
        if self.is_print: print(f"Alarm Accuracy: {valid_alarm_acc:.4f}")
        if self.is_print: print(f"Accuracy: {valid_accuracy:.4f}")

        # 绘制混淆矩阵
        if self.root_dir is None:
            plot_confusion_matrix(self.model_name, name, epoch, cm, classes=['Survive', 'Death'])
        else:
            plot_confusion_matrix(self.model_name, name, epoch, cm, classes=['Survive', 'Death'],
                                  root_dir=self.root_dir)

        return valid_accuracy, valid_specificity, valid_alarm_sen, valid_alarm_acc, cm, PPV, NPV, F1


def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def plot_roc(true_labels_flat_train, predicted_probs_flat_train,
             true_labels_flat_valid, predicted_probs_flat_valid, root_dir):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    # 计算每个数据集的FPR, TPR和AUC
    fpr_train, tpr_train, _ = roc_curve(true_labels_flat_train, predicted_probs_flat_train)
    roc_auc_train = auc(fpr_train, tpr_train)

    fpr_valid, tpr_valid, _ = roc_curve(true_labels_flat_valid, predicted_probs_flat_valid)
    roc_auc_valid = auc(fpr_valid, tpr_valid)

    # 绘制ROC曲线
    plt.figure()

    plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Train ROC curve (area = {roc_auc_train:.2f})')
    plt.plot(fpr_valid, tpr_valid, color='green', lw=2, label=f'Validation ROC curve (area = {roc_auc_valid:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC of {root_dir}')
    plt.legend(loc="lower right")

    plt.savefig(f"{root_dir}/ROC.png")
    plt.show()


def plot_prc(true_labels_flat_train, predicted_probs_flat_train,
             true_labels_flat_valid, predicted_probs_flat_valid, root_dir):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    # 计算每个数据集的Precision, Recall和Average Precision
    precision_train, recall_train, _ = precision_recall_curve(true_labels_flat_train, predicted_probs_flat_train)
    avg_precision_train = average_precision_score(true_labels_flat_train, predicted_probs_flat_train)

    precision_valid, recall_valid, _ = precision_recall_curve(true_labels_flat_valid, predicted_probs_flat_valid)
    avg_precision_valid = average_precision_score(true_labels_flat_valid, predicted_probs_flat_valid)

    # 绘制PRC曲线
    plt.figure()

    plt.plot(recall_train, precision_train, color='blue', lw=2, label=f'Train PRC (area = {avg_precision_train:.2f})')
    plt.plot(recall_valid, precision_valid, color='green', lw=2, label=f'Validation PRC (area = {avg_precision_valid:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PRC of {root_dir}')
    plt.legend(loc="lower left")

    plt.savefig(f"{root_dir}/PRC.png")
    plt.show()


def Zhongda_test_model(data_direction, observe_window, predict_window, model, epoch):
    root_dir = f'use_{observe_window}_predict_{predict_window}_{model.__class__.__name__}_{epoch}'
    with open(data_direction, 'rb') as f:
        data_split_dict = pickle.load(f)
    print(f'root_dir: {root_dir}')
    BATCH_SIZE = 256
    SEED = 42
    device = 'cuda'

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    info_train, true_labels_flat_train, predicted_probs_flat_train = TestModel(f'{root_dir}_train', observe_window, predict_window,
                                                                               model, device, epoch, data_split_dict['train_data'].drop(data_split_dict['train_data'].columns[0], axis=1),
                                                                               root_dir=root_dir, is_print=False).test()
    data_split_dict['train_data']['predicted_probs'] = predicted_probs_flat_train
    train_data_with_label = data_split_dict['train_data']
    info_valid, true_labels_flat_valid, predicted_probs_flat_valid = TestModel(f'{root_dir}_valid', observe_window, predict_window,
                                                                               model, device, epoch, data_split_dict['valid_data'].drop(data_split_dict['valid_data'].columns[0], axis=1),
                                                                               root_dir=root_dir, is_print=False).test()
    data_split_dict['valid_data']['predicted_probs'] = predicted_probs_flat_valid
    valid_data_with_label = data_split_dict['valid_data']


    plot_roc(true_labels_flat_train, predicted_probs_flat_train,
             true_labels_flat_valid, predicted_probs_flat_valid, root_dir)
    plot_prc(true_labels_flat_train, predicted_probs_flat_train,
             true_labels_flat_valid, predicted_probs_flat_valid, root_dir)
    with open(f'../Analysis/metrics_info/{model.__class__.__name__}_info_train_data.json', 'w') as json_file:
        json.dump(info_train, json_file, ensure_ascii=False, indent=4, default=convert_to_serializable)
    with open(f'../Analysis/metrics_info/{model.__class__.__name__}_info_valid_data.json', 'w') as json_file:
        json.dump(info_valid, json_file, ensure_ascii=False, indent=4, default=convert_to_serializable)
    train_data_with_label.to_csv(f'../Analysis/metrics_info/{model.__class__.__name__}_train_data_with_label.csv', index=False)
    valid_data_with_label.to_csv(f'../Analysis/metrics_info/{model.__class__.__name__}_valid_data_with_label.csv', index=False)

    print(f'finish {root_dir}')


if __name__ == '__main__':
    # model = torch.load('model_direction')
    # Zhongda_test_model(TIME_STPE, i, model, epoch)
    model = torch.load('../select_model/Zhongda_data/zzz_saved_model/use_20_predict_24_BiLSTM_BN_3layers_model_undersample_FocalLoss_50_5e-06_model_33.pth')
    data_direction = '../Analysis/data_split_dict_0724.pkl'
    Zhongda_test_model(data_direction, 20, 24, model, 33)
