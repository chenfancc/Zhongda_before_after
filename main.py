from related_function.function import *
from related_function.model_trainer import TrainModel
import numpy as np
from related_function.model import *
from related_function.function import plot_info
from datetime import datetime


class model_trainer_factory():
    BATCH_SIZE = 256
    EPOCH = 5
    LR = 5e-6
    GAMMA = 0.95
    STEP_SIZE = 20  # 每隔多少个 epoch 衰减一次学习率
    DECAY = 1e-4
    DEVICE = "cuda"
    SEED = 42
    ALPHA_LOSS = 1
    GAMMA_LOSS = 3

    hyperparameters = {
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCH": EPOCH,
        "LEARNING_RATE": LR,
        "GAMMA": GAMMA,
        "STEP_SIZE": STEP_SIZE,
        "DECAY": DECAY,
        "device": DEVICE,
        "SEED": SEED,
        "ALPHA_LOSS": ALPHA_LOSS,
        "GAMMA_LOSS": GAMMA_LOSS
    }

    def __init__(self):
        self.BATCH_SIZE = self.hyperparameters["BATCH_SIZE"]
        self.EPOCH = self.hyperparameters["EPOCH"]
        self.LR = self.hyperparameters["LEARNING_RATE"]
        self.GAMMA = self.hyperparameters["GAMMA"]
        self.STEP_SIZE = self.hyperparameters["STEP_SIZE"]
        self.DECAY = self.hyperparameters["DECAY"]
        self.DEVICE = self.hyperparameters["device"]
        self.SEED = self.hyperparameters["SEED"]
        self.ALPHA_LOSS = self.hyperparameters["ALPHA_LOSS"]
        self.GAMMA_LOSS = self.hyperparameters["GAMMA_LOSS"]
        pass

    def model_train(self):
        # input_size = observe_window
        for observe_window in [6]:
            # i：结果时间步
            for predict_window in [6]:
                tensor_direction = f'生成tensor/mice_mmscaler_use_{observe_window}_predict_{predict_window}.pth'

                root_dir = 'Results_zhongda_before_after'
                name = f'use_{observe_window}_predict_{predict_window}'
                data_process = '前后填充 + 均值方差标准化'

                for SAMPLE_METHOD in ["undersample"]:
                    # for model in [BiLSTM_BN, BiLSTM_BN_larger, BiLSTM_BN_Resnet, BiLSTM_BN_3layers,
                    #               BiLSTM_BN_4layers,
                    #               GRU_BN, GRU_BN_3layers, GRU_BN_4layers,
                    #               RNN_BN, RNN_BN_3layers, RNN_BN_4layers,
                    #               BiLSTM_BN_ResBlock, GRU_BN_ResBlock, RNN_BN_ResBlock,
                    #               BiLSTM_BN_ResBlock_3layers, GRU_BN_ResBlock_3layers, RNN_BN_ResBlock_3layers,
                    #               BiLSTM_BN_single, GRU_BN_single, RNN_BN_single]:
                    for model in [BiLSTM_BN]:
                        # if observe_window == 8 and predict_window == 6: continue

                        print(SAMPLE_METHOD, "_", model.__name__)
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间
                        print("Start Time =", current_time)

                        model_name = f"{name}_{model.__name__}_model_{SAMPLE_METHOD}_FocalLoss_{self.EPOCH}_{self.LR}"

                        print(f"\n")
                        print(
                            "==========================================模型训练开始：==========================================")
                        print(f"\n数据集：MIMIC数据\t数据处理：{data_process}\t采样方法：{SAMPLE_METHOD}")
                        print(f'模型：{model_name}')

                        # 设置随机种子
                        np.random.seed(self.SEED)
                        torch.manual_seed(self.SEED)
                        train_dataloader, val_dataloader, test_dataloader = main_data_loader(tensor_direction,
                                                                                             SAMPLE_METHOD,
                                                                                             self.BATCH_SIZE)
                        loss_f = FocalLoss(self.ALPHA_LOSS, self.GAMMA_LOSS)
                        trainer = TrainModel(model_name, model, self.hyperparameters, train_dataloader, val_dataloader,
                                             criterion_class=loss_f, root_dir=root_dir, is_print=False,
                                             model_self_name=model.__name__)
                        info = trainer.train()
                        trainer.save_model()
                        plot_info(info, model_name, root_dir=root_dir)
    def select_model(self, observe_window, predict_window, SAMPLE_METHOD, model, epoch):

        tensor_direction = f'生成tensor/mice_mmscaler_use_{observe_window}_predict_{predict_window}.pth'
        root_dir = 'select_model/Zhongda_data'
        name = f'use_{observe_window}_predict_{predict_window}'
        model_name = f"{name}_{model.__name__}_model_{SAMPLE_METHOD}_FocalLoss_{self.EPOCH}_{self.LR}"
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        train_dataloader, val_dataloader, test_dataloader = main_data_loader(tensor_direction,
                                                                             SAMPLE_METHOD,
                                                                             self.BATCH_SIZE)
        loss_f = FocalLoss(self.ALPHA_LOSS, self.GAMMA_LOSS)
        trainer = TrainModel(model_name, model, self.hyperparameters, train_dataloader, val_dataloader,
                             criterion_class=loss_f, root_dir=root_dir, is_print=True, save_model_index=epoch)
        info = trainer.train()
        trainer.save_model()
        plot_info(info, model_name, root_dir=root_dir)


if __name__ == '__main__':
    Trainer = model_trainer_factory()
    # Trainer.select_model(20, 24, "undersample", BiLSTM_BN_3layers, [30])
    # Trainer.select_model(20, 24, "undersample", GRU_BN_ResBlock, [6])
    # Trainer.select_model(20, 24, "undersample", RNN_BN, [30])
    Trainer.select_model(6, 6, "undersample", BiLSTM_BN, [2])

    # Trainer.model_train()
