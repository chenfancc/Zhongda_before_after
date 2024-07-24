import numpy as np
import os
from tqdm.autonotebook import tqdm
from related_function.function import *
from related_function.model_trainer import TrainModel
from related_function.model import *
from related_function.function import plot_info
from datetime import datetime


class model_trainer_factory():
    BATCH_SIZE = 512
    EPOCH = 50
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

    def MIMIC_train(self):
        is_print = False
        # input_size = TIME_STEP
        for observe_window in [20, 6, 8, 12, 18, 24]:
            # i：结果时间步
            for predict_window in [24, 6, 8, 12, 18, 20]:
                tensor_direction = f'生成tensor/delta/mice_mmscaler_use_{observe_window}_predict_{predict_window}_delta.pth'

                if os.path.exists(tensor_direction) and os.path.getsize(tensor_direction) > 0:
                    try:
                        root_dir = 'Results_zhongda_before_after_delta'
                        name = f'use_{observe_window}_predict_{predict_window}'
                        data_process = '前后填充 + 均值方差标准化'

                        for SAMPLE_METHOD in ["undersample"]:
                            for model in tqdm([
                                BiLSTM_BN,
                                BiLSTM_BN_larger, BiLSTM_BN_Resnet, BiLSTM_BN_3layers,
                                BiLSTM_BN_4layers,
                                GRU_BN, GRU_BN_3layers, GRU_BN_4layers,
                                RNN_BN, RNN_BN_3layers, RNN_BN_4layers,
                                BiLSTM_BN_ResBlock, GRU_BN_ResBlock, RNN_BN_ResBlock,
                                BiLSTM_BN_ResBlock_3layers, GRU_BN_ResBlock_3layers, RNN_BN_ResBlock_3layers,
                                BiLSTM_BN_single, GRU_BN_single, RNN_BN_single, sLSTM
                            ], desc=f'{name}: '):
                            # for model in [BiLSTM_BN_3layers, GRU_BN, GRU_BN_ResBlock, RNN_BN] if is_print else tqdm([BiLSTM_BN_3layers, GRU_BN, GRU_BN_ResBlock, RNN_BN]):

                                if is_print: print(SAMPLE_METHOD, "_", model.__name__)
                                if is_print: current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间
                                if is_print: print("Start Time =", current_time)

                                model_name = f"{name}_{model.__name__}_model_{SAMPLE_METHOD}_FocalLoss_{self.EPOCH}_{self.LR}"

                                info_json = f'{root_dir}/{model_name}/01_total_loss.png'
                                if os.path.exists(info_json) and os.path.getsize(info_json) > 0:
                                    print(f'{model_name} has been trained')
                                    continue

                                if is_print: print(f"\n")
                                if is_print: print(
                                    "==========================================模型训练开始：==========================================")
                                if is_print: print(
                                    f"\n数据集：MIMIC数据\t数据处理：{data_process}\t采样方法：{SAMPLE_METHOD}")
                                # print(f'模型：{model.__name__}')

                                # 设置随机种子
                                np.random.seed(self.SEED)
                                torch.manual_seed(self.SEED)
                                train_dataloader, val_dataloader = main_data_loader(tensor_direction, SAMPLE_METHOD, self.BATCH_SIZE)
                                loss_f = FocalLoss(self.ALPHA_LOSS, self.GAMMA_LOSS)
                                trainer = TrainModel(model_name, model, self.hyperparameters, train_dataloader,
                                                     val_dataloader,
                                                     criterion_class=loss_f, root_dir=root_dir, is_print=is_print,
                                                     model_self_name=model.__name__)
                                info = trainer.train()
                                trainer.save_model()
                                plot_info(info, model_name, root_dir=root_dir, is_print=is_print)
                    except Exception as e:
                        print(f"Error loading tensor: {e}")
                else:
                    print(f"File not found or is empty: {tensor_direction}")

    def select_model(self, TIME_STEP, i, SAMPLE_METHOD, model, epoch):

        tensor_direction = f'生成tensor/mice_mmscaler_origin_use_{TIME_STEP}_predict_{i}.pth'
        root_dir = 'Zhongda_data_origin'
        name = f'use_{TIME_STEP}_predict_{i}'
        model_name = f"{name}_{model.__name__}_model_{SAMPLE_METHOD}_FocalLoss_{self.EPOCH}_{self.LR}"
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        train_dataloader, val_dataloader = main_data_loader(tensor_direction, SAMPLE_METHOD, self.BATCH_SIZE)
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
    # Trainer.select_model(20, 24, "undersample", GRU_BN, [29])

    Trainer.MIMIC_train()
