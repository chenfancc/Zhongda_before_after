import pickle
from tqdm import tqdm
import numpy as np
import json

if __name__ == '__main__':
    model_names = [
                   'use_20_predict_24_BiLSTM_BN_3layers_30',
                   'use_20_predict_24_GRU_BN_29',
                   'use_20_predict_24_RNN_BN_30',
                   'use_20_predict_24_GRU_BN_ResBlock_6',
                   'Random Forest',
                   'SVM',
                   'Logistic Regression',
                   'MLP',
                   'GaussianNB',
                   'XGBoost',
                   'lightGBM',
                   'random',
                   'EWS',
                   'NEWS',
                   'MEWS',
                   'SPTTS'
    ]

    for dataset_type in ['train_data', 'valid_data']:
        for model_name in model_names:
            with open(f'details/details_{model_name}_{dataset_type}.pkl', 'rb') as file:
                details = pickle.load(file)

            machp_list = []

            thresholds = np.arange(0, 1, 0.001)
            for threshold in tqdm(thresholds, desc=f'Calculating MACHP for {model_name} in {dataset_type}'):
                tachp = []
                for i in range(len(details)):
                    # print(details[i])
                    num_1 = sum(1 for item in details[i] if item > threshold)
                    num_total = sum(1 for item in details[i] if item > -1)
                    tachp.append(num_1 / num_total)
                # print(tachp)
                machp = sum(tachp) / len(tachp)
                # print(machp)
                machp_list.append(machp)

            with open(f'MACHP/list/machp_list_{model_name}_{dataset_type}.pkl', 'wb') as file:
                pickle.dump(machp_list, file)

            from matplotlib import pyplot as plt

            plt.plot(thresholds, machp_list, label='MACHP')

            # 添加标题和标签
            plt.title(f'{model_name}: \n'
                      f'Thresholds vs MACHP in {dataset_type}')
            plt.xlabel('Threshold')
            plt.ylabel('Values')
            plt.legend()

            plt.savefig(f'MACHP/figure/Thresholds_vs_MACHP_in_{model_name}_{dataset_type}.png')
            plt.close()


            with open(f'info/{model_name}_info_{dataset_type}.json', 'r',
                      encoding='utf-8') as json_file:
                info_test_loaded = json.load(json_file)


            plt.plot(machp_list, info_test_loaded['alarm_sen_list'], label='MACHP_vs_alarm_sen')

            # 添加标题和标签
            plt.title(f'{model_name}: \n'
                      f'MACHP vs alarm_sen in {dataset_type}')

            plt.xlabel('MACHP')
            plt.ylabel('alarm_sen')
            plt.legend()
            plt.savefig(f'MACHP/vs_alarm_sen/MACHP_vs_alarm_sen_{model_name}_{dataset_type}.png')
            plt.close()
