import json
import os
import pickle
import random
import numpy as np
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    df = pd.DataFrame()
    while True:
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
                       'EWS',
                       'NEWS',
                       'MEWS',
                       'SPTTS',
                       'random']
        model_name_dict = {'use_20_predict_24_BiLSTM_BN_3layers_30': 'BiLSTM',
                           'use_20_predict_24_GRU_BN_29': 'GRU',
                           'use_20_predict_24_RNN_BN_30': 'RNN',
                           'use_20_predict_24_GRU_BN_ResBlock_6': 'GRU_ResBlock',
                           'Random Forest': 'Random Forest',
                           'SVM': 'SVM',
                           'Logistic Regression': 'Logistic Regression',
                           'MLP': 'MLP',
                           'GaussianNB': 'GaussianNB',
                           'XGBoost': 'XGBoost',
                           'lightGBM': 'lightGBM',
                           'EWS': 'EWS',
                           'NEWS': 'NEWS',
                           'MEWS': 'MEWS',
                           'SPTTS': 'SPTTS',
                           'random': 'random'}

        colors_list = ['#1890FF', '#40A9FF', '#26C9C3', '#45DAD1',
                       '#73D13D', '#95DE64', '#BAE637', '#FBE139',
                       '#FFD666', '#FFA940', '#F88D48', '#DD762A',
                       '#E65E67', '#F273B5', '#9F69E2', '#6682F5']
        colors = {
            'use_20_predict_24_BiLSTM_BN_3layers_30': colors_list[8],
            'use_20_predict_24_GRU_BN_29':            colors_list[10],
            'use_20_predict_24_RNN_BN_30':            colors_list[12],
            'use_20_predict_24_GRU_BN_ResBlock_6':    colors_list[14],
            'Random Forest':                          colors_list[0],
            'SVM':                                    colors_list[2],
            'Logistic Regression':                    colors_list[4],
            'MLP':                                    colors_list[6],
            'GaussianNB':                             colors_list[1],
            'XGBoost':                                colors_list[3],
            'lightGBM':                               colors_list[5],
            'EWS':                                    colors_list[7],
            'NEWS':                                   colors_list[9],
            'MEWS':                                   colors_list[11],
            'SPTTS':                                  colors_list[13],
            'random':                                 colors_list[15]
            }

        for dataset_type in ['train_data', 'test_data', 'val_data']:
            if dataset_type != 'train_data':
                plt.figure(figsize=(10, 6))
                for model_name in model_names:
                    file_path = f'info/{model_name}_info_{dataset_type}.json'
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        try:
                            with open(file_path, 'r',
                                      encoding='utf-8') as json_file:
                                info_test_loaded = json.load(json_file)
                                # print(f"Successfully loaded info_json for model {model_name} and dataset {dataset_type}")
                        except Exception as e:
                            print(f"Error loading machp_list: {e}")
                    else:
                        print(f"File not found or is empty: {file_path}")
                        continue

                    sen_list = info_test_loaded['alarm_sen_list']

                    file_path = f'MACHP/list/machp_list_{model_name}_{dataset_type}.pkl'

                    # 检查文件是否存在且大小大于零
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        try:
                            with open(file_path, 'rb') as file:
                                machp_list = pickle.load(file)
                            # print(f"Successfully loaded machp_list for model {model_name} and dataset {dataset_type}")

                            if model_name == 'random':
                                plt.plot(machp_list, info_test_loaded['alarm_sen_list'],
                                         label=f'{model_name_dict[model_name]}', color='black', linestyle='--')
                            elif model_name != 'SPTTS':
                                plt.plot(machp_list, info_test_loaded['alarm_sen_list'], label=f'{model_name_dict[model_name]}', color=colors[model_name])
                            else:
                                plt.plot(machp_list[0], info_test_loaded['alarm_sen_list'][0], 'x', label=f'{model_name}', color=colors[model_name])

                            df[f'{model_name_dict[model_name]}_machp_list_{dataset_type}'] = machp_list
                            df[f'{model_name_dict[model_name]}_alarm_sen_list_{dataset_type}'] = info_test_loaded['alarm_sen_list']
                            # 添加标题和标签
                            plt.title(f'MACHP vs alarm_sen in {dataset_type}')

                            plt.xlim(-0.05, 0.5)
                            plt.grid(True)

                            plt.xlabel('MACHP')
                            plt.ylabel('alarm_sen')
                            # 添加图例并将其放在图外
                            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                            # 调整图形布局，确保图例不遮挡图形内容
                            plt.subplots_adjust(right=0.6)

                        except Exception as e:
                            print(f"Error loading machp_list: {e}")
                    else:
                        print(f"File not found or is empty: {file_path}")



                plt.savefig(f'MACHP/figure/comparison_MACHP_vs_alarm_sen_{dataset_type}.png')
                plt.savefig(f'MACHP/figure/comparison_MACHP_vs_alarm_sen_{dataset_type}.svg')
                plt.close()

        current_time = datetime.now()
        print(f"Current time: {current_time}")
        df.to_csv(f'MACHP/vs_alarm_sen/comparison_MACHP_vs_alarm_sen.csv', index=False)
        break
        time.sleep(60)
