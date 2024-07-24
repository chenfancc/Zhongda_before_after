import os
import pickle
import sys
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm

# 获取 machp_dl.py 的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 获取父目录的路径，并添加到 sys.path
related_function_dir = os.path.abspath(os.path.join(script_dir, '../../'))
sys.path.append(related_function_dir)

# 现在可以导入相关模块
from related_function import *

def normalize_input(x):
    hr_max = 300
    hr_min = 15
    rr_max = 100
    rr_min = 0
    sbp_max = 291
    sbp_min = 0
    dbp_max = 219
    dbp_min = 0
    spo2_max = 100
    spo2_min = 0
    x['HR'] = (x['HR'] - hr_min) / (hr_max - hr_min)
    x['RR'] = (x['RR'] - rr_min) / (rr_max - rr_min)
    x['SBP'] = (x['SBP'] - sbp_min) / (sbp_max - sbp_min)
    x['DBP'] = (x['DBP'] - dbp_min) / (dbp_max - dbp_min)
    x['SpO2'] = (x['SpO2'] - spo2_min) / (spo2_max - spo2_min)
    return x


def process_data(model, data, observe_window):
    # 获取数据的时间范围
    min_time = int(data['time_diff_hours'].min())
    max_time = int(data['time_diff_hours'].max())

    results = []

    # 对于每个时间窗口
    for start_time in range(min_time, max_time - observe_window):
        end_time = start_time + observe_window
        proportion_array = []

        # 对于每个患者
        ew_ids = dropped_data['EW_ID'].unique()
        patient_data_tensor = None
        # 预先筛选数据，只进行一次
        filtered_data = dropped_data[
            (dropped_data['time_diff_hours'] >= start_time) &
            (dropped_data['time_diff_hours'] < end_time)
            ]

        # 创建一个空列表来收集所有的患者数据
        all_patient_data = []

        for ew_id in tqdm(ew_ids, desc=f'start_time: {start_time}/{max_time - observe_window}'):
            # 获取特定患者在该时间窗口内的数据
            patient_data = filtered_data[filtered_data['EW_ID'] == ew_id]

            if patient_data.shape[0] == observe_window:
                # 输入模型并获取输出
                patient_data = patient_data.drop(columns=['EW_ID', 'time_diff_hours', 'final_label'])
                all_patient_data.append(patient_data.to_numpy())

        # 将所有患者的数据转换为单个 numpy 数组，然后再转换为 tensor
        if all_patient_data:
            all_patient_data_np = np.array(all_patient_data)
            patient_data_tensor = torch.tensor(all_patient_data_np)
        else:
            patient_data_tensor = None

        # 检查最终的 tensor
        if patient_data_tensor is not None and patient_data_tensor.shape[0] == 1:
            patient_data_tensor = torch.cat((patient_data_tensor, torch.tensor(all_patient_data)), dim=0)

        # 加载数据并处理模型
        patient_data_tensor = patient_data_tensor[:, :, :5]
        # patient_label_tensor 现在包含所有患者的数据，可以继续处理
        if patient_data_tensor is None:
            predictions = [-1]
        else:
            # 输入模型并获取输出
            if patient_data_tensor.shape[0] > 8000:
                patient_data_tensor_1 = patient_data_tensor[:8000]
                patient_data_tensor_2 = patient_data_tensor[8000:]
                model_output_1 = model(patient_data_tensor_1.float())
                model_output_2 = model(patient_data_tensor_2.float())
                model_output = torch.cat((model_output_1, model_output_2), 0).detach().cpu().numpy()
            else:
                model_output = model(patient_data_tensor.float()).detach().cpu().numpy()
            # 根据阈值进行分类
            predictions = model_output

        results.append(predictions)

    return results


if __name__ == '__main__':
    # Load models
    model_paths = {
        'use_20_predict_24_BiLSTM_BN_3layers_33': '../../select_model/Zhongda_data/zzz_saved_model/use_20_predict_24_BiLSTM_BN_3layers_model_undersample_FocalLoss_50_5e-06_model_33.pth'
    }
    models = {name: torch.load(path, map_location=torch.device('cuda')).eval() for name, path in model_paths.items()}

    model_names = list(models.keys())

    # Load data
    with open('../data_split_dict_0724.pkl', 'rb') as file:
        proportions = pickle.load(file)

    for type in ['train_data', 'valid_data']:
        df = proportions[type]
        df = normalize_input(df)
        df['charttime'] = pd.to_datetime(df['charttime'])
        df['time_diff_hours'] = df.groupby('EW_ID')['charttime'].transform(
            lambda x: (x - x.min()).dt.total_seconds() / 3600)
        dropped_data = df.drop(columns=['charttime'])

        observe_window = 20  # 设定观察窗口

        for model_name in model_names:
            results = process_data(models[model_name], dropped_data, observe_window)

            with open(f'details_{model_name}_{type}.pkl', 'wb') as file:
                pickle.dump(results, file)
