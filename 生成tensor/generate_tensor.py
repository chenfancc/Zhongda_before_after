import pandas as pd
from datetime import datetime
import torch
from tqdm import tqdm
import numpy as np
import json


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


def generate_label(row, x):
    if 0 < row[-3] <= x:
        return row[-4]
    else:
        return 0


import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def build_tensor_tqdm_gpt(data, predict_x, TIME_STEPS, NUM_FEATURES):
    # 将 NumPy 数组转换为 Pandas DataFrame 以利用 groupby
    df = pd.DataFrame(data, columns=['ID'] + [f'feature_{i}' for i in range(data.shape[1] - 1)])

    # 按 ID 分组，利用 Pandas 的优化性能
    grouped = df.groupby('ID')

    data_list = []
    label_list = []
    id_list = []

    for id, group in tqdm(grouped):  # 使用 tqdm 进度条显示进度
        patient_data = group.iloc[:, 1:1 + NUM_FEATURES].to_numpy().astype(float)
        label_data = np.array([generate_label(row, predict_x) for row in group.to_numpy()]).astype(int)

        patient_meta_tensor = torch.tensor(patient_data)

        for j in range(patient_data.shape[0] - TIME_STEPS + 1):
            id_list.append(id)
            data_list.append(patient_meta_tensor[j:j + TIME_STEPS, :].unsqueeze(0))

            label_meta_tensor = torch.tensor(label_data[j + TIME_STEPS - 1]).unsqueeze(0)
            label_list.append(label_meta_tensor)

    # 将列表转换为张量
    data_tensor = torch.cat(data_list, dim=0)
    label_tensor = torch.cat(label_list, dim=0)

    return data_tensor, label_tensor, id_list


# def build_tensor_tqdm_gpt(data, predict_x, TIME_STEPS, NUM_FEATURES):
#     unique_ids = np.unique(data[:, 0])
#     print(unique_ids.shape)

#     data_list = []  # 临时存储列表
#     label_list = []  # 临时标签列表
#     id_list = []

#     # 提前将数据按 ID 分组
#     data_by_id = {id: data[data[:, 0] == id] for id in tqdm(unique_ids)}

#     for id in tqdm(unique_ids):  # 添加 tqdm 进度条

#         patient_data = data_by_id[id][:, 1:1 + NUM_FEATURES].astype(float)
#         label_data = np.array([generate_label(row, predict_x) for row in data_by_id[id]]).astype(float)


#         patient_meta_tensor = torch.tensor(patient_data)  # 先将整个数组转换为张量

#         for j in range(patient_data.shape[0] - TIME_STEPS + 1):
#             id_list.append(id)

#             # 使用切片操作
#             data_list.append(patient_meta_tensor[j:j + TIME_STEPS, :].unsqueeze(0))

#             label_meta_tensor = torch.tensor(label_data[j + TIME_STEPS - 1]).unsqueeze(0)  # 直接使用torch.tensor
#             label_list.append(label_meta_tensor)

#     # 将列表转换为张量
#     data_tensor = torch.cat(data_list, dim=0)
#     label_tensor = torch.cat(label_list, dim=0)

#     return data_tensor, label_tensor, id_list


# def build_tensor_tqdm_gpt(data, predict_x, TIME_STEPS, NUM_FEATURES):
#     unique_ids = np.unique(data[:, 0])
#     print(unique_ids.shape)

#     data_list = []  # 临时存储列表
#     label_list = []  # 临时标签列表
#     id_list = []
#     count = 0
#     for _, id in enumerate(tqdm(unique_ids)):  # 添加 tqdm 进度条

#         patient_data = data[data[:, 0] == id][:, 1:1 + NUM_FEATURES].astype(float)  # 获取特征A和特征B的值
#         # label_data = data[data['EW_ID'] == id].apply(lambda row: generate_label(row, predict_x), axis=1).astype(float)
#         # label_data = data[data[:, 0] == id].apply(lambda row: generate_label(row, predict_x), axis=1).astype(float)
#         label_data = np.array([generate_label(row, predict_x) for row in data[data[:, 0] == id]]).astype(float)

#         for j in range(patient_data.shape[0] - TIME_STEPS + 1):
#             id_list.append(id)
#             patient_meta_tensor = torch.tensor(patient_data[j:j + TIME_STEPS, :]).unsqueeze(0)  # 直接使用torch.tensor
#             data_list.append(patient_meta_tensor)

#             label_meta_tensor = torch.tensor(label_data[j + TIME_STEPS - 1]).unsqueeze(0)  # 直接使用torch.tensor
#             label_list.append(label_meta_tensor)

#     # 将列表转换为张量
#     data_tensor = torch.cat(data_list, dim=0)
#     label_tensor = torch.cat(label_list, dim=0)
#     # unique_selected_ids = np.unique(id_list)

#     return data_tensor, label_tensor, id_list

def split_tensor(data, train_rate, val_rate, predict_x, TIME_STEPS, NUM_FEATURES):
    unique_ids = np.unique(data[:, 0])

    train_size = int(train_rate * len(unique_ids))
    val_size = int(val_rate * len(unique_ids))

    # 使用集合来提高查找效率
    train_id_set = set(unique_ids[:train_size])
    val_id_set = set(unique_ids[train_size:train_size + val_size])
    test_id_set = set(unique_ids[train_size + val_size:])

    # 过滤数据集
    train_data = np.array([row for row in tqdm(data) if row[0] in train_id_set])
    val_data = np.array([row for row in tqdm(data) if row[0] in val_id_set])
    test_data = np.array([row for row in tqdm(data) if row[0] in test_id_set])

    print("data has been split")

    data_tensor_test, label_tensor_test, ids_test = build_tensor_tqdm_gpt(test_data, predict_x, TIME_STEPS,
                                                                          NUM_FEATURES)
    data_tensor_train, label_tensor_train, ids_train = build_tensor_tqdm_gpt(train_data, predict_x, TIME_STEPS,
                                                                             NUM_FEATURES)
    data_tensor_val, label_tensor_val, ids_val = build_tensor_tqdm_gpt(val_data, predict_x, TIME_STEPS, NUM_FEATURES)

    return (data_tensor_train, label_tensor_train, ids_train,
            data_tensor_val, label_tensor_val, ids_val,
            data_tensor_test, label_tensor_test, ids_test)


def run_step6_label_1(data_input, file, predict_window, observe_window, NUM_FEATURES):
    # i：结果时间步
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间
    print("Start Time =", current_time)
    data_row = data_input
    data = data_row.to_numpy()

    print('数据已载入')

    # data_tensor, label_tensor = build_tensor_tqdm_gpt(data, predict_x, TIME_STEPS, NUM_FEATURES)

    (data_tensor_train, label_tensor_train, ids_train,
     data_tensor_val, label_tensor_val, ids_val,
     data_tensor_test, label_tensor_test, ids_test) = (
        split_tensor(data, 0.8, 0.1, predict_window, observe_window, NUM_FEATURES))

    torch.save({'data_tensor_train': data_tensor_train,
                'label_tensor_train': label_tensor_train,
                'data_tensor_val': data_tensor_val,
                'label_tensor_val': label_tensor_val,
                'data_tensor_test': data_tensor_test,
                'label_tensor_test': label_tensor_test,
                # 'data_tensor_cell': data_tensor,
                # 'label_tensor_cell': label_tensor
                }, file)
    ids = {
        "ids_train_data": ids_train,
        "ids_valid_data": ids_val,
        "ids_test_data": ids_test
    }
    ids = convert_to_serializable(ids)
    ids_file = f"origin/mice_mmscaler_use_{observe_window}_predict_{predict_window}_ids_origin.json"
    with open(ids_file, 'w') as f:
        json.dump(ids, f)


if __name__ == '__main__':
    NUM_FEATURES = 5
    observe_windows = [20, 24, 6, 8, 12, 18]
    predict_windows = [24, 20, 6, 8, 12, 18]

    data = pd.read_csv('../生成tensor/mice_mmscaler_nomerge_adtime_0726.csv')
    data = data.drop(data.columns[[0, 1]], axis=1)

    for observe_window in observe_windows:
        for predict_window in predict_windows:
            file = f"origin/mice_mmscaler_use_{observe_window}_predict_{predict_window}_origin.pth"
            run_step6_label_1(data, file, predict_window, observe_window, NUM_FEATURES)
            print(file)
