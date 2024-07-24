import pickle
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import xgboost as xgb


def process_data(dropped_data, name, model_input, observe_window):
    # 获取数据的时间范围
    min_time = int(dropped_data['time_diff_hours'].min())
    max_time = int(dropped_data['time_diff_hours'].max())

    results = []

    # 对于每个时间窗口
    for start_time in tqdm(range(min_time, max_time - observe_window + 1), desc=f'{name}'):
        end_time = start_time + observe_window
        proportion_array = []

        # 对于每个患者
        ew_ids = dropped_data['EW_ID'].unique()
        filtered_data = dropped_data[
            (dropped_data['time_diff_hours'] >= start_time) &
            (dropped_data['time_diff_hours'] < end_time)
            ]
        for ew_id in ew_ids:
            # 获取特定患者在该时间窗口内的数据
            patient_data = filtered_data[(filtered_data['EW_ID'] == ew_id)]

            if patient_data.shape[0] == observe_window:
                if name == 'random':
                    patient_data = patient_data.drop(columns=['EW_ID', 'final_label', 'time_diff_hours'])
                    predictions = np.random.rand(1)
                elif name != 'XGBoost':
                    # 输入模型并获取输出
                    patient_data = patient_data.drop(columns=['EW_ID', 'final_label', 'time_diff_hours'])
                    # 输入模型并获取输出
                    model_output = model_input.predict_proba(patient_data)[:, 1] if hasattr(model_input,
                                                                                            "predict_proba") else model_input.decision_function(
                        patient_data)
                    # 根据阈值进行分类
                    predictions = model_output[0].flatten()
                else:
                    X_patient_data = patient_data.drop(columns=['EW_ID', 'final_label', 'time_diff_hours'])
                    y_patient_data = patient_data[['final_label']]
                    data_xgb = xgb.DMatrix(X_patient_data, label=y_patient_data)
                    model_output = model_input.predict(data_xgb)
                    # 根据阈值进行分类
                    predictions = model_output[0].flatten()

            else:
                # 如果没有数据，设置占比为0或其他合适的默认值
                predictions = -1

            proportion_array.append(predictions)

        results.append(proportion_array)

    return results


if __name__ == '__main__':
    observe_window = 1  # 设定观察窗口
    with open('results_model_0714.pkl', 'rb') as f:
        models = pickle.load(f)

    for data_type in ['test_data', 'val_data']:
        with open('data_split_dict_0710.pkl', 'rb') as f:
            data = pickle.load(f)
            df = data[data_type]
        #     val_test =  pd.concat([val_data, test_data], ignore_index=True)
        # df = pd.read_csv('mice_mmscaler.csv')
        # 将charttime列转换为datetime类型
        df['charttime'] = pd.to_datetime(df['charttime'])

        # 按EW_ID分组，并计算当前行与EW_ID第一行charttime的时间差
        df['time_diff_hours'] = df.groupby('EW_ID')['charttime'].transform(
            lambda x: (x - x.min()).dt.total_seconds() / 3600)
        # dropped_data = df.drop(columns=['final_label', 'charttime'])
        dropped_data = df.drop(columns=['charttime'])

        for name, model in models.items():
            if name:  # in ['SVM']:
                print(f'details_{name}_{data_type}.pkl')
                results = process_data(dropped_data, name, model, observe_window)
                with open(f'details/details_{name}_{data_type}.pkl', 'wb') as file:
                    pickle.dump(results, file)

        # name = 'random'
        # results = process_data(dropped_data, name, 'random', observe_window)
        # with open(f'details/details_{name}_{data_type}.pkl', 'wb') as file:
        #     pickle.dump(results, file)
