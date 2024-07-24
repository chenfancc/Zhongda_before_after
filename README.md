# Clone the code and prepare the environment:
```
git clone https://github.com/chenfancc/Zhongda_before_after.git
cd Zhongda_before_after

conda create -n zhongda_env python=3.11
conda activate zhongda_env

pip install -r requirements.txt
```
# Introduce the file structure:
## ***WARNING: NO NEED, NO MODIFY***
+ [main.py](main.py): the training code of the model.
  + `class main_trainer_factory()`: the factory class to create the trainer object.
    + you can adjust the hyperparameters at the beginning of the `main_trainier_factory()` class
  + `def model_train()`: the function to train the model.
    + you can adjust the `observe_window` and `predict_window` in the `for` loop of the `model_train()` function.
    + `tensor_direction`: the direction of the tensor, mainly used in the `model_train()` function. The default direction is `f'生成tensor/mice_mmscaler_use_{observe_window}_predict_{predict_window}.pth'
    + `root_dir`: the direction restored all the results
    + `name`: mainly used for naming the results.
    + `SAMPLE_METHOD`: the method used to sample the data, you can use 'origin'(original data), 'undersample', 'oversample', 'smote'. According to the previous experiments, 'undersample' is the best method.
    + `model`: the model used to train the data. The model for training is in the [model.py](related_function/model.py)
  + `def select_model()`: used to select the certain model:
    + `observe_window` and `predict_window`: the window size of the data used to train the model.
    + `SAMPLE_METHOD`: the method used to sample the data.
    + `model`: the model used to train the data. 
    + `epoch`: the number of epochs used to train the model.
    + NOTICE: all the parameters used for re-training the certain model should be manually set at the beginning of the `class model_trainer_factory()`
+ [related_function](related_function): the related functions used in the [main.py](main.py).
  + [function.py](related_function/function.py): restored the used functions
    + `def plot_confusion_matrix()`: This function prints and plots the confusion matrix.
    + `def calculate_metrics()`
    + `def main_data_loader()`: load the data used to train the model.
    + `def plot_info()`: plot the information of the metrics list.
    + `class FocalLoss()`: the focal loss used in the model.
  + [kan_model.py](related_function/kan_model.py): KAN block
  + [model.py](related_function/model.py): the model used to train the data.
  + [model_trainer.py](related_function/model_trainer.py): the trainer class used to train the model.
  + [sample.py](related_function/sample.py): the sample method used to sample the data.
  + [xlstm.py](related_function/xlstm.py): xlstm block
+ [生成tensor](生成tensor): the generated tensor used to train the model.
# Work flow:
## ***WARNING: NO NEED, NO MODIFY***
## Train the model:
1. Run [main.py](main.py): 
  + set the hyperparameters at the beginning of the `main_trainier_factory()` class.
  + run code: 
    + `Trainer = model_trainer_factory()`
    + `Trainer.model_train()`
  + You will get: `{root_dir}` directory
```text
Zhongda_before_after
├── {root_dir}
│   ├── use_6_predict_6_BiLSTM_BN_larger_model_undersample_FocalLoss_50_5e-06
│   │   ├── 00_hyperparameters.json # the hyperparameters used for training the model
│   │   ├── 00_info.json # the information of the metrics list
│   │   ├── 00_use_6_predict_6_BiLSTM_BN_larger_model_undersample_FocalLoss_50_5e-06_final_model.pth # the final model
│   │   ├── 01_total_loss.png # the total loss curve
│   │   ├── 02_loss_curve.png # the loss curve of each epoch
│   │   ├── 03_model_performance_auc.png # the metrics at the best Youden Index of each epoch
│   │   ├── 04_mdoel_performance_prc.png # the metrics at the best precision-recall curve of each epoch
│   │   ├── 05_auc_cureve.png # the auc curve of each epoch
│   │   ├── 06_prc_curve.png # the precision-recall curve of each epoch
│   │   ├── 06_roc_curve.png # the roc curve of each epoch
│   │   ├── 07_brier_score.png # the brier score of each epoch
│   │   ├── ... # Confusion matrix, ROC curve, PRC curve of each epoch
│   ├── ... # other models
```
2. Select the model you want: 
   1. Run [select_model.ipynb](select_model/select_model.ipynb), `auc_reslts.csv` was generated. Select the model you wanted
   2. For example, 20th and 27th epoch of `use_6_predict_6_BiLSTM_BN_larger_model_undersample_FocalLoss_50_5e-06`
   3. Set the hyperparameters at the beginning of the `model_trainer_factory()` class. 
      + The hyperparameters used for training such model were saved in the `00_hyperparameters.json` file.
   4. Run code [main.py](main.py): 
      + `Trainer = model_trainer_factory()`
      + `Trainer.select_model(6, 6, "undersample", BiLSTM_BN_larger, [20, 27])`
   5. Selected model will be saved in the `f'select_model/{root_dir}/zzz_saved_model'` folder. 
## Evaluate the model:
1. Run [test_model.py](test_model/test_model.py) --> `test/use_6_predict_6_BiLSTM_BN_larger_20` --> `info_train.json` and `info_valid.json`
2. bbb