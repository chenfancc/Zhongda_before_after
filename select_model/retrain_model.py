from main_delta import model_trainer_factory as Trainer_delta
from main_origin import model_trainer_factory as Trainer_origin
from related_function.model import *
def retrain_1():
    Trainer_delta().select_model(20, 24, 'delta', 'undersample', RNN_BN_3layers, [14])

def retrain_2():
    Trainer_origin().select_model(20, 24, 'origin', 'undersample', RNN_BN_3layers, [14])


if __name__ == '__main__':
    retrain_1()
    retrain_2()