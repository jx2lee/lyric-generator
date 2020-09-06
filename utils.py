#!/usr/bin/env python
from core.prepo import *
import os
import pickle

DATA_PATH = 'tmp/'

if __name__ == "__main__":
    if 'dictionary.pkl' in os.listdir(DATA_PATH):
        print('\033[31m \033[43m' + '[INFO]'+ '\033[0m' + ' Dictionary already exited')
        with open(DATA_PATH+'dictionary.pkl', 'rb') as p:
            dictionary = pickle.load(p)
            dictionary = {y:x for x, y in dictionary.items()}
    else:
        dictionary = make_dict(DATA_PATH, DATA_PATH)
    
    X_train, X_test, y_train, y_test = load_data(dictionary, DATA_PATH)
    df_train = PaddedData(array_to_df(X_train, y_train))
    df_test = PaddedData(array_to_df(X_test, y_test))
    
    with open(DATA_PATH + 'train.pkl', 'wb') as p:
        pickle.dump(df_train, p)
    with open(DATA_PATH + 'test.pkl', 'wb') as p:
        pickle.dump(df_test, p)

    print('\033[31m \033[43m' + '[INFO]'+ '\033[0m' + ' Save model input file')
