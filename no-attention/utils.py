from core.prepo import *
import pickle
import os

# parameter
DATA_PATH = './data/'

# set dictionary
if 'dictionary_word.pkl' in os.listdir(DATA_PATH):
    print('dictionary already exited..!')
    with open(DATA_PATH+'dictionary_word.pkl', 'rb') as p:
        dictionary = pickle.load(p)
        dictionary = {y:x for x, y in dictionary.items()}
else:
    dictionary = makeDic(DATA_PATH, DATA_PATH)

# train / test
X_train, X_test, y_train, y_test = loadData(dictionary, DATA_PATH)

df_train = PaddedData(arraytoDf(X_train, y_train))
df_test = PaddedData(arraytoDf(X_test, y_test))

# save
with open(DATA_PATH + 'train_df.pkl', 'wb') as p:
    pickle.dump(df_train, p)
with open(DATA_PATH + 'test_df.pkl', 'wb') as p:
    pickle.dump(df_test, p)
print('finished...!')
