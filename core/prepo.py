#-*- coding:utf-8 -*-
import re
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def make_dict(txtpath, dicPath):
    with open(txtpath+'measures.txt', 'r', encoding='utf-8') as f:
        mea = f.read()

    mea = re.sub('\n', ' ', mea)
    mea = re.sub('[-=.,)(#/?:^~!$}0-9]', '', mea)
    mea = re.sub('[^ \u3131-\u3163\uac00-\ud7a3]+', '', mea)
    mea = mea.replace('\\( x2 \\)', '')
    mea = mea.replace('\\', '')
    mea = mea.split(' ')
    mea = [word for word in mea if word]
    print('\033[31m \033[43m' + '[INFO]'+ '\033[0m' + ' Duplicated measure : {}'.format(len(mea)))
    
    unqMea = []
    for word in tqdm(mea):
        if word not in unqMea:
            unqMea.append(word)
    print('\033[31m \033[43m' + '[INFO]'+ '\033[0m' + ' Create unique measure')
    print('\033[31m \033[43m' + '[INFO]'+ '\033[0m' + ' Unique measure : {}'.format(len(unqMea)))

    wordsCount={}
    for word in mea:
        if word in wordsCount:
            wordsCount[word] += 1
        else:
            wordsCount[word] = 1

    sortedWordsCount = sorted([(k,v) for k,v in wordsCount.items()], 
                               key=lambda wordCount: -wordCount[1])
    labelWord = {i+2 : ch[0] for i, ch in enumerate(sortedWordsCount)}
    wordLabel = {y:x for x,y in labelWord.items()}

    with open(dicPath+'dictionary.pkl', 'wb') as p:
            pickle.dump(labelWord, p)
            print('\033[31m \033[43m' + '[INFO]'+ '\033[0m' + ' Save dictionary')

    return wordLabel


def load_data(label, txtpath):
    with open(txtpath+'measures.txt', 'r', encoding='utf-8') as f:
        meas = f.readlines()

    for i in tqdm(range(len(meas))):
        meas[i] = re.sub('\n', '', meas[i])
        meas[i] = meas[i].split(' ')
    
    dt = [[label[w] for w in sent if w in label.keys()] for sent in meas]
    
    seqX = []
    seqy = []
    
    for i in range(len(dt)-1):
        if dt[i] != []:

            seqX.append(dt[i])
            seqy.append(dt[i+1])

    Xtrain, Xtest, ytrain, ytest = train_test_split(seqX, seqy, test_size = 0.3)
    return Xtrain, Xtest, ytrain, ytest


def array_to_df(array1, array2):
    encX = []
    decX = []
    decy = []
    seqLen = []
    
    for i in range(len(array1)):
        # create for decoder input and output
        if array1[i] != [] and array2[i] != [] and len(array1[i]) >= 4 and len(array2[i]) >=4:
            sentence_dec = array2[i][:]
            sentence_dec.insert(0, int(1))
            sentence_dec = np.append(sentence_dec, int(1))
            
            encX.append(array1[i])
            decX.append(sentence_dec[:-1])
            decy.append(sentence_dec[1:])
            seqLen.append(max([len(array1[i]), len(array2[i])+1]))
        else:
            continue

    dic = {'encX' : encX, 'decX' : decX, 'decy' : decy, 'seqLength' : seqLen}
    df = pd.DataFrame(dic, columns=['encX', 'decX', 'decy', 'seqLength'])
    df = df[df.seqLength != 0]
    df = df[df.seqLength <= 12]
    df = df.reset_index(drop = True)
    return df


class Data():
    def __init__(self, df):
        self.df = df
        self.size = len(self.df)
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor+n-1 > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.iloc[self.cursor:self.cursor+n-1]
        self.cursor += n
        return res['encX'], res['decX'], res['ency'], res['seqLength']
    

class PaddedData(Data):
    def next_batch(self, n):
        if self.cursor+n > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.iloc[self.cursor:self.cursor+n]
        self.cursor += n
        
        # Pad sequences with 0s so they are all the same length
        max_len = max(res['seqLength'])
        
        x = np.zeros([n, max_len], dtype=np.int32)
        y = np.zeros([n, max_len], dtype=np.int32)
        z = np.zeros([n, max_len], dtype=np.int32)

        for i, x_i in enumerate(x):
            x_i[:len(res['encX'].values[i])] = res['encX'].values[i]
        for i, y_i in enumerate(y):
            y_i[:len(res['decX'].values[i])] = res['decX'].values[i]
        for i, z_i in enumerate(z):
            z_i[:len(res['decy'].values[i])] = res['decy'].values[i]
        
        return x, y, z, res['seqLength']
