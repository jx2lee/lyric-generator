#!/usr/bin/env python
from core.common import *
from core.model import *
from core.prepo import *
from core.var import *
import pickle
import sys


if __name__ == '__main__':
    # load train/test dataset.
    with open(PKL_PATH + 'train.pkl', 'rb') as p:
        tr = pickle.load(p)
    with open(PKL_PATH + 'test.pkl', 'rb') as p:
        te = pickle.load(p)

    # generate model
    g = generator_graph(VOCAB_SIZE, EMBEDDING_SIZE, HIDDENUNITS, BATCH_SIZE, KEEP_PROB) 
    
    # first or nex training
    train_type = sys.argv[1]
    if train_type == 'first':
        df = train(g, tr, te, BATCH_SIZE, EPOCHS, CHECKPOINTS_PATH, '/loss.pkl', additional_train = False)
        print_info('Training finished')

    elif train_type == 'next':
        df = train(g, tr, te, BATCH_SIZE, EPOCHS, CHECKPOINTS_PATH, '/loss.pkl')
        print_info('Training finished')

    else:
        print_error('You must put "first" or "next"')
