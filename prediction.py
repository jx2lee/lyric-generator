# -*- coding:utf-8 -*-
#!/usr/bin/env python
from core.generator import *
from core.model import generator_graph
from core.var import *
import pickle

if __name__ == '__main__':
    dictionary, test_data = import_data(PKL_PATH)
    prediction = generate_lyric(test_data, CHECKPOINTS_PATH)
    print_lyric(prediction, dictionary)
