#-*- coding:utf-8 -*-
import pickle
import tensorflow as tf
from core.model import generator_graph
from core.generator import *
from core.var import *

dictionary, test = import_data(PKL_PATH)
prediction = generate_lyric(test, CHECKPOINTS_PATH)
print_lyric(prediction, dictionary)