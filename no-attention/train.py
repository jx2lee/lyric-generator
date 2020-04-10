import pickle
from core.prepo import *
from core.model import *
from core.var import *

# load train/test
with open(PKL_PATH + 'train_df.pkl', 'rb') as p:
    tr = pickle.load(p)
with open(PKL_PATH + 'test_df.pkl', 'rb') as p:
    te = pickle.load(p)

# make graph & train
g = generator_graph(VOCAB_SIZE, EMBEDDING_SIZE, HIDDENUNITS, BATCH_SIZE, KEEP_PROB)
df = train(g, tr, te, BATCH_SIZE, EPOCHS, CHECKPOINTS_PATH, '/loss_df.pkl', '/learn_time.pkl', additional_train = False)