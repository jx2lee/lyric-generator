import tensorflow as tf
import numpy as np
import pickle
from core.model import generator_graph
from core.var import *

def import_data(path):
    
    with open(path+'dictionary_word.pkl', 'rb') as p:
        dictionary = pickle.load(p)
    with open(path+'test_df.pkl', 'rb') as p:
        test = pickle.load(p)
        
    return dictionary, test


def generate_lyric(df, checkpoint_path):
    
    # import graph (== train graph)
    graph = generator_graph(VOCAB_SIZE, EMBEDDING_SIZE, HIDDENUNITS, BATCH_SIZE, KEEP_PROB)
    saver = graph['saver']
    decoder_prediction = graph['decoder_prediction']
    
    # predict next measures
    test = df.next_batch(1)
    encoder_input = test[0]
    decoder_input = test[1]

    with tf.Session() as sess:
        saved_path = tf.train.latest_checkpoint(checkpoint_path)
        saver.restore(sess, saved_path)
        print(saved_path)
        
        result = []
        for _ in range(10):

            prediction_index = np.zeros(0).astype(int)
            generate_text_index = sess.run(decoder_prediction, feed_dict={graph['encoder_inputs'] : encoder_input,
                                                                          graph['decoder_inputs'] : decoder_input})
            
            result.append(generate_text_index)

            # create next batch df
            test = df.next_batch(1)
            
            # setting new input
            encoder_input = test[0]
            decoder_input = test[1]
            
            
    return result


def print_lyric(result, dic):
    
    for i in range(len(result)):
        print([dic[x] for x in result[i][0].tolist() if x in dic.keys()])

