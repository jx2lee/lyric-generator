from random import randint
import numpy as np
import os, timeit
import pandas as pd
import pickle
import tensorflow as tf


def reset_graph():
    """
    Reset tensorflow graph
    """
    if 'sess' in globals() and sess:
        sess.close()
    tf.compat.v1.reset_default_graph


def check_latest_point(ckpt_path):
    """
    Check latest model point
    :param ckpt_path: checkpoint path
    :return: max value
    """
    f = []
    res = []
    for _, dirs, files in os.walk(ckpt_path):
        f = files
    for i in range(len(f)):
        if len(f[i].split('.')) >= 2:
            if f[i].split('.')[1].split('-')[-1] not in res:
                res.append(int(f[i].split('.')[1].split('-')[-1]))
    return max(res)


def generator_graph(vocab_size, embedding_size, hidden_units, batch_size, keep_prob):
    """
    Generate Model
    :param vocab_size: vocabulary size
    :param embedding_size: embedding size
    :param hidden_units: hidden unit size
    :param batch_size: batch size
    :param keep_prob: dropout rate
    :return: model graph
    """
    reset_graph()
    # Placeholders
    encoder_inputs = tf.compat.v1.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
    decoder_targets = tf.compat.v1.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
    decoder_inputs = tf.compat.v1.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
    keep_prob = tf.compat.v1.placeholder_with_default(1.0, [])

    # Embedding-layer
    embeddings = tf.Variable(tf.random.uniform([vocab_size, embedding_size], -1.0, 1.0), dtype=tf.float32)

    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
    decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

    # Encoder-RNN
    encoder_cell = tf.contrib.rnn.LSTMCell(hidden_units)
    _, encoder_final_state = tf.nn.dynamic_rnn(
        encoder_cell, encoder_inputs_embedded, time_major=True,
        dtype=tf.float32)

    # Decoder-RNN
    decoder_cell = tf.contrib.rnn.LSTMCell(hidden_units)
    decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell, output_keep_prob=keep_prob)
    decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
        decoder_cell, decoder_inputs_embedded,
        initial_state=encoder_final_state,
        dtype=tf.float32, time_major=True, scope="plain_decoder")

    # Dropout
    decoder_outputs = tf.nn.dropout(decoder_outputs, keep_prob)

    # Prediction
    decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
    decoder_prediction = tf.argmax(decoder_logits, 2)

    # Loss
    """cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(decoder_targets,depth=vocab_size, 
                                                            dtype=tf.float32),logits=decoder_logits)"""
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(decoder_targets, depth=vocab_size,
                                                                                 dtype=tf.float32),
                                                               logits=decoder_logits)
    loss = tf.reduce_mean(cross_entropy)
    # train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
    train_op = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(loss)

    # Model save
    # saver = tf.train.Saver()
    saver = tf.compat.v1.train.Saver()

    return {
        'encoder_inputs': encoder_inputs,
        'decoder_inputs': decoder_inputs,
        'decoder_targets': decoder_targets,
        'dropout': keep_prob,
        'decoder_logits': decoder_logits,
        'decoder_prediction': decoder_prediction,
        'loss': loss,
        'to': train_op,
        'saver': saver
    }


def train(graph, tr, te, batch_size, num_epochs, checkpoint_path, pkl_name, additional_train=True):
    """
    Train model
    :param graph: generated model
    :param tr: train data
    :param te: test data
    :param batch_size: batch size
    :param num_epochs: epochs
    :param checkpoint_path: checkpoint path
    :param pkl_name: pickle name
    :param additional_train: continue train
    :return: result dataframe for loss
    """
    # additional training
    if additional_train:
        saver = graph['saver']
        saved_path = tf.train.latest_checkpoint(checkpoint_path)
        latest_path = saved_path.split('-')[0] + '-' + str(check_latest_point(checkpoint_path))
        saver.restore(sess, latest_path)
        print('latest epoch : {}'.format(check_latest_point(checkpoint_path)))
        current_epoch = check_latest_point(checkpoint_path)
        tr.epochs = int(saved_path.split('-')[-1])
    # train !
    with tf.compat.v1.Session() as sess:
        if not additional_train:
            sess.run(tf.compat.v1.global_variables_initializer())
        saver = graph['saver']
        step, losses = 0, 0
        tr_loss_track, te_loss_track = [], []
        current_epoch = 0
        while current_epoch < num_epochs:
            step += 1
            tr_batch = tr.next_batch(batch_size)
            feed = {graph['encoder_inputs']: tr_batch[0], graph['decoder_inputs']: tr_batch[1],
                    graph['decoder_targets']: tr_batch[2], graph['dropout']: 0.6}
            _, tr_batch_loss = sess.run([graph['to'], graph['loss']], feed_dict=feed)
            losses += tr_batch_loss

            if step % 500 == 0:
                print('current epoch : {} \t step : {}'.format(current_epoch, step))

            if tr.epochs > current_epoch:
                current_epoch += 1
                tr_loss_track.append(losses / step)
                with open('res/tr_loss_track.pkl', 'wb') as p:
                    pickle.dump(tr_loss_track, p)

                step, losses = 0, 0

                # model save
                saver.save(sess, checkpoint_path + 'generator_model.ckpt', global_step=current_epoch)

                # model test using test_data
                te_epoch = te.epochs
                while te.epochs == te_epoch:
                    step += 1
                    te_batch = te.next_batch(batch_size)
                    feed = {graph['encoder_inputs']: te_batch[0], graph['decoder_inputs']: te_batch[1],
                            graph['decoder_targets']: te_batch[2]}
                    te_batch_loss = sess.run(graph['loss'], feed_dict=feed)
                    losses += te_batch_loss

                te_loss_track.append(losses / step)
                with open('res/te_loss_track.pkl', 'wb') as p:
                    pickle.dump(te_loss_track, p)

                step, losses = 0, 0
                print("tr_losses : ", tr_loss_track[-1], "- te_losses : ", te_loss_track[-1])

    # result.df
    result_dic = {'tr_losses': tr_loss_track, 'te_losses': te_loss_track}
    result_df = pd.DataFrame(result_dic, columns=['tr_losses', 'te_losses'])

    # save df
    with open('res' + pkl_name, 'wb') as p:
        pickle.dump(result_df, p)
    return result_df
