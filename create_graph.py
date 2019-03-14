# -*- coding: utf-8 -*
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np

def weight_variable(shape, name=None):
    return np.sqrt(2.0 / shape[0]) * np.random.normal(size=shape)

def main():
    num_layer = 3
    input_dim = 784
    n_hidden = 200
    num_classes = 10
    dropout = 0.2

    # Create a model
    model = Sequential()

    # Input Layer
    model.add(InputLayer(input_shape=(input_dim,), name='input'))
    model.add(Dense(n_hidden, kernel_initializer=weight_variable))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    # Hidden Layers
    for i in range(num_layer):
        model.add(Dense(n_hidden, kernel_initializer=weight_variable))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

    # Output Layer
    model.add(Dense(num_classes, kernel_initializer=weight_variable))
    model.add(Activation('softmax', name='output'))

    model.summary()

    x = tf.placeholder(tf.float32, shape=[None, input_dim], name='image')
    y = model(x)
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='target')

    loss = tf.losses.mean_squared_error(y, y_)
    optimizer = tf.train.AdamOptimizer()

    train_op = optimizer.minimize(loss, name='train')

    init = tf.global_variables_initializer()

    saver_def = tf.train.Saver().as_saver_def()

    # Export Computation Graph
    tf.train.Saver().export_meta_graph('model.meta')

    with open('model.pb', 'wb') as f:
        f.write(tf.get_default_graph().as_graph_def().SerializeToString())

#    for n in tf.get_default_graph().as_graph_def().node:
#        print(n.name)

    print('Run this operation to initialize variables     : ', init.name)
    print('Run this operation for a train step            : ', train_op.name)
    print('Feed this tensor to set the checkpoint filename: ', saver_def.filename_tensor_name)
    print('Run this operation to save a checkpoint        : ', saver_def.save_tensor_name)
    print('Run this operation to restore a checkpoint     : ', saver_def.restore_op_name)



if __name__ == '__main__':
    main()

