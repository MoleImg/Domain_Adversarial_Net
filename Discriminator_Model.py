# coding: utf-8
'''
TODO:
    Discriminator model of the domain adversarial network
    Full-convolutional network
    or any other classification model
'''
__author__ = 'MoleImg'
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K

class FCNet():
    def __init__(self, input_h, input_w, input_c, output_c, filter_size, filter_num,
                 batch_norm=True):
        self._input_h = input_h
        self._input_w = input_w
        self._input_c = input_c
        self._output_c = output_c
        self._filter_size = filter_size
        self._filter_num = filter_num
        self._batch_norm = batch_norm
        self._build_model()

    def _build_model(self):
        '''
        A full-convolutional classification network
        used as the discriminator
        to classify the data from SOURCE or TARGET domain
        '''
        # input data
        # dimension of the image depth
        inputs = layers.Input((self._input_h, self._input_w, self._input_c), dtype=tf.float32)
        axis = 3

        conv1 = layers.Conv2D(self._filter_num, (self._filter_size, self._filter_size), strides=2, padding='same')(inputs)
        if self._batch_norm is True:
            conv1 = layers.BatchNormalization(axis=axis)(conv1)
        conv1 = layers.Activation('relu')(conv1)
        conv2 = layers.Conv2D(2*self._filter_num, (self._filter_size, self._filter_size), strides=2, padding='same')(conv1)
        if self._batch_norm is True:
            conv2 = layers.BatchNormalization(axis=axis)(conv2)
        conv2 = layers.Activation('relu')(conv2)
        conv3 = layers.Conv2D(4*self._filter_num, (self._filter_size, self._filter_size), strides=2, padding='same')(conv2)
        if self._batch_norm is True:
            conv3 = layers.BatchNormalization(axis=axis)(conv3)
        conv3 = layers.Activation('relu')(conv3)
        conv4 = layers.Conv2D(8*self._filter_num, (self._filter_size, self._filter_size), strides=2, padding='same')(conv3)
        if self._batch_norm is True:
            conv4 = layers.BatchNormalization(axis=axis)(conv4)
        conv4 = layers.Activation('relu')(conv4)

        conv_final = layers.Flatten()(conv4)
        conv_final = layers.Dense(self._output_c)(conv_final)
        conv_final = layers.Activation('softmax')(conv_final)

        # Model integration
        model = models.Model(inputs, conv_final, name="FCNet")
        return model