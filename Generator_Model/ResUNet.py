# coding: utf-8
'''
TODO:
    Generator model of the domain adversarial network
    Residual U-Net
    or any other encoder-decoder model
'''
__author__ = 'MoleImg'
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K

class ResUNet():
    def __init__(self, input_h, input_w, input_c, output_c, filter_num, filter_size,
                 batch_norm=True):
        self._input_h = input_h
        self._input_w = input_w
        self._input_c = input_c
        self._output_c = output_c
        self._filter_num = filter_num
        self._filter_size = filter_size
        self._batch_norm = batch_norm
        self._build_model()

    def _build_model(self):
        '''
            UNet construction
            convolution: 3*3 SAME padding
            pooling: 2*2 VALID padding
            upsampling: 3*3 VALID padding
            final convolution: 1*1
            :param dropout_rate: FLAG & RATE of dropout.
                    if < 0 dropout cancelled, if > 0 set as the rate
            :param batch_norm: flag of if batch_norm used,
                    if True batch normalization
            :return: UNet model for PACT recons
            '''
        # input data
        # dimension of the image depth
        inputs = layers.Input((self._input_h, self._input_w, self._input_c), dtype=tf.float32)
        axis = 3

        # Subsampling layers
        # double layer 1, convolution + pooling
        conv_128 = self.double_conv_layer(inputs, self._filter_size, self._filter_num, self._batch_norm)
        pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)
        # double layer 2
        conv_64 = self.double_conv_layer(pool_64, self._filter_size, 2*self._filter_num, self._batch_norm)
        pool_32 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)
        # double layer 3
        conv_32 = self.double_conv_layer(pool_32, self._filter_size, 4*self._filter_num, self._batch_norm)
        pool_16 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32)
        # double layer 4
        conv_16 = self.double_conv_layer(pool_16, self._filter_size, 8*self._filter_num, self._batch_norm)
        pool_8 = layers.MaxPooling2D(pool_size=(2, 2))(conv_16)
        # double layer 5, convolution only
        conv_8 = self.double_conv_layer(pool_8, self._filter_size, 16*self._filter_num, self._batch_norm)

        # Upsampling layers
        # double layer 6, upsampling + concatenation + convolution
        up_16 = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(conv_8)
        up_16 = layers.concatenate([up_16, conv_16], axis=axis)
        up_conv_16 = self.double_conv_layer(up_16, self._filter_size, 8*self._filter_num, self._batch_norm)
        # double layer 7
        up_32 = layers.concatenate(
            [layers.UpSampling2D(size=(2, 2), data_format="channels_last")(up_conv_16), conv_32],
            axis=axis)
        up_conv_32 = self.double_conv_layer(up_32, self._filter_size, 4*self._filter_num, self._batch_norm)
        # double layer 8
        up_64 = layers.concatenate(
            [layers.UpSampling2D(size=(2, 2), data_format="channels_last")(up_conv_32), conv_64],
            axis=axis)
        up_conv_64 = self.double_conv_layer(up_64, self._filter_size, 2*self._filter_num, self._batch_norm)
        # double layer 9
        up_128 = layers.concatenate(
            [layers.UpSampling2D(size=(2, 2), data_format="channels_last")(up_conv_64), conv_128],
            axis=axis)
        up_conv_128 = self.double_conv_layer(up_128, self._filter_size, self._filter_num, self._batch_norm)

        # 1*1 convolutional layers
        # valid padding
        # batch normalization
        # sigmoid nonlinear activation
        conv_final = layers.Conv2D(self._output_c, kernel_size=(1, 1))(up_conv_128)
        conv_final = layers.BatchNormalization(axis=axis)(conv_final)
        conv_final = layers.Activation('relu')(conv_final)

        # Model integration
        model = models.Model(inputs, conv_final, name="ResUNet")
        return model

    def double_conv_layer(self, x, filter_size, size, batch_norm):
        '''
        construction of a double convolutional layer using
        SAME padding
        RELU nonlinear activation function
        :param x: input
        :param filter_size: size of convolutional filter
        :param size: number of filters
        :param dropout: FLAG & RATE of dropout.
                if < 0 dropout cancelled, if > 0 set as the rate
        :param batch_norm: flag of if batch_norm used,
                if True batch normalization
        :return: output of a double convolutional layer
        '''
        axis = 3
        conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
        if batch_norm is True:
            conv = layers.BatchNormalization(axis=axis)(conv)
        conv = layers.Activation('relu')(conv)
        conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(conv)
        if batch_norm is True:
            conv = layers.BatchNormalization(axis=axis)(conv)
        conv = layers.Activation('relu')(conv)

        shortcut = layers.Conv2D(size, kernel_size=(1, 1), padding='same')(x)
        if batch_norm is True:
            shortcut = layers.BatchNormalization(axis=axis)(shortcut)

        res_path = layers.add([shortcut, conv])
        return res_path